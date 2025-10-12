import json
import time
from typing import Annotated, List

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel
from io import BytesIO
import base64

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OpenAIService(BaseService):
    openai_base_url: Annotated[
        str, "The base url to use for OpenAI-like models.  No trailing slash."
    ] = "https://api.openai.com/v1"
    openai_model: Annotated[str, "The model name to use for OpenAI-like model."] = (
        "gpt-4o-mini"
    )
    openai_api_key: Annotated[
        str, "The API key to use for the OpenAI-like service."
    ] = None
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "png"

    max_image_width: int = 896   # max width for vision LLM (OpenAI compatible api)
    max_image_height: int = 896  # max height for vision LLM (OpenAI compatible api)

    # per-model resolution presets (add as needed)
    model_resolutions = {
        "gpt-4o": (1024, 1024),
        "gpt-4o-mini": (896, 896),
        "gpt-4o-mini-vision": (896, 896),  # example alias
    }

    def get_model_resolution(self):
        """Return (max_width, max_height) for the current model."""
        return self.model_resolutions.get(
            self.openai_model,
            (self.max_image_width, self.max_image_height),  # fallback
        )

    def format_image_for_llm(self, image: Image.Image | list[Image.Image] | None):
        """Resize images to fit within model-specific resolution (keeping aspect ratio, no upscale)."""
        if image is None:
            return []

        if isinstance(image, Image.Image):
            image = [image]

        max_w, max_h = self.get_model_resolution()

        processed_images = []
        for img in image:
            img = self.resize_for_llm(img, max_w, max_h)
            processed_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{self.openai_image_format};base64,{self.img_to_base64(img)}",
                },
            })
        return processed_images

    def resize_for_llm(self, img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        """Resize so width ╬ô├½├▒ max_w and height ╬ô├½├▒ max_h.
        Keeps aspect ratio, never upscales."""
        w, h = img.size

        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        return img

    def img_to_base64(self, img: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buf = BytesIO()
        img.save(buf, format=self.openai_image_format.upper())
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        img_fmt, self.img_to_base64(img, format=img_fmt)
                    ),
                },
            }
            for img in images
        ]

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout * 2 # double timeout for vision model
        else:
            timeout = timeout * 2
        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/hungle-i3/marker",
                    },
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                # Rate limit exceeded
                if tries == total_tries:
                    # Last attempt failed. Give up
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"OpenAI inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)

# /digix/namiq/processing/llm_image_description.py

from pydantic import BaseModel
from typing import Annotated, List

from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData
from marker.schema import BlockTypes
from marker.schema.document import Document


class LLMImageDescriptionProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (
        BlockTypes.Picture,
        BlockTypes.Figure,
    )
    extract_images: Annotated[bool, "Extract images from the document."] = True
    image_description_prompt: Annotated[
        str,
        "The prompt to use for generating image descriptions.",
        "Default is a string containing the Gemini prompt.",
    ] = """You are a digital archaeologist and visual data engineer. Your function is to meticulously deconstruct any given visual into a structured text description that serves as a blueprint for its complete and lossless digital reconstruction.

**Core Directive:**
You will be given an image and optional text (`{raw_text}`). Your task is to first classify the image, then generate a comprehensive, multi-part description based on that classification. The description must be precise enough for a machine to reconstruct the original visual.

**Instructions & Output Format:**

**Instructions:**
1.  **Classify the Image:** Begin your response with the classification on a single line: `**Image Type:** [Category]`. Choose from: `Chart`, `Diagram`, `Software_UI_or_Website`, `Photograph`, `Logo`, `Map`, `Illustration`, `Document Scan`, `Other`.
2.  **Follow the Correct Structure:** Based on the classification, provide a detailed description using the corresponding Markdown structure below. Use bolded text for section titles, not Markdown headings (###).
3.  **Be Precise and Quantitative:** For every element/component, identify it and its specific visual styling (colors, fonts, line weights, etc.).
4.  **Use Image as Ground Truth:** The image is the primary source. Use the provided `{raw_text}` only to verify and correct text details.
5.  **[CRITICAL] Verify All Critical Values:** For any text or number found inside a dense graphical element (like a **Feature Control Frame** or a **Note Callout** triangle), you MUST perform a secondary check. First, identify the value visually. Second, search the `{raw_text}` for that value near the symbol's location to confirm your reading. This is especially important for **tolerance values**. Report only the confirmed value.

---
**Output Structures by Image Type:**

**If Image Type is `Chart`:**
**Overall Summary:** A concise, one-sentence summary of the chart.
**Visual Composition:** A description of the layout, background, and key visual blocks.
**Detailed Analysis:**
* **Title:** "[Title Text]"
    * **Location:** Relative position in the chart
    * **Style:** [Font size, color (hex), weight (bold, normal)]
* **X-Axis:**
    * **Label:** "[Label Text]"
    * **Location:** Relative position in the chart
    * **Style:** [Font and line styles]
* **Y-Axis:**
    * **Label:** "[Label Text]"
    * **Location:** Relative position in the chart
    * **Style:** [Font and line styles]
* **Legend:**
    * **Location:** Relative position in the chart)
    * **Items:**
        * **[Series 1 Name]:** Style: [Description of color, marker, line style].
* **Data Series:**
    * **[Series 1 Name]:**
        * **Style:** [e.g., "Solid, 2px-thick blue (#0000FF) line with 4px circular markers"]
        * **Data:** [(x1, y1), (x2, y2), ...]
**Key insights:** [Key insights extract from chart].

**If Image Type is `Diagram_or_Schematic`:**
**Overall Summary:** A concise, one-sentence summary of the diagram/schematic.
**Component List:**
* **[Component 1 ID] Component Name:**
    * **BoundingBox:** (x_min, y_min, x_max, y_max)
    * **Type:** [Group / UML Notation Name / GD&T Symbol Name / Geometric Tolerances]
    * **Text:** [Label/text/note/GD&T symbol description extracted at the component]
    * **Description:** [Details of the component].
    * **Style:** [Fill color, border color, border style, font style of label]
* **[Component 2 Id] Component Name:**
**Connections:**
* **Connection 1:**
    * **From:** [Component 1 ID]
    * **To:** [Component 2 Name]
    * **Style:** [e.g., "Solid 1px black line with a filled triangular arrowhead at the 'To' end"]
    * **Label:** "[Associated Text]"
**Mermaid diagram markdown:** A complete markdown representive the diagram for Mermaid visualization with correct grammars.

**If Image Type is `Software_UI_or_Website`:**
**Overall Summary:** A concise summary of the UI's purpose (e.g., "A user login screen for a web application," "The main dashboard of a data analytics software.").
**Content Analysis:** A brief analysis of the primary content types on the screen (e.g., "The screen is dominated by a user registration form and a header navigation bar.", "The main content is a data table showing financial results.").
**Layout-Preserving Description:** A hierarchical breakdown of the UI. **If a form or table is present, its structure must be described by recreating its layout.**
* **For Forms:** Describe fields in their logical order. Use nesting for grouped elements.
    * **Form: 'User Registration'**
        * **Label: 'First Name'**
            * **Type:** [Text Input]
            * **BoundingBox:** (x_min, y_min, x_max, y_max)
            * **Style:** [Placeholder text, border style]
        * **Label: 'Email Address'**
            * **Type:** [Text Input]
            * **BoundingBox:** (x_min, y_min, x_max, y_max)
            * **Style:** [Placeholder text, border style]
        * **Button: 'Create Account'**
            * **Type:** [Button]
            * **BoundingBox:** (x_min, y_min, x_max, y_max)
            * **Style:** [e.g., "Blue (#3366CC) background, white text, 14pt sans-serif"]
* **For Data Tables:** Recreate the table using Markdown, including headers. Add notes for styling.
    * **Table: 'Recent Orders'**
        * **BoundingBox:** (x_min, y_min, x_max, y_max)
        * **Style Notes:** [e.g., "Header row has a light gray background. Alternating row colors."]
        * **Markdown Table:**
| Order ID | Product Name | Date | Status |
| :--- | :--- | :--- | :--- |
| #7821 | Wireless Mouse | 2025-07-08 | Shipped |
| #7820 | USB-C Hub | 2025-07-08 | Processing |
| #7819 | Keyboard | 2025-07-07 | Delivered |
* **For Other Components (Headers, Navbars, etc.):**
    * **Header:**
        * **BoundingBox:** (x_min, y_min, x_max, y_max)
        * **Contains:**
            * **Logo:** Type: [Image], BoundingBox: ..., Style: ...
            * **Navigation Menu:** Type: [Menu], BoundingBox: ..., Style: ...
**Key insights:** Key insights/functionalities extract from the extracted content.

**If Image Type is `Photograph`:**
**Overall Summary:** A concise, one-sentence summary of the photograph with insight like sentiment, atmotphere, context.
**Subject and Setting:** A description of the main subject(s) and environment.
* **[Main Subject]:** BoundingBox: (x_min, y_min, x_max, y_max)
* **[Secondary Object]:** BoundingBox: (x_min, y_min, x_max, y_max)
**Composition and Lighting:** A description of the visual composition and lighting.
**Prompt:** A complete and detailed prompt that describe the picture accurately to re-generate the image by a text2image tool.

**If Image Type is `Logo`, `Map`, `Illustration` or `Other`:**
**Overall Summary:** A concise summary of the image including background and foreground objects with insight like sentiment, atmotphere, context.
**Detected Entities:**
* **[Entity ID] Entiry name:**
    * **BoundingBox:** (x_min, y_min, x_max, y_max)
    * **Style:** [Fill color, stroke, style, etc.]
    * **Entiry type:** [Object type / Object classification]
    * **Description:** [A clear description of the entity].
**Relationships:**
* **Relationship 1:**
    * **From:** [Entity ID 1]
    * **To:** [Entity ID 2]
    * **Description:** [e.g., "Arrow indicates process flow", "Text label for Entity 1"]
**Prompt:** A complete and detailed prompt that describe the picture accurately to re-generate the image by a text2image tool.
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)
        if self.extract_images:
            return []
        return blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            prompt = self.image_description_prompt.replace(
                "{raw_text}", block.raw_text(document)
            )
            image = self.extract_image(document, block)

            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block,
                    "schema": ImageSchema,
                    "page": block_data["page"],
                }
            )

        return prompt_data

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        block = prompt_data["block"]

        if not response or "image_description" not in response:
            block.update_metadata(llm_error_count=1)
            return

        image_description = response["image_description"]
        if len(image_description) < 10:
            block.update_metadata(llm_error_count=1)
            return

        block.description = image_description


class ImageSchema(BaseModel):
    image_description: str

import base64
import logging
from pathlib import Path
from typing import List

import openai
from ai_router.logging import llm_log
from pydantic import BaseModel, Field

from scribe.chunking.basechunker import BaseChunker
from scribe.schema import Chunk

logger = logging.getLogger(__name__)


class ImageChunker(BaseChunker):
    def __call__(self, image_folder: Path) -> List[Chunk]:
        return self.process_images(image_folder)

    def process_images(self, image_folder: Path) -> List[Chunk]:
        chunk_list = []
        for i, image_file in enumerate(sorted(image_folder.glob("*.png"))):
            print("#" * 50)
            print(f"Processing image: {image_file.name}")
            try:
                propositions = self.get_propositions_from_image(image_file)
                chunk = self.create_chunk(
                    heading=f"Image Analysis: {image_file.name}",
                    propositions=propositions,
                    idx=i,
                    image_path=str(image_file),
                )
                chunk_list.append(chunk)
            except Exception as e:
                logger.error(f"Error processing image: {image_file.name}")
                logger.error(f"Error: {str(e)}")

        return chunk_list

    def get_propositions_from_image(self, image_file: Path) -> List[str]:
        system_prompt = """Analyze this image and convert ALL information into factual propositions.

# Phase 1: Text-Based Propositions (Priority)
1. Read ALL text in the image using OCR (numbers, dates, headers, body text, labels, footnotes)
2. Convert text directly into self-contained factual statements
3. Combine related information (e.g., "Invoice #12345 dated March 15, 2024 for €1,234.56")
4. Preserve numerical values exactly as shown with units
5. Use Markdown formatting where appropriate

# Phase 2: Visual Element Propositions
After text conversion, describe non-textual elements as factual statements:
- Charts/Graphs: State trends and data points (e.g., "Bar chart shows 15% revenue increase")
- Tables: Describe structure and key values (e.g., "Table contains 5 columns with quarterly data")
- Diagrams/Images: Describe visual content (e.g., "Flowchart shows 3-step approval process")
- Logos/Icons: Note presence and description (e.g., "Company logo 'Acme Corp' in header")

# Output Requirements
- Return as JSON with a list of propositions
- Each proposition = single, complete, self-contained statement
- First: All text-based propositions (complete coverage)
- Then: Visual element propositions
- If image is blurred or empty: return empty list
- Use same language as original text (German text -> German propositions)

# Example Good Propositions
- "Invoice number 12345 issued on 15.03.2024 with total amount of 1,234.56 EUR"
- "Line chart demonstrates revenue growth from 100k EUR to 150k EUR over Q1-Q4"
- "Document header states 'Jahresabschluss 2024' for company XYZ GmbH"

# Example Bad Propositions (Avoid)
- "Invoice" (incomplete)
- "12345" (no context)
- "Chart present" (no information content)

Output format:
Return the result as JSON with a list of propositions."""

        class Propositions(BaseModel):
            propositions: List[str] = Field(description="List of key propositions extracted from the text")

        if self.client is None:
            logger.warning("LLM not available, returning empty propositions")
            return []

        with image_file.open("rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        user_prompt = (
            f"Analyze the following image and extract propositions:\n\n![image](data:image/png;base64,{image_data})"
        )
        try:
            with llm_log("chunker_image", self.model_name, user_prompt=f"[image: {image_file.name}]") as log:
                result, parsed = self.client.invoke(system_prompt, user_prompt, output_schema=Propositions)
                if parsed:
                    log.output = str(parsed.propositions)[:5000]
                    return parsed.propositions
                log.output = result.content[:5000]
        except openai.BadRequestError as e:
            logger.warning(f"Error processing image: {image_file.name}")
            logger.warning(f"Filter cause: {e}")
            return []
        return []

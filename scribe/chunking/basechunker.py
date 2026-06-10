import logging
from typing import List

import openai
from ai_router.client import get_llm_client
from ai_router.logging import llm_log
from django.conf import settings

from scribe.schema import Chunk

logger = logging.getLogger(__name__)


class BaseChunker:
    def __init__(self, model: str = None):
        self.model_name = model or settings.DEFAULT_MODEL_SCRIBE_CHUNKER
        try:
            self.client = get_llm_client(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.client = None

    def create_chunk(
        self,
        section: str = "",
        heading: str = "",
        propositions: List[str] = None,
        idx: int = 0,
        image_path: str = "",
    ) -> Chunk:
        if propositions is None:
            propositions = []
        if not heading:
            heading = self._create_heading_from_section(section) if section else "Image Analysis"

        try:
            content = f"**{heading}**\n\n" + "\n".join(propositions)
        except TypeError as e:
            logger.error(
                f"Error joining propositions: {e}, propositions type: {type(propositions)}, value: {propositions}"
            )
            raise
        metadata = {
            "source": "processed_pdf" if section else "image_analysis",
            "idx": idx,
            "page_number": idx + 1,
            "propositions_count": len(propositions),
            "raw_section": section if section else "",
            "image_path": image_path if image_path else "",
        }

        if section:
            metadata["raw_section"] = section
        if image_path:
            metadata["image_path"] = image_path

        return Chunk(
            splits=[content],
            is_triggered=False,
            token_count=len(content.split()),
            metadata=metadata,
        )

    def _create_heading_from_section(self, section: str) -> str:
        system_prompt = """You are an AI trained to create a heading from a section of text which may include a table.

Guidelines:
- Generate a heading that is a single, self-contained statement.
- Capture the main topic of the propositions.
- The heading should be concise and informative, without going into too much detail.
- The heading should be in the same language as the section.
- If the section contains a table, look at the first column and often-used words or sub-headings and create a heading
from that, if you find a common theme."""

        if self.client is None:
            logger.warning("LLM not available, returning empty heading")
            return ""

        user_prompt = f"Section: {section}"
        try:
            with llm_log("chunker_heading", self.model_name, user_prompt=section[:2000]) as log:
                result, _ = self.client.invoke(system_prompt, user_prompt)
                log.output = result.content
                log.input_tokens = result.input_tokens
                log.output_tokens = result.output_tokens
        except openai.BadRequestError as e:
            logger.warning(f"Error processing section: {section}")
            logger.warning(f"Filter cause: {e}")
            return ""
        return result.content

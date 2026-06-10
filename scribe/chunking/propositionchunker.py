import logging
import re
from typing import List

import openai
from ai_router.logging import llm_log
from pydantic import BaseModel, Field

from scribe.chunking.basechunker import BaseChunker
from scribe.schema import Chunk

logger = logging.getLogger(__name__)


class PropositionChunker(BaseChunker):
    def __call__(self, chunk_list: List[Chunk], md_string: str) -> List[Chunk]:
        return self.process_text_content(md_string, chunk_list)

    def process_text_content(self, md_string: List[str], chunk_list: List[Chunk]) -> List[Chunk]:
        pages = self._split_at_page_border(md_string, merge_threshold=800)

        for i, page_text in enumerate(pages):
            print(f"Page: {i+1}")

            assert len(page_text) <= 65535, "Text length exceeds Milvus limit"
            propositions = self.get_propositions_from_text(page_text)
            chunk = self.create_chunk(section=page_text, propositions=propositions, idx=i)
            chunk_list.append(chunk)

        return chunk_list

    @staticmethod
    def _split_at_page_border(doc: str, merge_threshold: int) -> List[str]:
        pages = re.split(r"\n\s*</?page\s*\d*>\s*\n", doc)
        if len(pages) <= 1:
            pages = re.split(r"\n\s*-----+\s*\n", doc)
        if len(pages) <= 1 or any(len(page) > 5000 for page in pages):
            pattern = r"(#{1,6}\s+.+?(?:\n|$))"  # Matches markdown headings (e.g., # Heading, ## Subheading)
            sections = re.split(pattern, doc)
            result = []
            current_section = ""

            for i in range(len(sections)):
                if re.match(r"#{1,6}\s+", sections[i]):
                    if current_section.strip():
                        result.append(current_section.strip())
                    current_section = sections[i]
                else:
                    current_section += sections[i]

            if current_section.strip():
                result.append(current_section.strip())

            combined_result = []
            i = 0
            while i < len(result):
                current_chunk = result[i]
                while len(current_chunk) < merge_threshold and i + 1 < len(result):
                    current_chunk += "\n\n" + result[i + 1]
                    i += 1
                combined_result.append(current_chunk.strip())
                i += 1
            return [section for section in combined_result if len(section.strip()) > 10]
        else:
            return [page.strip() for page in pages if page.strip()]

    def get_propositions_from_text(self, text: str) -> List[str]:
        system_prompt = """You are an AI trained to extract key propositions from
text. Your task is to create concise, factual statements that capture the main ideas and important details of the given
text.

Guidelines:
- If a row states several values for a single proposition, create a single proposition that states all
the values.
- Each proposition should be a single, self-contained statement.
- Focus on the most important and relevant information.
- All numbers should be displayed in total value, not in thousands or millions, and contain the unit of measurement
(if applicable).
- For all text, use the same language of the original text, e.g. the propositions should be German if the text is in
German.

Output format:
Return the result as JSON with a list of propositions."""

        class Propositions(BaseModel):
            propositions: List[str] = Field(description="List of key propositions extracted from the text")

        user_prompt = f"Text: {text}"
        try:
            with llm_log("chunker_propositions", self.model_name, user_prompt=text[:2000]) as log:
                result, parsed = self.client.invoke(system_prompt, user_prompt, output_schema=Propositions)
                if parsed:
                    log.output = str(parsed.propositions)[:5000]
                    return parsed.propositions
                log.output = result.content[:5000]
        except openai.BadRequestError as e:
            logger.warning(f"Error processing text: {text}")
            logger.warning(f"Filter cause: {e}")
            return []
        return []

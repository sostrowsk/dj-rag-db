import logging

from ai_router.client import get_llm_client
from ai_router.logging import llm_log
from ai_router.types import Document
from django.conf import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert at providing concise context for document chunks.
Your task is to provide a brief, informative context that situates this chunk within the whole document.
This context will be prepended to the chunk to improve search retrieval accuracy.

Guidelines:
- Keep the context short (50-100 tokens)
- Include essential information like document type, topic, and relevant entities
- Identify the section/part of the document this chunk belongs to
- Mention any key dates, numbers, or entities relevant to understanding this chunk
- If there are important terms defined elsewhere in the document that relate to this chunk, mention them
- DO NOT repeat the chunk's content verbatim
- DO NOT add any opinions or interpretations, just factual context
- Answer ONLY with the context text, nothing else"""


class DocumentContextualizer:

    def __init__(self, model: str = None):
        self.model_name = model or settings.DEFAULT_MODEL_SCRIBE_CONTEXTUALIZER
        try:
            self.client = get_llm_client(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.client = None

    def contextualize_chunk(self, chunk: Document, whole_document: str) -> Document:
        try:
            safe_document = whole_document[:100000] if len(whole_document) > 100000 else whole_document
            safe_chunk = chunk.page_content[:10000] if len(chunk.page_content) > 10000 else chunk.page_content
            if self.client is None:
                logger.warning("LLM not available, returning original chunk")
                return chunk

            user_prompt = f"""<document>
{safe_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{safe_chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving
search retrieval of the chunk. Write in the language of the overall document.
"""
            with llm_log("contextualizer", self.model_name, user_prompt=safe_chunk[:2000]) as log:
                result, _ = self.client.invoke(SYSTEM_PROMPT, user_prompt)
                context_text = result.content.strip()
                log.output = context_text
                log.input_tokens = result.input_tokens
                log.output_tokens = result.output_tokens
            contextualized_content = f"<context>\n{context_text}\n</context>\n\n{chunk.page_content}"
            new_metadata = chunk.metadata.copy() if chunk.metadata else {}
            new_metadata["has_context"] = True
            new_metadata["context_length"] = len(context_text.split())
            return Document(page_content=contextualized_content, metadata=new_metadata)
        except Exception as e:
            logger.error(f"Error contextualizing chunk: {str(e)}")
            return chunk

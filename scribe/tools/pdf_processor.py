# scribe/tools/pdf_processor.py
import hashlib
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List

from ai_router.types import Document
from django.conf import settings
from progress.consumers import send_websocket_update

from scribe.schema import Chunk
from scribe.utils import tiktoken_length

from ..chunking.markdownchunker import split_markdown_at_headings
from ..chunking.propositionchunker import PropositionChunker
from .ocr_processor import OCRProcessor, OCRSettings

logger = logging.getLogger(__name__)


def build_chunk_documents(valid_chunks, document) -> List[Document]:
    """Baut die Document-Objekte fuer die Indexierung aus (idx, Chunk)-Tupeln.

    chunk_id ist IMMER der globale Listen-Index: Image-Chunks bringen ein
    eigenes, bei 0 startendes ``idx`` in ihrer Metadata mit, das mit den
    Text-Chunk-Indizes kollidiert — unter dem UniqueConstraint
    (collection_name, document_id, chunk_id) wuerden solche Duplikate beim
    bulk_create(ignore_conflicts=True) still verworfen.
    """
    if hasattr(document, "project") and document.project:
        parent_id = document.project.id
        parent_key = "project_id"
    elif hasattr(document, "client") and document.client:
        parent_id = document.client.id
        parent_key = "client_id"
    else:
        parent_id = 0
        parent_key = "project_id"

    return [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": idx,
                "document_id": document.id,
                parent_key: parent_id,
                "raw_section": (chunk.metadata.get("raw_section", "") if chunk.metadata else ""),
                "image_path": (chunk.metadata.get("image_path", "") if chunk.metadata else ""),
                "page_number": (chunk.metadata.get("page_number", 0) if chunk.metadata else 0),
                "full_document_text_available": True,
                "has_context": False,
                "original_content": "",
                **({"document_path": document.file.path} if hasattr(document, "file") and document.file else {}),
            },
        )
        for idx, chunk in valid_chunks
    ]


class PDFProcessor:
    def __init__(self, scribe_instance):
        self.scribe = scribe_instance
        self.ocr_settings = OCRSettings()
        self.ocr_processor = OCRProcessor(self.ocr_settings)

    def process_pdf(self, document, image_folder=None, process_images=True, user_id=None) -> List[Document]:
        if settings.DISABLE_IMAGE_PROPOSITIONS:
            logger.info("Image propositions disabled via DISABLE_IMAGE_PROPOSITIONS setting")
            process_images = False
        if not document.file or (not document.file.name.lower().endswith(".pdf") and not document.markdown):
            file_name = document.file.name if document.file else "No file"
            error_msg = f"This function only handles PDF documents. Got: {file_name}"
            logger.error(error_msg)
            document.markdown = f"Error: {error_msg}"
            document.save(update_fields=["markdown"])
            raise ValueError(error_msg)
        if not document.file.storage.exists(document.file.name):
            error_msg = f"File does not exist: {document.file.name}"
            logger.error(error_msg)
            document.markdown = f"Error: {error_msg}"
            document.save(update_fields=["markdown"])
            raise FileNotFoundError(error_msg)
        if not image_folder:
            unique_id = f"{uuid.uuid4()}{time.time()}"
            temp_hash = hashlib.sha256(unique_id.encode()).hexdigest()
            temp_dir_name = f"tmp_images_{temp_hash}"
            image_folder = Path(os.path.join(settings.BASE_DIR, temp_dir_name))
        try:
            # Use pre-extracted markdown if available (from extract_markdown_task)
            if document.markdown:
                markdown_content = document.markdown
                logger.info(f"Using existing document.markdown ({len(markdown_content)} chars), skipping OCR")
            else:
                logger.info("Converting PDF to Markdown with OCR preprocessing...")
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                for image_file in image_folder.glob("*.png"):
                    image_file.unlink()
                markdown_content = self.ocr_processor.extract_markdown(document.file.path, image_folder)
                if markdown_content:
                    document.markdown = markdown_content
                    document.save(update_fields=["markdown"])
                    logger.info(f"Saved full document text ({len(markdown_content)} chars) to document.markdown")
                else:
                    logger.info("Failed to extract Markdown")
            logger.info("Chunking document...")
            if user_id:
                send_websocket_update(user_id=user_id, message="Chunking document...", action="update")
                time.sleep(0.5)  # Brief delay to ensure visibility
            image_chunks = []
            if process_images and image_folder and os.path.exists(image_folder):
                logger.info("Processing images...")
                if user_id:
                    send_websocket_update(user_id=user_id, message="Processing images...", action="update")
                    time.sleep(0.5)  # Brief delay to ensure visibility
                image_chunks = self.scribe.image_chunker(image_folder)
            else:
                logger.info(f"Skipping image processing (process_images={process_images})")
            text_chunks = []
            if isinstance(self.scribe.chunker, PropositionChunker):
                text_chunks = self.scribe.chunker(text_chunks, markdown_content)
            else:
                section_list = split_markdown_at_headings(markdown_content)
                section = ""
                section_chunks = []
                for item in section_list:
                    section += item
                    section_token_count = tiktoken_length(section)
                    logger.info(f"Section tokens: {section_token_count}")
                    if section_token_count < self.scribe.embedding_size / 5:
                        continue
                    elif section_token_count > self.scribe.embedding_size:
                        doc_list = self.scribe.chunker([section])
                        for doc in doc_list:
                            text_chunks.extend(doc)
                    else:
                        section_chunks.append(section)
                    section = ""
                if len(section) > 0:
                    section_chunks.append(section)
                if len(section_chunks) > 0:
                    for section in section_chunks:
                        chunk = Chunk(
                            splits=[section],
                            is_triggered=False,
                            token_count=section_token_count,
                            content=section,
                        )
                        text_chunks.append(chunk)
            chunk_list = text_chunks + image_chunks
            logger.info(f"Number of chunks generated: {len(chunk_list)}")
            logger.info("Creating Document objects...")
            if user_id:
                send_websocket_update(
                    user_id=user_id,
                    message="Creating Document objects...",
                    action="update",
                )
                time.sleep(0.5)  # Brief delay to ensure visibility

            # Filter out chunks with empty text (uses chunk.text which falls back to joining splits)
            valid_chunks = [(idx, chunk) for idx, chunk in enumerate(chunk_list) if chunk.text]
            if len(valid_chunks) < len(chunk_list):
                logger.warning(f"Filtered out {len(chunk_list) - len(valid_chunks)} chunks with empty content")

            doc_list = build_chunk_documents(valid_chunks, document)
            if hasattr(document, "markdown"):
                logger.info(f"Document text already saved in document.markdown ({len(document.markdown)} chars)")
            else:
                logger.info("Document text not saved in document model, using metadata approach")
            logger.info(f"Number of documents to add: {len(doc_list)}")
            return doc_list
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        finally:
            if os.path.exists(image_folder):
                for image_file in image_folder.glob("*.png"):
                    image_file.unlink()
                os.rmdir(image_folder)

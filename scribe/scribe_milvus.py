# scribe/scribe_milvus.py
"""SCRIBE facade: chunking/contextualization + Postgres-SSOT indexing + search.

Phase A6 refactor: the facade no longer talks to Milvus via langchain.
- Indexing always writes ``DocumentChunk`` rows via :class:`PgvectorBackend`
  (delete-then-insert per document = idempotent re-index). When
  ``settings.VECTORSTORE_BACKEND == "milvus"`` the same pre-computed
  embeddings are mirrored into Milvus via :class:`MilvusBackend`.
- Search embeds the query once, delegates to the configured backend
  (hybrid dense + sparse with RRF fusion) and applies the adaptive cutoff
  over the fused scores (settings ``VECTORSTORE_*``).
- delete/drop always hit Postgres; Milvus is best-effort when configured.

Public API names are kept for call-site compatibility: ``process_pdf``,
``add_documents_to_collection``, ``search_similar_chunks``,
``delete_documents``, ``drop_collection``, ``check_milvus_health[_static]``,
``close``.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Union

from ai_router.azure_client import openai_embeddings, openai_encoder
from ai_router.types import Document
from asgiref.sync import sync_to_async
from django.conf import settings
from progress.consumers import send_websocket_update_async

from scribe.backends import ChunkRecord, SearchFilter, get_search_backend
from scribe.backends.milvus_backend import MilvusBackend
from scribe.backends.pgvector_backend import PgvectorBackend
from scribe.chunking.imagechunker import ImageChunker
from scribe.processing.contextualizer import DocumentContextualizer
from scribe.retrieval import adaptive_cutoff

from .chunking.statisticalchunker import StatisticalChunker_GaussianSmoothing
from .tools.md_regex_splitter import MDRegexSplitter
from .tools.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

SearchResults = List[Tuple[Document, float]]


class SCRIBE:
    def __init__(
        self,
        collection_name: str,
        min_chunk_tokens: Optional[int] = None,
        max_chunk_tokens: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.encoder = openai_encoder()
        self.embeddings = openai_embeddings()
        self.embedding_size = self._get_embedding_size()
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.chunker = self._setup_statistical_chunker()
        self.image_chunker = ImageChunker(model=settings.DEFAULT_MODEL_SCRIBE_IMAGE_CHUNKER)
        self.contextualizer = DocumentContextualizer(model=settings.DEFAULT_MODEL_SCRIBE)
        self.pdf_processor = PDFProcessor(self)
        self.pg_backend = PgvectorBackend()
        self.search_backend = get_search_backend()
        self.use_contextual_retrieval = getattr(settings, "SCRIBE_USE_CONTEXTUAL_RETRIEVAL", True)
        logger.debug(
            f"Initializing SCRIBE: collection={self.collection_name} "
            f"backend={settings.VECTORSTORE_BACKEND} contextual={self.use_contextual_retrieval}"
        )

    @classmethod
    async def create(
        cls,
        collection_name: str,
        min_chunk_tokens: Optional[int] = None,
        max_chunk_tokens: Optional[int] = None,
    ):
        return await asyncio.to_thread(cls, collection_name, min_chunk_tokens, max_chunk_tokens)

    def _setup_statistical_chunker(self) -> StatisticalChunker_GaussianSmoothing:
        # Use instance variables if provided, otherwise fall back to settings
        min_split_tokens = self.min_chunk_tokens or getattr(settings, "SCRIBE_MIN_CHUNK_TOKENS", 500)
        max_split_tokens = self.max_chunk_tokens or getattr(
            settings, "SCRIBE_MAX_CHUNK_TOKENS", int(self.embedding_size * 0.95)
        )

        logger.info(f"Chunker config: min_split_tokens={min_split_tokens}, max_split_tokens={max_split_tokens}")

        return StatisticalChunker_GaussianSmoothing(
            encoder=self.encoder,
            splitter=MDRegexSplitter(),
            name="statistical_chunker",
            min_split_tokens=min_split_tokens,
            max_split_tokens=max_split_tokens,
            dynamic_threshold=True,
            window_size=5,
        )

    # Embedding dim is a property of the model — hardcode known ones so
    # SCRIBE() construction stays usable when the embedding endpoint is
    # unreachable (e.g. during index removal, which generates no new
    # embeddings).
    _KNOWN_EMBEDDING_DIMS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def _get_embedding_size(self) -> int:
        model = getattr(self.embeddings, "model", None)
        if model in self._KNOWN_EMBEDDING_DIMS:
            return self._KNOWN_EMBEDDING_DIMS[model]
        sample_text = "This is a sample text to determine the embedding size."
        sample_embedding = self.embeddings.embed_query(sample_text)
        embedding_size = len(sample_embedding)
        assert embedding_size < 32768, "Embedding size exceeds Milvus limit"
        logger.info(f"Detected embedding size via probe: {embedding_size}")
        return embedding_size

    # -- backend plumbing --------------------------------------------------

    def _milvus_mirror_backend(self) -> Optional[MilvusBackend]:
        """Milvus backend used as derived index during writes.

        Only active when Milvus is THE configured search backend — then a
        failing Milvus write must fail the indexing run (the search index
        would silently miss chunks otherwise).
        """
        if settings.VECTORSTORE_BACKEND == "milvus":
            return self.search_backend
        return None

    def _milvus_best_effort_backend(self) -> Optional[MilvusBackend]:
        """Milvus backend for best-effort cleanup (delete/drop), if configured."""
        if settings.VECTORSTORE_BACKEND == "milvus":
            return self.search_backend
        if getattr(settings, "MILVUS_HOST", None):
            return MilvusBackend(embedding_dim=self.embedding_size)
        return None

    # -- health / initialization -------------------------------------------

    def check_milvus_health(self, keep_connection: bool = False) -> bool:
        """Backend-agnostic readiness check (name kept for API compat)."""
        return self.search_backend.is_ready()

    @classmethod
    def check_milvus_health_static(cls) -> dict:
        """Static Milvus health check for external health endpoints."""
        return MilvusBackend.health_check()

    def initialize_collection(self) -> dict:
        """Ensure the namespace is usable on the configured backend."""
        try:
            if not self.search_backend.is_ready():
                logger.error(f"Search backend '{settings.VECTORSTORE_BACKEND}' is not ready")
                return {"success": False, "existed": False, "entity_count": None}
            existed = True
            if settings.VECTORSTORE_BACKEND == "milvus":
                existed = not self.search_backend.ensure_collection(self.collection_name)
            entity_count = self.search_backend.count(SearchFilter(collection_name=self.collection_name))
            return {"success": True, "existed": existed, "entity_count": entity_count}
        except Exception as e:
            logger.error(f"Error initializing collection '{self.collection_name}': {e}")
            return {"success": False, "existed": False, "entity_count": None}

    # -- contextualization ---------------------------------------------------

    async def contextualize_chunks(self, chunks: List[Document], document_text: str) -> List[Document]:
        if not self.use_contextual_retrieval:
            logger.info("Contextual retrieval disabled, skipping contextualization")
            return chunks
        if not document_text:
            logger.warning("No document text provided for contextualization")
            return chunks
        max_chunks_to_contextualize = getattr(settings, "SCRIBE_MAX_CHUNKS_TO_CONTEXTUALIZE", 50)
        if len(chunks) > max_chunks_to_contextualize:
            logger.warning(
                f"Too many chunks ({len(chunks)}), "
                f"limiting contextualization to first {max_chunks_to_contextualize} chunks"
            )
            chunks_to_process = chunks[:max_chunks_to_contextualize]
            remaining_chunks = chunks[max_chunks_to_contextualize:]
            for chunk in remaining_chunks:
                chunk.metadata["has_context"] = False
        else:
            chunks_to_process = chunks
            remaining_chunks = []
        try:
            logger.info(f"Contextualizing {len(chunks_to_process)} chunks with document of length {len(document_text)}")
            batch_size = getattr(settings, "SCRIBE_CONTEXTUALIZATION_BATCH_SIZE", 10)
            contextualized = []
            for i in range(0, len(chunks_to_process), batch_size):
                batch = chunks_to_process[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(chunks_to_process) + batch_size - 1)//batch_size}"
                )
                batch_results = await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            self.contextualizer.contextualize_chunk,
                            chunk,
                            document_text,
                        )
                        for chunk in batch
                    ]
                )
                contextualized.extend(batch_results)
            return contextualized + remaining_chunks
        except Exception as e:
            logger.error(f"Error in contextualization process: {str(e)}")
            return chunks

    # -- indexing --------------------------------------------------------------

    async def add_documents_to_collection(
        self,
        documents: List[Document],
        batch_size: int = 100,
        document_text: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        user_id: Optional[int] = None,
    ) -> None:
        if not documents:
            logger.warning("No valid documents to add to the collection.")
            return

        if not document_text and documents and len(documents) > 0:
            doc_id = documents[0].metadata.get("document_id")
            if doc_id:
                try:
                    from scribe.conf import get_project_document_model

                    def get_document(doc_id):
                        try:
                            return get_project_document_model().objects.get(id=doc_id)
                        except Exception:
                            return None

                    doc = await asyncio.to_thread(get_document, doc_id)
                    if doc and hasattr(doc, "markdown") and doc.markdown:
                        document_text = doc.markdown
                        logger.info(
                            f"Retrieved document text from document model (id: {doc_id}, "
                            f"length: {len(document_text)})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to retrieve document text from model: {str(e)}")
        if self.use_contextual_retrieval:
            if document_text:
                logger.info(f"Applying contextual enrichment to {len(documents)} chunks...")
                if user_id:
                    await send_websocket_update_async(
                        user_id=user_id,
                        message=f"Applying contextual enrichment to {len(documents)} chunks...",
                        action="update",
                    )
                    await asyncio.sleep(0.5)  # Brief delay to ensure visibility
                try:
                    contextualized_docs = await self.contextualize_chunks(documents, document_text)
                    context_count = sum(1 for d in contextualized_docs if d.metadata.get("has_context", False))
                    logger.info(f"Successfully contextualized {context_count} of {len(contextualized_docs)} chunks")
                    documents = contextualized_docs
                except Exception as e:
                    logger.error(f"Error during contextualization: {str(e)}")
            else:
                logger.warning("No document text available for contextual enrichment - will index without context")
        else:
            logger.info("Contextual retrieval disabled")

        await self._embed_and_store(
            documents,
            batch_size=batch_size,
            progress_callback=progress_callback,
            user_id=user_id,
        )

    async def _embed_and_store(
        self,
        documents: List[Document],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
        user_id: Optional[int] = None,
    ) -> None:
        """Embed chunk contents once (batched) and persist them.

        Postgres (``DocumentChunk``) is the source of truth and always written.
        When Milvus is the active search backend, the same embeddings are
        mirrored there. Existing rows per document are deleted first so a
        re-index is idempotent.
        """
        milvus_backend = self._milvus_mirror_backend()
        insert_filters = SearchFilter(collection_name=self.collection_name)

        document_ids = sorted(
            {d.metadata.get("document_id") for d in documents if d.metadata.get("document_id") is not None}
        )
        for doc_id in document_ids:
            delete_filters = SearchFilter(collection_name=self.collection_name, document_id=doc_id)
            await sync_to_async(self.pg_backend.delete)(delete_filters)
            if milvus_backend:
                await asyncio.to_thread(milvus_backend.delete, delete_filters)

        total_batches = (len(documents) + batch_size - 1) // batch_size
        processed_batches = 0

        logger.info(f"Adding {len(documents)} chunks to {self.collection_name} in {total_batches} batches...")
        if user_id:
            await send_websocket_update_async(
                user_id=user_id,
                message=f"Starting batch processing: 0/{total_batches} batches complete",
                action="update",
            )
            await asyncio.sleep(0.5)  # Brief delay to ensure visibility

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            if user_id and processed_batches == 0:  # Only show once for first batch
                await send_websocket_update_async(user_id=user_id, message="Creating embeddings...", action="update")
                await asyncio.sleep(0.5)  # Brief delay to ensure visibility

            embeddings = await asyncio.to_thread(self.embeddings.embed_documents, [doc.page_content for doc in batch])
            records = [
                ChunkRecord(
                    content=doc.page_content,
                    embedding=embedding,
                    metadata=self._chunk_metadata(doc.metadata),
                )
                for doc, embedding in zip(batch, embeddings)
            ]

            await sync_to_async(self.pg_backend.insert_chunks)(records, insert_filters)
            if milvus_backend:
                await asyncio.to_thread(milvus_backend.insert_chunks, records, insert_filters)

            processed_batches += 1
            if progress_callback:
                progress_callback(processed_batches, total_batches)
            if user_id:
                progress_percent = int((processed_batches / total_batches) * 100)
                await send_websocket_update_async(
                    user_id=user_id,
                    message=f"Processing batch {processed_batches}/{total_batches} - {progress_percent}% complete",
                    action="update",
                )

        logger.info("All documents added successfully.")

    def _chunk_metadata(self, metadata: dict) -> dict:
        """Copy chunk metadata and map the document FK from the namespace."""
        meta = dict(metadata)
        doc_id = meta.get("document_id")
        if doc_id is not None:
            if self.collection_name.startswith("project_"):
                meta.setdefault("project_document_id", doc_id)
            elif self.collection_name.startswith("client_"):
                meta.setdefault("client_document_id", doc_id)
        return meta

    # -- search ------------------------------------------------------------------

    async def search_similar_chunks(
        self,
        query: str,
        max_k: Optional[int] = None,
        use_reranker: bool = True,  # kept for API compat; fusion is always on
        initial_fetch_k: Optional[int] = None,
        return_diagnostics: bool = False,
        project_id: Optional[int] = None,
        document_id: Optional[int] = None,
        **kwargs,
    ) -> Union[SearchResults, Tuple[SearchResults, dict]]:
        """Hybrid search with adaptive cutoff over fused RRF scores.

        Returns ``[(Document, score), ...]`` (score = fused RRF, higher is
        better). With ``return_diagnostics=True`` returns
        ``(results, diagnostics)`` where diagnostics carries all pre-cutoff
        candidate scores plus the applied cutoff config — the contract for
        ai_chat's RetrievalLog (plan part B).
        """
        max_k = max_k or settings.VECTORSTORE_MAX_K
        initial_fetch_k = initial_fetch_k or settings.VECTORSTORE_INITIAL_FETCH_K
        min_k = settings.VECTORSTORE_MIN_K
        rel_floor = settings.VECTORSTORE_RELATIVE_CUTOFF
        elbow_drop = settings.VECTORSTORE_ELBOW_DROP

        filters = SearchFilter(
            collection_name=self.collection_name,
            project_id=project_id,
            document_id=document_id,
        )
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        hits = await self.search_backend.search(
            query=query,
            query_embedding=query_embedding,
            filters=filters,
            initial_fetch_k=initial_fetch_k,
            max_k=max_k,
            rrf_k=settings.VECTORSTORE_RRF_K,
        )

        candidate_scores = [hit.score for hit in hits]
        final_k = adaptive_cutoff(
            candidate_scores,
            max_k=max_k,
            min_k=min_k,
            rel_floor=rel_floor,
            elbow_drop=elbow_drop,
        )
        results = [(hit.document, hit.score) for hit in hits[:final_k]]
        logger.info(
            f"Search in {self.collection_name}: {len(candidate_scores)} candidates, " f"adaptive cutoff kept {final_k}"
        )

        if return_diagnostics:
            diagnostics = {
                "candidate_scores": candidate_scores,
                "cutoff_config": {
                    "rel_floor": rel_floor,
                    "elbow_drop": elbow_drop,
                    "min_k": min_k,
                    "max_k": max_k,
                    "backend": settings.VECTORSTORE_BACKEND,
                },
                "final_k": final_k,
            }
            return results, diagnostics
        return results

    # -- deletion -------------------------------------------------------------

    async def delete_documents(self, project_id: Optional[int] = None, document_id: Optional[int] = None) -> None:
        if project_id is None and document_id is None:
            logger.warning("No valid project_id or document_id provided.")
            return
        filters = SearchFilter(
            collection_name=self.collection_name,
            project_id=project_id,
            document_id=document_id,
        )
        deleted = await sync_to_async(self.pg_backend.delete)(filters)
        logger.info(f"Deleted {deleted} chunks from Postgres for {filters}")

        milvus_backend = self._milvus_best_effort_backend()
        if milvus_backend:
            try:
                await asyncio.to_thread(milvus_backend.delete, filters)
            except Exception as e:
                logger.warning(f"Best-effort Milvus delete failed for {filters}: {e}")

    async def drop_collection(self) -> bool:
        result = await sync_to_async(self.pg_backend.drop_namespace)(self.collection_name)

        milvus_backend = self._milvus_best_effort_backend()
        if milvus_backend:
            try:
                await asyncio.to_thread(milvus_backend.drop_namespace, self.collection_name)
            except Exception as e:
                logger.warning(f"Best-effort Milvus drop failed for {self.collection_name}: {e}")
        return result

    # -- misc -------------------------------------------------------------------

    def process_pdf(self, document, image_folder=None, process_images=True, user_id=None) -> List[Document]:
        return self.pdf_processor.process_pdf(document, image_folder, process_images, user_id)

    def close(self) -> None:
        """No persistent connections anymore — kept for API compatibility."""

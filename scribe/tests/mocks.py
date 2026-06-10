from contextlib import contextmanager
from typing import List, Optional
from unittest.mock import AsyncMock, Mock, patch


@contextmanager
def mock_milvus_client(
    has_collection: bool = False,
    hybrid_search_results: Optional[list] = None,
    insert_count: int = 0,
    delete_count: int = 0,
    count_result: int = 0,
    collections: Optional[List[str]] = None,
):
    """Mock the pymilvus MilvusClient as used by scribe.backends.milvus_backend.

    Patches the MilvusClient name imported into the backend module, so both
    instance construction (``MilvusClient(uri=...)``) and the schema/index
    builder calls (``create_schema`` / ``prepare_index_params``) hit the mock.
    """
    with patch("scribe.backends.milvus_backend.MilvusClient") as mock_cls:
        client = mock_cls.return_value
        client.has_collection.return_value = has_collection
        client.create_collection.return_value = None
        client.drop_collection.return_value = None
        client.insert.return_value = {"insert_count": insert_count}
        client.delete.return_value = {"delete_count": delete_count}
        client.query.return_value = [{"count(*)": count_result}]
        client.hybrid_search.return_value = [hybrid_search_results if hybrid_search_results is not None else []]
        client.list_collections.return_value = collections if collections is not None else []

        yield {
            "cls": mock_cls,
            "client": client,
            "schema": client.create_schema.return_value,
            "index_params": client.prepare_index_params.return_value,
        }


def make_search_backend_mock(results: Optional[list] = None, ready: bool = True) -> Mock:
    """Mock fulfilling the SearchBackend contract (async search, sync CRUD)."""
    backend = Mock()
    backend.search = AsyncMock(return_value=results if results is not None else [])
    backend.is_ready.return_value = ready
    backend.insert_chunks.return_value = len(results) if results else 0
    backend.delete.return_value = 0
    backend.drop_namespace.return_value = True
    backend.count.return_value = 0
    backend.ensure_collection.return_value = False
    return backend


@contextmanager
def mock_scribe_service(
    backend: Optional[Mock] = None,
    pg_backend: Optional[Mock] = None,
    milvus_backend: Optional[Mock] = None,
    embedding_model: str = "text-embedding-3-large",
    embedding_size: int = 8,
):
    """Mock all SCRIBE facade dependencies: search backends + embeddings + encoder.

    Patches the names imported into ``scribe.scribe_milvus`` so the facade can
    be exercised without network, Milvus or a pgvector extension.
    """
    from ai_router.encoders import BaseEncoder

    class _MockEncoder(BaseEncoder):
        def __init__(self):
            super().__init__(name="test_encoder")

        def __call__(self, texts):
            return [[0.1] * 10 for _ in texts]

        def encode(self, texts):
            return [len(text.split()) for text in texts]

    mock_embed = Mock()
    mock_embed.model = embedding_model
    mock_embed.embed_query.return_value = [0.1] * embedding_size
    mock_embed.embed_documents.side_effect = lambda texts: [[0.1] * embedding_size for _ in texts]

    backend = backend if backend is not None else make_search_backend_mock()
    pg_backend = pg_backend if pg_backend is not None else make_search_backend_mock()
    milvus_backend = milvus_backend if milvus_backend is not None else make_search_backend_mock()

    with (
        patch("scribe.scribe_milvus.openai_embeddings", return_value=mock_embed),
        patch("scribe.scribe_milvus.openai_encoder", return_value=_MockEncoder()),
        patch("scribe.scribe_milvus.get_search_backend", return_value=backend),
        patch("scribe.scribe_milvus.PgvectorBackend", return_value=pg_backend),
        patch("scribe.scribe_milvus.MilvusBackend", return_value=milvus_backend),
    ):
        yield {
            "embeddings": mock_embed,
            "backend": backend,
            "pg_backend": pg_backend,
            "milvus_backend": milvus_backend,
        }

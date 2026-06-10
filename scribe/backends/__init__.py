"""Search backend factory: resolves the backend from settings.VECTORSTORE_BACKEND."""

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from .base import ChunkRecord, SearchBackend, SearchFilter, SearchResult

__all__ = [
    "ChunkRecord",
    "SearchBackend",
    "SearchFilter",
    "SearchResult",
    "get_search_backend",
]

BACKEND_PATHS = {
    "pgvector": "scribe.backends.pgvector_backend.PgvectorBackend",
    "milvus": "scribe.backends.milvus_backend.MilvusBackend",
}


def get_search_backend() -> SearchBackend:
    """Return a fresh instance of the configured search backend."""
    backend_name = settings.VECTORSTORE_BACKEND
    try:
        dotted_path = BACKEND_PATHS[backend_name]
    except KeyError:
        raise ImproperlyConfigured(
            f"Unknown VECTORSTORE_BACKEND: {backend_name!r}. " f"Valid values: {sorted(BACKEND_PATHS)}"
        )
    backend_class = import_string(dotted_path)
    return backend_class()

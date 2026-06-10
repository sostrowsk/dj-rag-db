# dj-rag-db

RAG document indexing and hybrid search for Django projects. Ships the
`SCRIBE` facade (`scribe.scribe_milvus.SCRIBE`): statistical chunking with
Gaussian smoothing, markdown/image/proposition chunkers, contextual
retrieval (Anthropic-style chunk contextualization), hybrid dense+full-text
search with RRF fusion and adaptive cutoff, two vector backends (Postgres
pgvector — source of truth — and Milvus) plus an OCR/PDF processing
pipeline (PyMuPDF, pymupdf4llm, Tesseract, Claude vision fallback).

Python package name: **`scribe`** (the repo name `dj-rag-db` is only the
distribution name — app label, import path, DB tables and migrations stay
`scribe`).

## Installation

Installed by the host project as a Poetry git dependency (single lock
authority lives in the host):

```toml
[tool.poetry.dependencies]
dj-rag-db = { git = "ssh://git@github.com/sostrowsk/dj-rag-db.git", branch = "main" }
```

```python
INSTALLED_APPS = [
    ...
    "ai_router",   # peer, see below
    "progress",    # peer, see below
    "scribe",
]
```

No URLs, no templates, no static files — the app ships the `DocumentChunk`
model, search backends, chunkers, the processing pipeline and management
commands (`reindex_documents`, `backfill_chunks_from_milvus`,
`rebuild_milvus_from_postgres`).

## Peer requirements

scribe imports these Django apps at runtime but does **not** declare them in
`pyproject.toml` (the host pins all dj-* packages — single lock authority).
A Django system check (`scribe.E001` / `scribe.E002`) fails fast when a peer
is missing from `INSTALLED_APPS`:

| Peer app | Package | Used for |
| --- | --- | --- |
| `ai_router` | dj-ai-router | `azure_client.openai_embeddings/openai_encoder`, `encoders.BaseEncoder`, `get_llm_client` (OCR vision, chunker/contextualizer LLM calls), `types.Document` |
| `progress` | dj-progress | `consumers.send_websocket_update[_async]` (indexing progress over Channels) |

## Host contract

- `AUTH_USER_MODEL`-style model indirection: `DocumentChunk` has FKs to a
  "project document" and a "client document" model resolved via
  `SCRIBE_PROJECT_DOCUMENT_MODEL` / `SCRIBE_CLIENT_DOCUMENT_MODEL`
  (see settings catalog). The referenced models need `file` /
  `processing_status` style fields as used by the host indexing task.
- `reindex_documents` dispatches the host's indexing Celery task resolved
  via `SCRIBE_INDEX_DOCUMENT_TASK` (dotted path, `.delay(document_id,
  model_name, user_id)` contract with literal model names
  `ProtectedProjectDocument` / `ProtectedClientDocument`).
- Tests live in the package and run from the host:
  `pytest --pyargs scribe.tests` (no own settings/pytest infrastructure).

## Settings catalog

### Required (no defaults — `django.conf.settings` attribute access)

| Setting | Used for |
| --- | --- |
| `BASE_DIR` | temp image folder during PDF image extraction |
| `VECTORSTORE_BACKEND` | `"pgvector"` or `"milvus"` — backend factory + dual-write/search routing |
| `VECTORSTORE_INITIAL_FETCH_K` | hybrid search: candidates fetched before cutoff |
| `VECTORSTORE_MAX_K` | hybrid search: max results returned |
| `VECTORSTORE_MIN_K` | adaptive cutoff: minimum kept results |
| `VECTORSTORE_RELATIVE_CUTOFF` | adaptive cutoff: relative score floor |
| `VECTORSTORE_ELBOW_DROP` | adaptive cutoff: elbow drop threshold |
| `VECTORSTORE_RRF_K` | reciprocal-rank-fusion constant |
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus connection (only when backend/dual-write is milvus; guarded reads use `getattr`) |
| `TESSDATA_DIR` | Tesseract `tessdata` directory for the OCR pipeline |
| `DISABLE_IMAGE_PROPOSITIONS` | skip LLM image proposition chunking |
| `DEFAULT_MODEL_SCRIBE` | contextualizer model in the `SCRIBE` facade |
| `DEFAULT_MODEL_SCRIBE_CHUNKER` | proposition/base chunker LLM model |
| `DEFAULT_MODEL_SCRIBE_IMAGE_CHUNKER` | image chunker LLM model |
| `DEFAULT_MODEL_SCRIBE_CONTEXTUALIZER` | `DocumentContextualizer` default model |
| `DEFAULT_MODEL_SCRIBE_OCR_PROCESSOR` | Claude vision OCR model |

(Plus everything the `ai_router` peer requires for embeddings/LLM calls:
`AZURE_LEASING_API_KEY`, `AZURE_EMBEDDINGS_*`, … — see dj-ai-router README.)

### Optional (read via `getattr` with stable defaults)

| Setting | Default | Used for |
| --- | --- | --- |
| `SCRIBE_PROJECT_DOCUMENT_MODEL` | `"data_room.ProtectedProjectDocument"` | `DocumentChunk.project_document` FK target + `scribe.conf.get_project_document_model()` |
| `SCRIBE_CLIENT_DOCUMENT_MODEL` | `"data_room.ProtectedClientDocument"` | `DocumentChunk.client_document` FK target + `scribe.conf.get_client_document_model()` |
| `SCRIBE_INDEX_DOCUMENT_TASK` | `"data_room.tasks.index_document.index_document_task"` | Celery task dispatched by `reindex_documents` |
| `SCRIBE_USE_CONTEXTUAL_RETRIEVAL` | `True` | enable chunk contextualization |
| `SCRIBE_MIN_CHUNK_TOKENS` | `500` | statistical chunker minimum split tokens |
| `SCRIBE_MAX_CHUNKS_TO_CONTEXTUALIZE` | `50` | contextualization cap per document |
| `SCRIBE_CONTEXTUALIZATION_BATCH_SIZE` | `10` | contextualization batch size |
| `VECTORSTORE_SEARCH_CONFIG` | `"german"` | Postgres full-text search config (pgvector backend) |
| `TESSERACT_CMD` | `"tesseract"` | Tesseract binary |
| `TESSERACT_TIMEOUT` | `300` | per-page OCR timeout (s) |
| `TESSERACT_LANGUAGE` | `"eng+deu"` | OCR languages |
| `TESSERACT_DPI` | `300` | OCR rendering DPI |
| `PDF_EXTRACTION_STRATEGY` | `"claude"` | `"claude"` (vision) or pymupdf4llm-first |
| `PDF_EXTRACTION_MAX_TOKENS` | `32000` | max output tokens for vision OCR |
| `VISION_OCR_ENABLED` | `True` | allow Claude vision fallback |

**Migration note:** the FK targets in `scribe/migrations/0001_initial.py` are
pinned to `data_room.protected*document`. Hosts overriding
`SCRIBE_PROJECT_DOCUMENT_MODEL` / `SCRIBE_CLIENT_DOCUMENT_MODEL` (module-level
settings FKs, like django-taggit) need their own migrations via
`MIGRATION_MODULES = {"scribe": "<host_pkg>.scribe_migrations"}`.

## System dependencies

| Dependency | Used for |
| --- | --- |
| Postgres + **pgvector** extension | `DocumentChunk.embedding` (`vector(1536)`), hybrid search backend |
| **Milvus** (optional) | secondary vector backend when `VECTORSTORE_BACKEND = "milvus"` |
| **tesseract-ocr** (+ `deu`/`eng` traineddata, see `TESSDATA_DIR`) | page-by-page OCR fallback |
| **pandoc** | DOC/DOCX → PDF conversion (`pypandoc`) |

## Dev workflow

```bash
# In the host project: override the git dep with the local checkout
poetry run pip install -e ../dj-rag-db   # NOTE: `poetry install` reverts this

# Run the package tests from the host
poetry run pytest --pyargs scribe.tests
```

Release: commit + push to `main`, then in the host
`poetry update dj-rag-db`.

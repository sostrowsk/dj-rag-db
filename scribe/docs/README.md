# SCRIBE: Semantic Content Retrieval and Information-Based Extraction

SCRIBE verarbeitet, speichert und durchsucht Dokumenten-Inhalte (Chunking,
Kontextualisierung, Hybrid Search). Seit dem pgvector-Refactor gilt:

**Postgres ist Source of Truth.** Alle Chunks (inkl. Embeddings) liegen als
`scribe.models.DocumentChunk`-Rows in Postgres. Milvus ist ein optionaler,
abgeleiteter Index, der jederzeit aus Postgres rebuildet werden kann.

## Architektur

```
Upload → PDFProcessor/Chunker → Contextualizer → embed_documents (einmal, batched)
       → DocumentChunk-Rows (Postgres, immer; delete-then-insert = idempotent)
       → Milvus-Mirror (nur wenn VECTORSTORE_BACKEND=milvus)

Suche  → Query einmal embedden → SearchBackend.search()
         (hybrid: Dense + BM25/FTS, weighted RRF-Fusion)
       → adaptive_cutoff über fused Scores → List[Tuple[Document, score]]
```

Komponenten:

- **`scribe/models.py`** — `DocumentChunk`: `collection_name`-Namespace
  (`project_{id}`, `client_{id}`, `general_chat`), nullable FKs auf
  Projekt-/Client-Dokument, `HalfVectorField(3072)` mit HNSW-Index
  (halfvec_cosine), German-`tsvector` (Trigger-gepflegt) mit GIN-Index,
  UniqueConstraint `(collection_name, document_id, chunk_id)`.
- **`scribe/backends/`** — `SearchBackend`-Interface (`base.py`),
  `PgvectorBackend` (ORM: CosineDistance + SearchQuery/SearchRank, RRF in
  Python), `MilvusBackend` (pymilvus `MilvusClient`, zwei `AnnSearchRequest`
  + `RRFRanker`; kein langchain). Auswahl via `get_search_backend()`.
- **`scribe/retrieval/adaptive_cutoff.py`** — pure function; schneidet die
  Ergebnisliste per relativem Score-Floor + Elbow-Erkennung (statt fixem k).
- **`scribe/scribe_milvus.py`** — `SCRIBE`-Facade (Public API:
  `process_pdf`, `add_documents_to_collection`, `search_similar_chunks`,
  `delete_documents`, `drop_collection`, `check_milvus_health[_static]`).
  Delete/Drop treffen immer Postgres, Milvus best-effort.

## Settings (`VECTORSTORE_*`, alle env-overridable)

| Setting | Default | Bedeutung |
|---|---|---|
| `VECTORSTORE_BACKEND` | `milvus` | `pgvector` \| `milvus` (Such-Backend; Default bleibt `milvus` bis zum Backfill-Flip) |
| `VECTORSTORE_SEARCH_CONFIG` | `german` | Postgres-FTS-Konfiguration |
| `VECTORSTORE_INITIAL_FETCH_K` | `150` | Kandidaten pro Branch vor Fusion |
| `VECTORSTORE_MAX_K` | `50` | harte Obergrenze nach Cutoff |
| `VECTORSTORE_MIN_K` | `3` | Untergrenze nach Cutoff |
| `VECTORSTORE_RELATIVE_CUTOFF` | `0.35` | Cut wenn `score/top < floor` |
| `VECTORSTORE_ELBOW_DROP` | `0.45` | Cut bei Einzelschritt-Drop `> x` |
| `VECTORSTORE_RRF_K` | `60` | RRF-Konstante der Fusion |

Voraussetzung pgvector: Postgres-Extension `vector` (Migration `0001` macht
`CREATE EXTENSION IF NOT EXISTS vector`; Paket via ansible-Rolle
`postgresql_pgvector`, CI nutzt das Image `pgvector/pgvector:pg17`).

## Management-Commands (Rollout)

- `backfill_chunks_from_milvus [--collection X] [--dry-run]` — einmaliger
  Import bestehender Milvus-Collections nach Postgres **ohne Re-Embedding**
  (Vektoren werden uebernommen); idempotent, Orphans werden reported.
- `reindex_documents [--project N] [--document N] [--only-missing] [--limit N] [--user-id N]`
  — voller Rebuild (Re-Chunk + Re-Embed aus Markdown) ueber die normale
  Celery-Pipeline; kostet Embedding-Calls.
- `rebuild_milvus_from_postgres [--collection X] [--dry-run]` — Milvus
  komplett aus Postgres neu aufbauen (drop + bulk insert, keine API-Calls).
  Garantiert "Milvus = derived index"; auch fuer Server-Upgrades.

### Rollout-Reihenfolge (Produktion)

1. Deploy ohne gesetzte Env (`VECTORSTORE_BACKEND` defaultet auf `milvus` — Verhalten unveraendert).
2. `backfill_chunks_from_milvus` → Postgres-SSOT befuellt.
3. Paritaets-Spot-Check (Rank-Vergleich, nicht Raw-Scores).
4. Flip auf `VECTORSTORE_BACKEND=pgvector`.
5. Optional: Milvus-3-Server-Upgrade (ansible) + `rebuild_milvus_from_postgres`,
   falls Milvus aktiv bleiben soll.
6. Nach 1–2 Wochen Traffic: `tune_retrieval` (Teil B) → Thresholds nachziehen.

## Contextual Retrieval

Beim Indexing wird jeder Chunk mit Kontext aus dem Gesamtdokument angereichert
(`<context>…</context>`-Prefix), bevor er embedded wird. Dense- und
BM25/FTS-Branch profitieren beide vom Kontext. Steuerung via
`SCRIBE_USE_CONTEXTUAL_RETRIEVAL` (Default an).

```python
scribe = SCRIBE("project_42")
chunks = await scribe.process_pdf(document)
await scribe.add_documents_to_collection(chunks, document)

results = await scribe.search_similar_chunks("Umsatz Q2 2023?")
# -> List[Tuple[Document, fused_rrf_score]], Laenge adaptiv (min_k..max_k)

results, diagnostics = await scribe.search_similar_chunks(
    "Umsatz Q2 2023?", return_diagnostics=True
)
# diagnostics = {"candidate_scores": [...], "cutoff_config": {...}, "final_k": n}
```

## Tests

```bash
poetry run pytest scribe/
```

Integrationstests gegen echtes Postgres+pgvector liegen in
`scribe/tests/integration/test_pgvector_search.py`; Milvus wird ausschliesslich
ueber `scribe.tests.mocks.mock_milvus_client` gemockt.

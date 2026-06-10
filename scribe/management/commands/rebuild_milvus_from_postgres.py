"""Rebuild Milvus collections from the Postgres SSOT (DocumentChunk).

Guarantees "Milvus = derived index": per collection the old Milvus
collection is dropped, re-created (HNSW/COSINE + BM25 schema via
``MilvusBackend.ensure_collection``) and bulk-filled from the stored
``DocumentChunk`` rows. Embeddings are reused — NO embedding API calls.

Milvus VARCHAR fields are capped at 65535 bytes; overlong text fields are
truncated at 65000 bytes (UTF-8-safe) with a warning.

Usage:
    python manage.py rebuild_milvus_from_postgres
    python manage.py rebuild_milvus_from_postgres --collection project_42
    python manage.py rebuild_milvus_from_postgres --dry-run
"""

import logging

from django.core.management.base import BaseCommand, CommandError

from scribe.backends.base import ChunkRecord, SearchFilter
from scribe.backends.milvus_backend import MilvusBackend
from scribe.models import DocumentChunk

logger = logging.getLogger(__name__)

#: Stay safely below the Milvus VARCHAR limit of 65535 bytes.
TRUNCATE_AT_BYTES = 65000

BATCH_SIZE = 500

TEXT_FIELDS = ["content", "original_content", "raw_section", "document_path", "image_path"]


class Command(BaseCommand):
    help = "Drop + rebuild Milvus collections from DocumentChunk rows (no embedding API calls)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            type=str,
            default=None,
            help="Only rebuild this collection (e.g. project_42).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be rebuilt without touching Milvus.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        only_collection = options["collection"]

        collections = list(
            DocumentChunk.objects.order_by("collection_name").values_list("collection_name", flat=True).distinct()
        )
        if only_collection:
            if only_collection not in collections:
                raise CommandError(f"Keine DocumentChunk-Rows fuer Collection '{only_collection}' vorhanden.")
            collections = [only_collection]

        if dry_run:
            self.stdout.write(self.style.NOTICE("Dry-Run: Milvus wird nicht angefasst."))
            for collection_name in collections:
                count = DocumentChunk.objects.filter(collection_name=collection_name).count()
                self.stdout.write(f"{collection_name}: {count} Chunks wuerden neu aufgebaut.")
            return

        backend = MilvusBackend()
        total = 0
        for collection_name in collections:
            inserted = self._rebuild_collection(backend, collection_name)
            total += inserted
            self.stdout.write(f"{collection_name}: {inserted} Chunks nach Milvus geschrieben.")

        self.stdout.write(self.style.SUCCESS(f"Rebuild abgeschlossen: {len(collections)} Collections, {total} Chunks."))

    def _rebuild_collection(self, backend: MilvusBackend, collection_name: str) -> int:
        backend.drop_namespace(collection_name)
        filters = SearchFilter(collection_name=collection_name)

        inserted = 0
        records: list[ChunkRecord] = []
        queryset = DocumentChunk.objects.filter(collection_name=collection_name).order_by("document_id", "chunk_id")
        for chunk in queryset.iterator(chunk_size=BATCH_SIZE):
            records.append(self._to_record(chunk))
            if len(records) >= BATCH_SIZE:
                inserted += backend.insert_chunks(records, filters)
                records = []
        if records:
            inserted += backend.insert_chunks(records, filters)
        return inserted

    def _to_record(self, chunk: DocumentChunk) -> ChunkRecord:
        metadata = {
            "document_id": chunk.document_id,
            "project_id": chunk.project_id,
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "has_context": chunk.has_context,
        }
        for field in TEXT_FIELDS:
            if field == "content":
                continue
            metadata[field] = self._truncated(chunk, field)
        # halfvec rows come back as pgvector HalfVector — Milvus wants plain floats.
        embedding = chunk.embedding.to_list() if hasattr(chunk.embedding, "to_list") else list(chunk.embedding)
        return ChunkRecord(
            content=self._truncated(chunk, "content"),
            embedding=[float(value) for value in embedding],
            metadata=metadata,
        )

    def _truncated(self, chunk: DocumentChunk, field: str) -> str:
        """UTF-8-safe byte truncation below the Milvus VARCHAR limit."""
        value = getattr(chunk, field) or ""
        encoded = value.encode("utf-8")
        if len(encoded) <= TRUNCATE_AT_BYTES:
            return value
        truncated = encoded[:TRUNCATE_AT_BYTES].decode("utf-8", errors="ignore")
        self.stdout.write(
            self.style.WARNING(
                f"{chunk.collection_name}/doc {chunk.document_id}/chunk {chunk.chunk_id}: "
                f"Feld '{field}' truncated auf {TRUNCATE_AT_BYTES} Bytes (Milvus VARCHAR-Limit)."
            )
        )
        return truncated

"""One-time backfill: copy chunks (incl. embeddings) from Milvus into Postgres.

Reads every scribe collection (``project_*``, ``client_*``, ``general_chat``)
via ``MilvusClient.query_iterator`` and writes ``DocumentChunk`` rows with
``bulk_create(ignore_conflicts=True)`` — re-runs are idempotent and no
embedding API calls are made (text-embedding-3-large vectors are reused).

FKs are mapped over the denormalized ``document_id``: project collections
link ``ProtectedProjectDocument``, client collections link
``ProtectedClientDocument``. Rows whose document no longer exists are still
inserted (FK = NULL) and reported as orphans.

Usage:
    python manage.py backfill_chunks_from_milvus
    python manage.py backfill_chunks_from_milvus --collection project_42
    python manage.py backfill_chunks_from_milvus --dry-run
"""

import hashlib
import logging

from django.core.management.base import BaseCommand, CommandError

from scribe.backends.milvus_backend import MilvusBackend
from scribe.conf import get_client_document_model, get_project_document_model
from scribe.models import DocumentChunk

logger = logging.getLogger(__name__)

GENERAL_COLLECTION = "general_chat"

#: Legacy Milvus schema fields needed to reconstruct a DocumentChunk row.
OUTPUT_FIELDS = [
    "content",
    "embedding",
    "document_id",
    "project_id",
    "chunk_id",
    "document_path",
    "raw_section",
    "image_path",
    "original_content",
    "page_number",
    "has_context",
]

BATCH_SIZE = 500


class Command(BaseCommand):
    help = "Backfill DocumentChunk rows (incl. embeddings) from existing Milvus collections."

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            type=str,
            default=None,
            help="Only backfill this collection (e.g. project_42).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be inserted without writing to Postgres.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        only_collection = options["collection"]

        backend = MilvusBackend()
        all_collections = backend.client.list_collections()
        targets = [name for name in all_collections if self._is_scribe_collection(name)]

        if only_collection:
            if only_collection not in targets:
                raise CommandError(f"Collection '{only_collection}' not found in Milvus (or not a scribe collection)")
            targets = [only_collection]

        if dry_run:
            self.stdout.write(self.style.NOTICE("Dry-Run: es werden keine Rows geschrieben."))

        total_rows = 0
        total_orphans = 0
        for collection_name in targets:
            rows, orphans = self._backfill_collection(backend, collection_name, dry_run)
            total_rows += rows
            total_orphans += orphans
            self.stdout.write(f"{collection_name}: {rows} Chunks, {orphans} Orphans (Dokument fehlt in DB)")

        self.stdout.write(
            self.style.SUCCESS(
                f"Backfill {'(Dry-Run) ' if dry_run else ''}abgeschlossen: "
                f"{len(targets)} Collections, {total_rows} Chunks, {total_orphans} Orphans."
            )
        )

    @staticmethod
    def _is_scribe_collection(name: str) -> bool:
        return name.startswith("project_") or name.startswith("client_") or name == GENERAL_COLLECTION

    @staticmethod
    def _unique_chunk_id(document_id: int, chunk_id: int, seen: set[tuple[int, int]]) -> int:
        """Naechste freie chunk_id pro Dokument; deterministisch bei Re-Runs."""
        while (document_id, chunk_id) in seen:
            chunk_id += 1
        seen.add((document_id, chunk_id))
        return chunk_id

    @staticmethod
    def _content_fingerprint(content: str) -> str:
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def _backfill_collection(self, backend: MilvusBackend, collection_name: str, dry_run: bool) -> tuple[int, int]:
        """Iterate one collection and insert its rows. Returns (rows, orphans)."""
        iterator = backend.client.query_iterator(
            collection_name=collection_name,
            batch_size=BATCH_SIZE,
            output_fields=OUTPUT_FIELDS,
        )
        rows = 0
        orphans = 0
        # Legacy-Milvus enthaelt doppelte (document_id, chunk_id)-Paare (Text-
        # vs. Image-Chunks mit eigenem idx ab 0). Unter dem UniqueConstraint
        # wuerde ignore_conflicts sie still verwerfen — stattdessen bekommt
        # jedes Duplikat die naechste freie chunk_id. Beide Dedupe-Sets werden
        # aus den DB-Bestandszeilen geseedet, damit Re-Runs gegen teilweise
        # backfillte Collections weder Content duplizieren noch fehlende
        # Chunks erneut verwerfen (Reihenfolge des Milvus-Streams ist egal):
        # Bei ID-Kollision entscheidet der Content-Fingerprint, ob die Zeile
        # schon existiert (skip) oder ein echter neuer Chunk ist (reassign).
        seen_chunk_ids: set[tuple[int, int]] = set()
        seen_content: set[tuple[int, str]] = set()
        for doc_id, chunk_id, content in DocumentChunk.objects.filter(collection_name=collection_name).values_list(
            "document_id", "chunk_id", "content"
        ):
            seen_chunk_ids.add((doc_id, chunk_id))
            seen_content.add((doc_id, self._content_fingerprint(content)))
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                chunk_rows, batch_orphans = self._build_rows(collection_name, batch, seen_chunk_ids, seen_content)
                if not dry_run:
                    DocumentChunk.objects.bulk_create(chunk_rows, ignore_conflicts=True)
                rows += len(chunk_rows)
                orphans += batch_orphans
        finally:
            iterator.close()
        return rows, orphans

    def _build_rows(
        self,
        collection_name: str,
        batch: list[dict],
        seen_chunk_ids: set[tuple[int, int]],
        seen_content: set[tuple[int, str]],
    ) -> tuple[list[DocumentChunk], int]:
        fk_field, existing_ids = self._resolve_fk_targets(collection_name, batch)

        chunk_rows = []
        orphans = 0
        for row in batch:
            document_id = int(row.get("document_id") or 0)
            source_chunk_id = int(row.get("chunk_id") or 0)
            fingerprint = (document_id, self._content_fingerprint(row.get("content") or ""))
            if (document_id, source_chunk_id) in seen_chunk_ids and fingerprint in seen_content:
                continue  # logisch identische Zeile existiert bereits (Re-Run)
            seen_content.add(fingerprint)
            fk_value = None
            if fk_field:
                if document_id in existing_ids:
                    fk_value = document_id
                else:
                    orphans += 1
            chunk_id = self._unique_chunk_id(document_id, source_chunk_id, seen_chunk_ids)
            chunk_rows.append(
                DocumentChunk(
                    collection_name=collection_name,
                    project_document_id=fk_value if fk_field == "project_document_id" else None,
                    client_document_id=fk_value if fk_field == "client_document_id" else None,
                    document_id=document_id,
                    project_id=self._project_id_for(collection_name, row),
                    chunk_id=chunk_id,
                    content=row.get("content") or "",
                    original_content=row.get("original_content") or "",
                    raw_section=row.get("raw_section") or "",
                    document_path=row.get("document_path") or "",
                    image_path=row.get("image_path") or "",
                    page_number=row.get("page_number"),
                    has_context=bool(row.get("has_context") or False),
                    embedding=row.get("embedding"),
                )
            )
        return chunk_rows, orphans

    @staticmethod
    def _resolve_fk_targets(collection_name: str, batch: list[dict]) -> tuple[str | None, set[int]]:
        """Which FK field to fill and which document ids actually exist."""
        if collection_name.startswith("project_"):
            model, fk_field = get_project_document_model(), "project_document_id"
        elif collection_name.startswith("client_"):
            model, fk_field = get_client_document_model(), "client_document_id"
        else:
            return None, set()
        document_ids = {int(row.get("document_id") or 0) for row in batch}
        existing = set(model.objects.filter(id__in=document_ids).values_list("id", flat=True))
        return fk_field, existing

    @staticmethod
    def _project_id_for(collection_name: str, row: dict) -> int | None:
        """Denormalized project_id: stored value, else parsed from the namespace."""
        value = row.get("project_id")
        if value:
            return int(value)
        if collection_name.startswith("project_"):
            suffix = collection_name.removeprefix("project_")
            if suffix.isdigit():
                return int(suffix)
        return None

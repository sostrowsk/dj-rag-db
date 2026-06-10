"""Full re-index (re-chunk + re-embed from markdown) via the normal pipeline.

Dispatches ``index_document_task`` per document (Celery), i.e. the same
SCRIBE path the upload pipeline uses: chunking + contextualization +
embedding + Postgres-SSOT write (and Milvus mirror when configured).

Costs embedding API calls — use ``backfill_chunks_from_milvus`` instead when
the existing Milvus vectors are still valid. Documents without markdown are
skipped (nothing to chunk).

Usage:
    python manage.py reindex_documents
    python manage.py reindex_documents --project 42
    python manage.py reindex_documents --document 1337
    python manage.py reindex_documents --only-missing --limit 100
"""

import logging

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from scribe.conf import get_client_document_model, get_index_document_task, get_project_document_model
from scribe.models import DocumentChunk

logger = logging.getLogger(__name__)

User = get_user_model()


class Command(BaseCommand):
    help = "Re-chunk + re-embed documents from markdown through the normal SCRIBE pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--project", type=int, default=None, help="Only documents of this project ID.")
        parser.add_argument("--document", type=int, default=None, help="Only this single document ID.")
        parser.add_argument(
            "--only-missing",
            action="store_true",
            help="Skip documents that already have DocumentChunk rows in Postgres.",
        )
        parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to dispatch.")
        parser.add_argument(
            "--user-id",
            type=int,
            default=None,
            help="User ID for pipeline context (default: first superuser).",
        )

    def handle(self, *args, **options):
        user_id = self._resolve_user_id(options["user_id"])
        candidates = self._collect_candidates(options["project"], options["document"])

        if options["only_missing"]:
            candidates = [
                (doc, model_name, collection)
                for doc, model_name, collection in candidates
                if not DocumentChunk.objects.filter(collection_name=collection, document_id=doc.id).exists()
            ]

        if options["limit"] is not None:
            candidates = candidates[: options["limit"]]

        total_tokens = sum(doc.tokens or 0 for doc, _, _ in candidates)
        self.stdout.write(f"Re-Index: {len(candidates)} Dokumente, ~{total_tokens} Tokens werden neu embedded.")

        index_document_task = get_index_document_task()
        for doc, model_name, _ in candidates:
            index_document_task.delay(doc.id, user_id, model_name)

        self.stdout.write(self.style.SUCCESS(f"{len(candidates)} Indexierungs-Tasks dispatched."))

    def _collect_candidates(self, project_id, document_id):
        """Build [(doc, model_name, collection_name)] for docs with markdown."""
        project_docs = get_project_document_model().objects.exclude(markdown="").order_by("id")
        client_docs = get_client_document_model().objects.exclude(markdown="").order_by("id")

        if project_id is not None:
            project_docs = project_docs.filter(project_id=project_id)
            client_docs = client_docs.none()

        if document_id is not None:
            project_docs = project_docs.filter(id=document_id)
            # A project doc takes precedence when both tables contain the id.
            client_docs = client_docs.none() if project_docs.exists() else client_docs.filter(id=document_id)
            if not project_docs.exists() and not client_docs.exists():
                raise CommandError(f"Kein Dokument mit ID {document_id} (mit Markdown) gefunden.")

        candidates = [(doc, "ProtectedProjectDocument", f"project_{doc.project_id}") for doc in project_docs] + [
            (doc, "ProtectedClientDocument", f"client_{doc.client_id}") for doc in client_docs
        ]
        return candidates

    def _resolve_user_id(self, user_id):
        if user_id is not None:
            return user_id
        superuser_id = User.objects.filter(is_superuser=True).values_list("id", flat=True).first()
        if superuser_id is None:
            raise CommandError("Kein Superuser gefunden — bitte --user-id angeben.")
        return superuser_id

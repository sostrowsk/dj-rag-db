"""Host-configurable indirection for scribe's document-model coupling.

scribe stores chunks for two host document models and re-dispatches the
host's indexing pipeline from management commands. Hosts repoint these via
settings; the defaults match the leasing monorepo (data_room app):

- ``SCRIBE_PROJECT_DOCUMENT_MODEL`` (default ``data_room.ProtectedProjectDocument``)
- ``SCRIBE_CLIENT_DOCUMENT_MODEL`` (default ``data_room.ProtectedClientDocument``)
- ``SCRIBE_INDEX_DOCUMENT_TASK`` (default
  ``data_room.tasks.index_document.index_document_task``)

All lookups are lazy (call-time), so scribe itself never imports data_room
at module level. Note: the model settings are also read at model-definition
time for the ``DocumentChunk`` FKs — hosts that override them need their own
migrations (see README of dj-rag-db).
"""

from django.apps import apps
from django.conf import settings
from django.utils.module_loading import import_string

DEFAULT_PROJECT_DOCUMENT_MODEL = "data_room.ProtectedProjectDocument"
DEFAULT_CLIENT_DOCUMENT_MODEL = "data_room.ProtectedClientDocument"
DEFAULT_INDEX_DOCUMENT_TASK = "data_room.tasks.index_document.index_document_task"


def get_project_document_model():
    """Return the model class chunks of project documents belong to."""
    return apps.get_model(getattr(settings, "SCRIBE_PROJECT_DOCUMENT_MODEL", DEFAULT_PROJECT_DOCUMENT_MODEL))


def get_client_document_model():
    """Return the model class chunks of client documents belong to."""
    return apps.get_model(getattr(settings, "SCRIBE_CLIENT_DOCUMENT_MODEL", DEFAULT_CLIENT_DOCUMENT_MODEL))


def get_index_document_task():
    """Return the host's Celery indexing task (dotted-path setting)."""
    return import_string(getattr(settings, "SCRIBE_INDEX_DOCUMENT_TASK", DEFAULT_INDEX_DOCUMENT_TASK))

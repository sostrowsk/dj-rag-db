"""Decoupling tests for scribe.conf — host-configurable data_room indirection.

scribe must not hard-import data_room at module level; the document models
and the indexing task are resolved lazily via settings with leasing-defaults
(SCRIBE_PROJECT_DOCUMENT_MODEL, SCRIBE_CLIENT_DOCUMENT_MODEL,
SCRIBE_INDEX_DOCUMENT_TASK).
"""

import ast
from pathlib import Path

from django.test import SimpleTestCase, override_settings

import scribe
from scribe import conf
from scribe.models import DocumentChunk


class TestConfDefaults(SimpleTestCase):
    """conf resolves to the data_room defaults when no setting is present."""

    def test_get_project_document_model_defaults_to_data_room(self):
        from data_room.models import ProtectedProjectDocument

        self.assertIs(conf.get_project_document_model(), ProtectedProjectDocument)

    def test_get_client_document_model_defaults_to_data_room(self):
        from data_room.models import ProtectedClientDocument

        self.assertIs(conf.get_client_document_model(), ProtectedClientDocument)

    def test_get_index_document_task_defaults_to_data_room_pipeline(self):
        from data_room.tasks.index_document import index_document_task

        self.assertIs(conf.get_index_document_task(), index_document_task)


class TestConfOverrides(SimpleTestCase):
    """Hosts can repoint the document models / indexing task via settings."""

    @override_settings(SCRIBE_PROJECT_DOCUMENT_MODEL="scribe.DocumentChunk")
    def test_project_document_model_setting_overrides_default(self):
        self.assertIs(conf.get_project_document_model(), DocumentChunk)

    @override_settings(SCRIBE_CLIENT_DOCUMENT_MODEL="scribe.DocumentChunk")
    def test_client_document_model_setting_overrides_default(self):
        self.assertIs(conf.get_client_document_model(), DocumentChunk)

    @override_settings(SCRIBE_INDEX_DOCUMENT_TASK="scribe.conf.get_index_document_task")
    def test_index_document_task_setting_overrides_default(self):
        self.assertIs(conf.get_index_document_task(), conf.get_index_document_task)


class TestModelFKTargets(SimpleTestCase):
    """DocumentChunk FK targets stay byte-stable on the data_room defaults."""

    def test_project_document_fk_points_to_configured_model(self):
        field = DocumentChunk._meta.get_field("project_document")
        self.assertIs(field.remote_field.model, conf.get_project_document_model())
        self.assertEqual(field.remote_field.model._meta.label, "data_room.ProtectedProjectDocument")

    def test_client_document_fk_points_to_configured_model(self):
        field = DocumentChunk._meta.get_field("client_document")
        self.assertIs(field.remote_field.model, conf.get_client_document_model())
        self.assertEqual(field.remote_field.model._meta.label, "data_room.ProtectedClientDocument")


class TestNoModuleLevelDataRoomImports(SimpleTestCase):
    """No scribe production module may import data_room (only conf strings)."""

    def test_scribe_sources_have_no_data_room_imports(self):
        package_dir = Path(scribe.__file__).resolve().parent
        offenders = []
        for path in sorted(package_dir.rglob("*.py")):
            relative = path.relative_to(package_dir)
            if "tests" in relative.parts or "__pycache__" in relative.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == "data_room" or name.startswith("data_room.") for name in names):
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], f"data_room imports found in scribe: {offenders}")

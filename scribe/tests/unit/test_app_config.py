"""Tests for scribe Django app registration and VECTORSTORE_* settings."""

from django.apps import apps
from django.conf import settings
from django.test import SimpleTestCase


class TestScribeAppConfig(SimpleTestCase):
    """scribe must be a registered Django app."""

    def test_scribe_app_is_registered(self):
        config = apps.get_app_config("scribe")
        self.assertEqual(config.name, "scribe")

    def test_scribe_app_uses_big_auto_field(self):
        config = apps.get_app_config("scribe")
        self.assertEqual(config.default_auto_field, "django.db.models.BigAutoField")


class TestVectorstoreSettings(SimpleTestCase):
    """VECTORSTORE_* settings exist with the documented defaults."""

    def test_backend_in_test_env_is_pgvector(self):
        self.assertEqual(settings.VECTORSTORE_BACKEND, "pgvector")

    def test_search_config_is_german(self):
        self.assertEqual(settings.VECTORSTORE_SEARCH_CONFIG, "german")

    def test_fetch_and_k_limits(self):
        self.assertEqual(settings.VECTORSTORE_INITIAL_FETCH_K, 150)
        self.assertEqual(settings.VECTORSTORE_MAX_K, 50)
        self.assertEqual(settings.VECTORSTORE_MIN_K, 3)

    def test_cutoff_heuristics(self):
        self.assertAlmostEqual(settings.VECTORSTORE_RELATIVE_CUTOFF, 0.35)
        self.assertAlmostEqual(settings.VECTORSTORE_ELBOW_DROP, 0.45)

    def test_rrf_k(self):
        self.assertEqual(settings.VECTORSTORE_RRF_K, 60)

    def test_numeric_settings_have_proper_types(self):
        self.assertIsInstance(settings.VECTORSTORE_INITIAL_FETCH_K, int)
        self.assertIsInstance(settings.VECTORSTORE_MAX_K, int)
        self.assertIsInstance(settings.VECTORSTORE_MIN_K, int)
        self.assertIsInstance(settings.VECTORSTORE_RRF_K, int)
        self.assertIsInstance(settings.VECTORSTORE_RELATIVE_CUTOFF, float)
        self.assertIsInstance(settings.VECTORSTORE_ELBOW_DROP, float)

"""Peer-requirement system check: scribe requires ai_router + progress in the host.

scribe imports ``ai_router.encoders`` / ``ai_router.get_llm_client`` and
``progress.consumers`` at runtime but (per architecture rule) does NOT declare
them as package dependencies — the host pins all dj-* packages. The system
check makes a missing peer fail fast at ``manage.py check`` time.
"""

from unittest import mock

from django.test import SimpleTestCase


class TestScribePeerCheck(SimpleTestCase):
    def test_check_passes_when_peers_installed(self):
        from scribe.apps import check_peer_apps

        self.assertEqual(check_peer_apps(app_configs=None), [])

    def test_check_reports_one_error_per_missing_peer(self):
        from scribe.apps import check_peer_apps

        with mock.patch("scribe.apps.django_apps.is_installed", return_value=False):
            errors = check_peer_apps(app_configs=None)

        self.assertEqual({e.id for e in errors}, {"scribe.E001", "scribe.E002"})
        joined = " ".join(e.msg for e in errors)
        self.assertIn("ai_router", joined)
        self.assertIn("progress", joined)

    def test_check_is_registered_with_django(self):
        from django.core.checks.registry import registry

        from scribe.apps import check_peer_apps

        self.assertIn(check_peer_apps, registry.registered_checks)

from django.apps import AppConfig
from django.apps import apps as django_apps
from django.core import checks

#: Peer Django apps scribe imports at runtime (ai_router.encoders /
#: get_llm_client, progress.consumers). Per architecture rule the package
#: does NOT declare them in pyproject — the host pins all dj-* packages and
#: this system check fails fast when a peer is missing.
PEER_APPS = ("ai_router", "progress")


def check_peer_apps(app_configs, **kwargs):
    errors = []
    for index, peer in enumerate(PEER_APPS, start=1):
        if not django_apps.is_installed(peer):
            errors.append(
                checks.Error(
                    f"scribe requires the '{peer}' Django app to be installed.",
                    hint=f"Add '{peer}' to INSTALLED_APPS (host pins the dj-* package).",
                    id=f"scribe.E{index:03d}",
                )
            )
    return errors


class ScribeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "scribe"

    def ready(self):
        checks.register(check_peer_apps)

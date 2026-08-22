"""Earth Engine initialization.

Two paths, both driven by server/.env:

  * Interactive (default) — run `earthengine authenticate` once. Credentials are
    cached in your user profile, outside this repo, and nothing sensitive is
    stored here. This is what you want on a laptop.
  * Service account — only for non-interactive/CI runs. Keep the key file
    OUTSIDE the repo and point GEE_SERVICE_ACCOUNT_KEY_PATH at an absolute path.

Either way GEE_PROJECT must be set; ee.Initialize requires a Cloud project that
is registered for Earth Engine.
"""

from __future__ import annotations

import sys

import ee

from app.core.config import settings


class GEEAuthError(RuntimeError):
    """Raised with a message that names the fix, not just the failure."""


def initialize() -> None:
    if not settings.gee_project:
        raise GEEAuthError(
            "GEE_PROJECT is not set.\n"
            "  1. Register a Cloud project at https://console.cloud.google.com/earth-engine\n"
            "     (noncommercial is free and needs no billing)\n"
            f"  2. Add GEE_PROJECT=<your-project-id> to {settings.model_config['env_file']}\n"
            "     (copy server/.env.example to server/.env if it does not exist yet)"
        )

    key_path = settings.gee_service_account_key_path
    email = settings.gee_service_account_email

    try:
        if key_path and email:
            credentials = ee.ServiceAccountCredentials(email, key_path)
            ee.Initialize(credentials, project=settings.gee_project)
        else:
            ee.Initialize(project=settings.gee_project)
    except Exception as exc:  # noqa: BLE001 - surface the real cause with a fix
        raise GEEAuthError(
            f"Earth Engine failed to initialize for project '{settings.gee_project}'.\n"
            f"  Underlying error: {exc}\n"
            "  If this says credentials are missing, run: earthengine authenticate\n"
            "  If it says the project is not registered, complete signup at\n"
            "  https://console.cloud.google.com/earth-engine"
        ) from exc


def initialize_or_exit() -> None:
    """Same as initialize(), but prints the fix and exits rather than tracebacking."""
    try:
        initialize()
    except GEEAuthError as exc:
        print(f"\n[gee] {exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc

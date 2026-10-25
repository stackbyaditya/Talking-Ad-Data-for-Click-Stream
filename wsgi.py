"""Vercel entrypoint (Vercel looks for a top-level `app` in wsgi.py).

Local and Docker runs use webapp/app.py directly; this file only exposes the same Flask app
to Vercel's Python runtime."""

import os
import sys
from pathlib import Path

# Public serverless demo: never write to disk (the filesystem is read-only apart from /tmp).
os.environ.setdefault("SAVE_ENABLED", "0")
os.environ.setdefault("STORE_DIR", "/tmp/collected")

sys.path.insert(0, str(Path(__file__).resolve().parent / "webapp"))

from app import create_app  # noqa: E402  (webapp/app.py)

app = create_app()

"""Vercel entry point for the ReviewGuard FastAPI application."""

from pathlib import Path
import sys

# The application uses a src/ package layout; make that package visible to the
# Vercel Python runtime when it imports this root-level entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from reviewguard.api.main import app  # noqa: E402

__all__ = ["app"]

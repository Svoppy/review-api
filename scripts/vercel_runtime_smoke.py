"""Check that the slim Vercel API boots without ML packages or a checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Production Vercel installs only the base API dependencies. Prevent this smoke
# check from accidentally passing because the CI test environment has ML extras.
sys.modules["torch"] = None
sys.modules["transformers"] = None

from fastapi.testclient import TestClient

from index import app

client = TestClient(app)
health = client.get("/health")
assert health.status_code == 200, health.text
assert health.json()["model_ready"] is False, health.text
assert client.get("/").status_code == 200
analyze = client.post("/analyze", json={"text": "A useful product review."})
assert analyze.status_code == 503, analyze.text
print("Vercel runtime smoke passed: UI and health are available; inference reports no checkpoint.")

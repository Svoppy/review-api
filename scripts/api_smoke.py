from __future__ import annotations

import json

from fastapi.testclient import TestClient

from reviewguard.api.main import app


def main() -> int:
    client = TestClient(app)

    health = client.get("/health")
    if health.status_code != 200:
        raise SystemExit(f"/health failed: {health.status_code} {health.text}")

    payload = {
        "text": "Fast delivery, decent packaging, but the item quality feels inconsistent and a bit suspicious."
    }
    analyze = client.post("/analyze", json=payload)
    if analyze.status_code != 200:
        raise SystemExit(f"/analyze failed: {analyze.status_code} {analyze.text}")

    print("health:", json.dumps(health.json(), ensure_ascii=False))
    print("analyze:", json.dumps(analyze.json(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Recorded Test Environment

Verification date: `2026-06-11`

## 1. Current Working Snapshot Environment

The locally verified environment for the current desktop snapshot is:

- Python: `3.14.2`
- virtualenv: `.venv314`
- shell: `zsh`
- OS: macOS desktop environment

This is the environment in which the following were confirmed:

- full local `pytest`;
- creation of `joint_reviews.current.jsonl`;
- creation of `joint_reviews.balanced6k.jsonl`;
- generation of `reports/audit/joint_reviews.balanced6k.audit.json`.

## 2. Pinned Environment Files

The repository also contains:

- `.python-version` = `3.12.7`
- `requirements-dev.txt`

These should be treated as the target pinned environment for more stable reproducibility, but not as a perfect historical description of every previously generated empirical artifact.

## 3. Minimal Setup Scenario

Recommended setup for the current snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

If work is done inside the already existing `.venv314`, local verification can use:

```bash
source .venv314/bin/activate
python -m pytest -q
```

## 4. Remaining Alignment Required

For article-ready submission, the following still need alignment:

1. the environment of record for saved `models/` and `reports/`;
2. the pinned Python version and the environment actually used for runs;
3. runbook commands and local paths to raw/prepared datasets.

# Vercel deployment

ReviewGuard is exposed to Vercel through the root [`index.py`](../index.py) FastAPI entry point. Vercel reads the Python version from `.python-version` and the small API-only dependency set from the base `[project.dependencies]` in [`pyproject.toml`](../pyproject.toml). Training and inference libraries are optional extras, so the Vercel function stays within the platform bundle budget when no checkpoint is deployed. [`vercel.json`](../vercel.json) excludes tests, research artifacts, and training code from the function bundle.

The GitHub Actions workflow runs tests first. After a successful push build, it deploys a preview for non-default branches and production for the repository default branch. Pull requests run CI but do not receive deployment credentials. Production deployments use the GitHub `production` environment; configure required reviewers there if production approval is desired.

## One-time Vercel setup

1. Import `Svoppy/review-api` into Vercel and use the repository root as the project root. Keep `dissertation-mvp` as the production branch unless the repository default branch changes.
2. In Vercel project settings, confirm Python 3.12 and that the project recognizes the FastAPI app. The included `.python-version` selects the runtime.
3. Add the following GitHub Actions repository variables under **Settings → Secrets and variables → Actions → Variables**:
   - `VERCEL_ORG_ID`
   - `VERCEL_PROJECT_ID`
   - `VERCEL_DEPLOY_ENABLED` = `true`
4. Add `VERCEL_TOKEN` as an Actions secret. Do not commit it or paste it into an issue or chat. The workflow exposes it only to Vercel CLI authentication steps.
5. If you want approvals before production deployments, configure required reviewers for the GitHub `production` environment.
6. Push to a non-default branch to create a Vercel Preview deployment. A successful push to the default branch creates a Production deployment.

The CI workflow currently skips its deploy job until `VERCEL_DEPLOY_ENABLED` is `true`, so the repository remains green before the Vercel project and credentials are configured.

## Runtime scope

The deployed service includes the UI, `/health`, `/research-context`, and `/analyze` routes. Model checkpoints under `models/` are ignored by Git and excluded from deployment. Until a checkpoint is provided through an external model store and configured for the function, `/health` reports `model_ready: false` and `/analyze` returns the existing `503` not-ready response. The lightweight deployment intentionally omits PyTorch and Transformers; those remain available as the `inference` extra for a separate model-serving environment. The Vercel deployment is therefore a live API shell; it does not include a trained model by default.

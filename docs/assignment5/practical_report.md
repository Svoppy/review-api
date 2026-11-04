# Practical Report: Docker and Jenkins CI for ReviewGuard

**Author: Dias Kazikhanov**

## Project and repository

ReviewGuard is a Python research project for sentiment and authenticity analysis of online reviews. The project repository is [github.com/Svoppy/review-api](https://github.com/Svoppy/review-api). The root `Jenkinsfile` defines the CI stages, and the root `Dockerfile` defines the API image.

## Implementation

The Docker image uses Python 3.12, installs the project as a package, runs Uvicorn as a non-root user, and exposes port 8000. Inference dependencies are included by default. Local datasets, model weights, generated reports, and development files are excluded from the build context; a trained checkpoint is mounted separately at `/models/latest`. The `INSTALL_INFERENCE=0` build argument produces a smaller image for testing API startup and `/health` without downloading the large model stack.

The local Jenkins server is defined in `devops/jenkins/compose.yml` and built from `devops/jenkins/Dockerfile`. Jenkins loads the pipeline from the repository's `Jenkinsfile`. Its stages install the declared research and development dependencies in Python 3.12, run `PYTHONPATH=src python -m pytest -q`, build the Docker image, and start a temporary container to check `/health`. Docker Hub publishing and Kubernetes deployment are not enabled; credentials and a target cluster are not configured.

The repository already had a GitHub Actions workflow that runs pytest on pushes and pull requests. The Jenkins setup adds the requested self-hosted workflow rather than replacing the existing CI. The local demonstration used a temporary Git snapshot of the current working tree because the Assignment 5 files are not yet published to the public remote. The Jenkins job itself is configured to load its pipeline definition from SCM.

## Pipeline evidence

Jenkins build **#10** completed successfully from revision `e9e91b1` of the local SCM snapshot. All **152 tests passed in 3.64 seconds**; Jenkins then built `reviewguard-ci:10`, started a temporary container, and received a successful response from `/health`. The accompanying Jenkins Stage View screenshot records the green result. The local server is available at `http://127.0.0.1:8080` while its Docker Compose service is running.

## Difficulties and conclusions

The machine did not have a running Docker daemon at first, so a local Docker runtime was started. Jenkins also needed its Docker CLI, Git in the Python test agent, packaged FastAPI static assets, and a writable local home directory. The initial health check exposed that the API package omitted its HTML file; adding it to package data fixed the container startup. The CPU-only runner also uses the CPU PyTorch wheel so CI does not download unused CUDA libraries. The Docker socket and Jenkins endpoint are kept local for this demonstration; the Compose file should be adapted to the organization's security and infrastructure before a shared deployment.

The run demonstrates that repository code can be tested, packaged, and started through one Jenkins pipeline. The health check confirms that the API process starts; it does not establish model accuracy or scientific validity. For a shared runner, the next steps are to publish the assignment changes, connect Jenkins to the public GitHub repository with webhooks, pin dependency/image versions, and store model artifacts with explicit provenance. The existing GitHub Actions workflow continues to run the full test suite on pushes and pull requests.

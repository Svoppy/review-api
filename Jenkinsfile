pipeline {
    agent any

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        REVIEWGUARD_IMAGE = "reviewguard-ci:${BUILD_NUMBER}"
    }

    stages {
        stage('Unit tests') {
            agent {
                docker {
                    image 'python:3.12-slim'
                    args '--user root:root --volume reviewguard-pip-cache:/root/.cache/pip'
                    reuseNode true
                }
            }
            steps {
                sh '''
                    apt-get update
                    apt-get install --no-install-recommends --yes git
                    rm -rf /var/lib/apt/lists/*
                    python -m pip install --upgrade pip setuptools wheel
                    # This runner is CPU-only; avoid downloading CUDA runtime wheels.
                    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
                    python -m pip install -e ".[research,dev]"
                    PYTHONPATH=src python -m pytest -q
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build --build-arg INSTALL_INFERENCE=0 --tag "$REVIEWGUARD_IMAGE" .'
            }
        }

        stage('Container health check') {
            steps {
                sh '''
                    set -eu
                    container="reviewguard-smoke-${BUILD_NUMBER}"
                    cleanup() { docker rm --force "$container" >/dev/null 2>&1 || true; }
                    trap cleanup EXIT
                    docker run --detach --name "$container" --publish 18080:8000 "$REVIEWGUARD_IMAGE"
                    attempt=0
                    until docker exec "$container" python -c "from urllib.request import urlopen; print(urlopen('http://127.0.0.1:8000/health', timeout=2).read().decode())"; do
                        attempt=$((attempt + 1))
                        if [ "$attempt" -ge 20 ]; then
                            docker logs "$container"
                            exit 1
                        fi
                        sleep 2
                    done
                '''
            }
        }
    }

    post {
        always {
            echo "ReviewGuard CI finished with status: ${currentBuild.currentResult}"
        }
    }
}

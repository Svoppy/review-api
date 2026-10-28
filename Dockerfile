# Build with `docker build -t reviewguard:local .`.
# Set INSTALL_INFERENCE=0 for a smaller API/health-check image without model execution.
FROM python:3.12-slim AS runtime

ARG INSTALL_INFERENCE=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REVIEWGUARD_CHECKPOINT_DIR=/models/latest

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 reviewguard \
    && useradd --uid 10001 --gid reviewguard --create-home --shell /usr/sbin/nologin reviewguard

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ "$INSTALL_INFERENCE" = "1" ]; then \
         python -m pip install ".[inference]"; \
       else \
         python -m pip install .; \
       fi

USER reviewguard
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "reviewguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

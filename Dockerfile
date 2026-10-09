# Multi-stage build: smaller final image, no build toolchain shipped to prod.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

# Run as a non-root user. Basic container hygiene, and the kind of thing
# a firm that cares about data protection will ask about.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=cre-default-risk \
    MODEL_ALIAS=champion

USER appuser
EXPOSE 8000

# Container-native health check so the orchestrator knows when we're ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

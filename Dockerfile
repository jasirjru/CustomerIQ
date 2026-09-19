# ==============================================================================
# CustomerIQ — Production API Dockerfile
# Minimal Linux container for FastAPI model serving
# ==============================================================================

# Base image: Official lightweight Python Debian bookworm-slim
FROM python:3.11.16-slim-bookworm

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED: Ensures stdout and stderr streams are sent straight to terminal/logs without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements-api.lock .

# Install Python dependencies without storing pip wheel cache (reduces image size)
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-api.lock

# Packaging tools are not needed after dependency installation and would add
# unnecessary runtime vulnerability surface to the serving image.
RUN python -m pip uninstall --yes pip setuptools wheel

# Copy only inference source, registered serving artifacts, and configuration
COPY src/__init__.py /app/src/__init__.py
COPY src/api/ /app/src/api/
RUN mkdir -p /app/models
COPY models/manifest.v1.json /app/models/manifest.v1.json
COPY models/preprocessor.joblib /app/models/preprocessor.joblib
COPY models/champion_model.joblib /app/models/champion_model.joblib
COPY config.py /app/config.py
COPY LICENSE /app/LICENSE

# Run inference as an unprivileged user. Artifacts remain read-only at runtime.
RUN groupadd --system customeriq && \
    useradd --system --gid customeriq --home-dir /nonexistent --no-create-home customeriq
USER customeriq

# Expose the port Uvicorn listens on
EXPOSE 8000

# Container healthcheck: ping the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

# Command to execute when the container starts
# Bind to 0.0.0.0 and use $PORT provided by cloud host (Render, Railway), fallback to 8000
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

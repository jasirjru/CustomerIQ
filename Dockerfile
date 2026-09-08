# ==============================================================================
# CustomerIQ — Production API Dockerfile
# Multi-stage lightweight Linux container for FastAPI model serving
# ==============================================================================

# Base image: Official lightweight Python Debian bookworm-slim
FROM python:3.11-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED: Ensures stdout and stderr streams are sent straight to terminal/logs without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies without storing pip wheel cache (reduces image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, serialized models, and configuration
COPY src/ /app/src/
COPY models/ /app/models/
COPY config.py /app/config.py

# Expose the port Uvicorn listens on
EXPOSE 8000

# Container healthcheck: ping the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to execute when the container starts
# Bind to 0.0.0.0 and use $PORT provided by cloud host (Render, Railway), fallback to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

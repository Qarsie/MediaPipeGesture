# syntax=docker/dockerfile:1

# ============================================
# ASL Recognition API - Production Dockerfile
# Multi-stage build for optimized image size
# ============================================

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================
# Final Stage - Production Runtime
# ============================================
FROM python:${PYTHON_VERSION}-slim AS runtime

# Labels for image metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="ASL Recognition API with MediaPipe and TensorFlow"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MEDIAPIPE_DISABLE_GPU=1
# Fix matplotlib home directory issue
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HOME=/tmp

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security with a proper home directory
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser \
    && mkdir -p /home/appuser/.config \
    && chown -R appuser:appuser /home/appuser

# Copy virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Copy application code
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser realtime_predict.py .
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser utils/ ./utils/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]

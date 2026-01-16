# FrameSift Backend Dockerfile
# Multi-stage build with pre-downloaded CLIP model for instant cold starts

FROM python:3.11-slim as builder

# Install build dependencies including ffmpeg for Phase 2 audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# CRITICAL: Pre-download the CLIP model during build
# This eliminates cold start delays in production
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"

# ============================================================================
# Production Stage
# ============================================================================

FROM python:3.11-slim

# Install runtime dependencies only (including ffmpeg for Phase 2 audio)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 framesift

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded models from builder
COPY --from=builder /root/.cache /home/framesift/.cache

# Copy application code (Phase 2: Now includes scout.py and processor.py)
COPY main.py scout.py processor.py ./

# Set ownership
RUN chown -R framesift:framesift /app /home/framesift/.cache

# Switch to non-root user
USER framesift

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Hugging Face Spaces uses 7860, but we'll support both)
EXPOSE 7860
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/')" || exit 1

# Start the application on port 7860 for Hugging Face Spaces
# Can be overridden with PORT env variable for other platforms
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}

FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to keep image small)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY fusion_resnet.py .
COPY fryze_utils.py .
COPY deploy/ deploy/
COPY checkpoints/best.pt checkpoints/best.pt

# Environment
ENV MODEL_CHECKPOINT=/app/checkpoints/best.pt
ENV DEVICE=cpu
ENV MAX_BATCH=1024

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

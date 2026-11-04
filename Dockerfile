FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/home/appuser \
    PORT=7860 \
    API_PORT=8000 \
    API_URL=http://127.0.0.1:8000

WORKDIR /app

# Install native dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user (UID 1000 required for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/checkpoints /app/dataset /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Install PyTorch CPU and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and adjust ownership
COPY --chown=appuser:appuser . .

# Ensure entrypoint is executable
RUN chmod +x docker/entrypoint.sh

# Hugging Face Spaces routes web traffic strictly to port 7860
EXPOSE 7860 8000

# Switch to non-root execution
USER appuser

# Healthcheck against internal FastAPI microservice
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Launch supervisor entrypoint (starts FastAPI on :8000 and Streamlit on :7860)
CMD ["bash", "docker/entrypoint.sh"]

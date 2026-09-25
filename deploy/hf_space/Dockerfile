FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache/huggingface

# Install system dependencies including FFmpeg and graphic libraries for OpenCV/InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create cache and upload directories
RUN mkdir -p /app/cache /app/webapp/uploads /app/checkpoints/full && \
    chmod -R 777 /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose standard Hugging Face Spaces application port
EXPOSE 7860

# Run FastAPI via uvicorn on port 7860
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "7860"]

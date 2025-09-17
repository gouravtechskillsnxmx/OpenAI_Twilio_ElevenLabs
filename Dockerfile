# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# ffmpeg needed for any audio conversions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

EXPOSE ${PORT}

# Use uvicorn (single worker to avoid concurrency problems w/ synchronous S3/OpenAI calls)
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

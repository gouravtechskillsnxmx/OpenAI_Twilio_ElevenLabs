# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system deps: ffmpeg, libpq-dev (psycopg2), postgresql-client (psql), etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      libpq-dev \
      postgresql-client \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code into container
COPY . /app

# Copy entrypoint and make executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE ${PORT}

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


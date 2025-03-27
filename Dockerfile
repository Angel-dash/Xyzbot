# Declare build arguments at the top (for initial stage)
ARG USER_UID=1000
ARG USER_GID=1000

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim

# Re-declare build arguments for this stage
ARG USER_UID=1000
ARG USER_GID=1000

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY . .

# Create the group and user first
RUN groupadd -g ${USER_GID} appuser && \
    useradd -m -u ${USER_UID} -g appuser appuser

# Set environment variables for persistent storage
ENV CHROMA_PERSISTENCE_DIRECTORY=/data/chromadb
ENV TRANSCRIPTS_FOLDER=/data/transcripts

# Optionally create directories (not strictly necessary as code handles it)
RUN mkdir -p /data/chromadb /data/transcripts

# Create directories with correct permissions (remove unnecessary chromadb.db)
RUN mkdir -p /app/Rag /app/Data && \
    chown -R appuser:appuser /app

USER appuser

CMD ["python", "app.py"]
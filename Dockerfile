FROM python:3.12-slim

# Install git, curl, and Docker CLI (for sandbox self-improvement tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
        https://download.docker.com/linux/debian \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/      ./app/
COPY sandbox/  ./sandbox/
COPY config/   ./config/

RUN mkdir -p /workspace /config /state /cache

# Ensure app is importable as a package
RUN touch /app/app/__init__.py /app/sandbox/__init__.py

# Pre-download the ChromaDB ONNX embedding model so it's baked into the image
# and containers don't need internet access at runtime.
RUN python3 -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

# Venv init script — creates /workspace/venv on first container start
COPY sandbox/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8090 9000

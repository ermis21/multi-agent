FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download piper voice model (~28 MB) at build time so container works offline
RUN mkdir -p /models && \
    curl -L -o /models/en_US-ryan-low.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx" && \
    curl -L -o /models/en_US-ryan-low.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json"

# Download Kokoro ONNX model + voices file (~330 MB total) for the new TTS backend.
# Files come from the upstream onnx-community release. Container starts offline-capable.
RUN curl -L -o /models/kokoro-v1.0.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" && \
    curl -L -o /models/voices-v1.0.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]

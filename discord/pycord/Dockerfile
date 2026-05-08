# Pycord variant of phoebe-discord. Same base as ../Dockerfile but installs
# py-cord instead of discord.py. The two libs share the `discord` namespace
# so they cannot coexist in one image.
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY discord/pycord/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Same model layout as the legacy container so volumes carry over.
RUN mkdir -p /models && \
    curl -L -o /models/en_US-ryan-low.onnx \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx" && \
    curl -L -o /models/en_US-ryan-low.onnx.json \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/low/en_US-ryan-low.onnx.json"

RUN curl -L -o /models/kokoro-v1.0.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" && \
    curl -L -o /models/voices-v1.0.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Bot source lives in /app — the compose mount layers individual files on top
# (see phoebe-discord-pycord service definition). Skeleton main.py is here
# so the container still builds even before any code is bind-mounted.
COPY discord/pycord/main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]

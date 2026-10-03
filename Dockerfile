# Syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS runtime

LABEL maintainer="Hawk Point NPU LLM Project"
LABEL description="Containerized runtime environment for Hawk Point XDNA1 NPU LLM"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    libboost-all-dev \
    libdrm-dev \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app --no-create-home app

# Create virtualenv and install locked dependencies
RUN python3 -m venv /app/venv
COPY requirements.lock /app/requirements.lock
RUN /app/venv/bin/pip install --no-cache-dir --require-hashes -r /app/requirements.lock

# Copy application source
COPY . /app

# Create models directory mount point and set ownership
RUN mkdir -p /app/models && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["python3"]
CMD ["npu_llm/api_server.py", "--host", "0.0.0.0", "--port", "8000", "--models-dir", "/app/models"]

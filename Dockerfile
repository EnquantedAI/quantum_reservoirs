# Environment aligned with requirements_staszek.txt (Python 3.13.x)
FROM python:3.13-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Optional tooling; keeps the scientific stack closer to a typical dev machine
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_staszek.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements_staszek.txt

COPY . .

# Default: interactive shell (docker-compose overrides this for Jupyter Lab)
CMD ["bash"]

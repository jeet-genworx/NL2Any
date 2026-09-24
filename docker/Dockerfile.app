# Dockerfile for the NL2AnyQuery Python application (API, Streamlit UI, CLI/seed scripts)
FROM python:3.12-slim

# Build deps needed for psycopg (libpq) and general compilation
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/*.sources 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager used by this project)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --extra dev --no-install-project

# Copy the rest of the application source
COPY . .

# Install the project itself into the venv created by uv sync
RUN uv sync --extra dev

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

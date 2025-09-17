# syntax=docker/dockerfile:1.4
# the preceding line lets us use .gitignore for exclusions

FROM python:3.13-slim

WORKDIR /app

# Install FFmpeg
RUN apt-get update && \
  apt-get install -y ffmpeg && \
  rm -rf /var/lib/apt/lists/*

RUN pip install uv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml .
COPY uv.lock .

# do the initial sync for dependencies from the lock file
RUN uv sync \
  --no-install-project --no-install-workspace --no-sources \
  --frozen

COPY . .

# do the full sync including installing the project etc
RUN uv sync --frozen

ENV IEVA_DATA=/data

CMD ["uv", "run", "fastapi", "run", "src.slop.transcribe:app", "--host", "0.0.0.0", "--port", "8080"]

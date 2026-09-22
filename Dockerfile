FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev && mkdir -p /app/artifacts/transformer

CMD ["uv", "run", "uvicorn", "relationscope.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

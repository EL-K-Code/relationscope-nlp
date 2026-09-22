FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RELATIONSCOPE_MODEL_DIR=/tmp/relationscope/model

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv sync --no-dev \
    && useradd --create-home --uid 10001 relationscope \
    && mkdir -p /tmp/relationscope \
    && chown -R relationscope:relationscope /tmp/relationscope /app

ENV PATH="/app/.venv/bin:$PATH"

USER relationscope

CMD ["uvicorn", "relationscope.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

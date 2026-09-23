FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    RELATIONSCOPE_MODEL_DIR=/tmp/relationscope/model

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/
COPY pyproject.toml README.md ./
COPY src ./src

RUN test -f src/relationscope/models/entity_classifier.py \
    && uv pip install \
      --system \
      --no-cache \
      --torch-backend=cpu \
      . \
    && python -c "from relationscope.models.entity_classifier import EntityAwareRelationClassifier" \
    && useradd --create-home --uid 10001 relationscope \
    && mkdir -p /tmp/relationscope \
    && chown -R relationscope:relationscope /tmp/relationscope /app

USER relationscope

CMD ["uvicorn", "relationscope.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

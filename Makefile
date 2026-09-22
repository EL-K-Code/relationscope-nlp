.PHONY: install install-legacy data prepare test lint web-install web-dev api-dev docker-up

install:
	uv sync --all-groups

install-legacy:
	uv sync --all-groups --extra legacy

data:
	uv run python scripts/download_fewrel.py

prepare:
	uv run python scripts/prepare_fewrel.py
	uv run python scripts/prepare_open_set_fewrel.py

test:
	uv run pytest

lint:
	uv run ruff check .

api-dev:
	uv run uvicorn relationscope.api.main:app --reload --port 8080

web-install:
	cd apps/web && npm install

web-dev:
	cd apps/web && npm run dev

docker-up:
	docker compose up --build

# Local development topology

RelationScope runs as two services:

```text
browser
   |
   v
Next.js :3000
   |
   | same-origin /api/relationscope/*
   v
FastAPI :8080
   |
   v
model bundle
```

The browser never needs the private API URL.

## Without a trained model

Both applications can be developed before training:

- `GET /health` remains available;
- the web interface renders normally;
- model-dependent endpoints return a clear unavailable state.

## Docker Compose

After a model bundle exists at `artifacts/transformer/best`:

```bash
docker compose up --build
```

Open:

```text
http://localhost:3000
```

The API remains available locally on port 8080 for development and OpenAPI inspection.

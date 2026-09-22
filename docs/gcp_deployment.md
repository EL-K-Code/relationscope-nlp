# GCP deployment architecture

RelationScope is designed as two Cloud Run services with least-privilege service identities.

```text
Internet
   |
   v
PUBLIC Cloud Run
relationscope-web
Next.js
   |
   | Google-signed ID token
   | roles/run.invoker
   v
PRIVATE Cloud Run
relationscope-api
FastAPI
   |
   | roles/storage.objectViewer
   v
GCS model bucket
versioned model bundle
```

The browser never receives the private FastAPI URL or a Google identity token.

The Next.js route handler obtains a Google-signed ID token from Application Default Credentials when `RELATIONSCOPE_API_AUDIENCE` is configured. Local development leaves this variable empty and uses direct service-to-service HTTP inside Docker Compose.

## Infrastructure

`infra/terraform` manages:

- required Google APIs;
- Artifact Registry;
- versioned GCS model storage;
- API and web service accounts;
- private FastAPI Cloud Run;
- public Next.js Cloud Run;
- `roles/run.invoker` from web identity to API only;
- read-only model-bucket access for the API identity.

## Terraform state bootstrap

One bucket is needed before Terraform can use its GCS backend:

```bash
bash scripts/bootstrap_gcp_state.sh \
  YOUR_PROJECT_ID \
  europe-west1 \
  YOUR_TERRAFORM_STATE_BUCKET
```

Then initialize:

```bash
cd infra/terraform

terraform init \
  -backend-config="bucket=YOUR_TERRAFORM_STATE_BUCKET" \
  -backend-config="prefix=relationscope"
```

Copy `terraform.tfvars.example` to `terraform.tfvars` and replace placeholders.

## Why model artifacts are separate from the image

Training and serving have separate lifecycles.

The API image contains code only. The promoted model bundle is stored in GCS and downloaded on first model-dependent request. This lets us:

- roll code without retraining;
- promote another model without rebuilding application source;
- keep large weights out of Git;
- verify exact model files using SHA-256.

Publish a chosen bundle:

```bash
uv run python scripts/publish_model_bundle.py \
  artifacts/open_set/best \
  gs://YOUR_MODEL_BUCKET/models/current \
  --model-version v1
```

## GitHub deployment

`.github/workflows/deploy-gcp.yml` is manual and uses Workload Identity Federation: no long-lived JSON service-account key is required.

Required repository variables:

```text
GCP_PROJECT_ID
GCP_REGION
GCP_WIF_PROVIDER
GCP_DEPLOY_SERVICE_ACCOUNT
TF_STATE_BUCKET
MODEL_BUCKET
GAR_REPOSITORY
```

The workflow:

1. authenticates through Workload Identity Federation;
2. creates/refreshes bootstrap infrastructure with targeted Terraform resources;
3. builds API and web images;
4. pushes them to Artifact Registry;
5. applies the complete Terraform stack with immutable commit-SHA image tags.

Training is deliberately not part of deployment.

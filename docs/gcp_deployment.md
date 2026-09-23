# GCP deployment architecture

RelationScope V2 uses two Cloud Run services, two purpose-separated GCS buckets, and a bounded Cloud Tasks queue.

```text
Internet
   |
   v
PUBLIC Cloud Run
relationscope-web
Next.js
   |
   | Google-signed ID token
   v
PRIVATE Cloud Run
relationscope-api
FastAPI
   |
   +---- roles/storage.objectViewer ----> GCS model bucket
   |
   +---- roles/storage.objectAdmin -----> GCS operations bucket
   |
   +---- roles/cloudtasks.enqueuer -----> Cloud Tasks queue
                                              |
                                              | OIDC
                                              v
                                      PRIVATE Cloud Run API
                                      /v2/internal/batch-jobs/*
```

The browser never receives a Google identity token. The Next.js route handler uses the attached web service account to obtain a Google-signed ID token for the private API.

The public proxy explicitly rejects `/v2/internal/*`.

## Infrastructure

`infra/terraform` manages:

- required Google APIs, including Cloud Tasks;
- Artifact Registry;
- versioned GCS model storage;
- private V2 operations GCS storage;
- API, web and task-caller service accounts;
- private FastAPI Cloud Run;
- public Next.js Cloud Run;
- Cloud Tasks queue for durable batch inference;
- `roles/run.invoker` from web identity to API;
- `roles/run.invoker` from task identity to API;
- read-only model-bucket access for the API;
- read/write operations-bucket access for the API;
- API permission to enqueue Cloud Tasks;
- API permission to use the dedicated task-caller service account.

## Terraform state bootstrap

The state bucket is created once before Terraform begins managing the application stack.

Initialize with the production prefix used by the current deployment:

```bash
cd infra/terraform

terraform init \
  -backend-config="bucket=YOUR_TERRAFORM_STATE_BUCKET" \
  -backend-config="prefix=terraform/state"
```

The manual GitHub deployment workflow uses the same `terraform/state` prefix.

Copy `terraform.tfvars.example` to `terraform.tfvars` and replace placeholders. The file is intentionally ignored by Git.

## Operations bucket

V2 stores telemetry, feedback and durable batch state in a bucket separate from model artifacts.

By default Terraform derives:

```text
<project_id>-relationscope-ops
```

Override it with `ops_bucket_name` if necessary.

Default lifecycle retention:

```text
90 days
```

The bucket has `force_destroy = false`, preventing an accidental Terraform destroy from silently deleting non-empty operational data.

## Model lifecycle

The serving image contains code, not model weights.

Register the already-trained V1 model as an immutable release:

```bash
uv run python scripts/register_model_release.py \
  artifacts/open_set/best \
  --bucket YOUR_MODEL_BUCKET \
  --model-version modernbert-open-set-v1.1
```

This creates:

```text
gs://YOUR_MODEL_BUCKET/models/releases/modernbert-open-set-v1.1/
```

Promote a registered model:

```bash
uv run python scripts/promote_model_release.py \
  --bucket YOUR_MODEL_BUCKET \
  --model-version modernbert-open-set-v1.1
```

Promotion updates `models/current` and records an audit JSON under `models/promotions/`.

The API continues to load only the promoted `models/current` alias.

## Async batch execution

The API creates durable job state before enqueueing work.

Terraform provisions the queue with conservative portfolio defaults:

- max concurrent dispatches: 1;
- max dispatch rate: 1 request/s;
- maximum task attempts: 3;
- bounded retry backoff.

The API Cloud Run request timeout is 900 seconds so one bounded batch worker callback can finish its chunked inference.

## GitHub deployment

`.github/workflows/deploy-gcp.yml` is manual and uses Workload Identity Federation: no long-lived service-account JSON key is required.

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
2. initializes the exact production Terraform state;
3. bootstraps required services, registries, buckets and identities;
4. builds API and web images;
5. pushes them to Artifact Registry;
6. applies the complete Terraform stack.

Training is deliberately not part of deployment.

## Manual deployment from the current workstation

For the current Cloud Build workflow used during development:

1. build API and web images with Cloud Build;
2. resolve their immutable Artifact Registry digests;
3. place those digests in the ignored local `terraform.tfvars`;
4. run a complete `terraform plan`;
5. review for unexpected replacements/destruction;
6. apply the saved plan;
7. smoke-test the public web proxy and private service-to-service path.

Do not deploy V2 by applying Terraform with new infrastructure while still pointing at an untested V2 image. Build and validate the images first, then apply the infrastructure and image update together.

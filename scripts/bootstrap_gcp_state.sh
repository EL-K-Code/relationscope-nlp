#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="\${1:?usage: bootstrap_gcp_state.sh PROJECT_ID REGION STATE_BUCKET}"
REGION="\${2:?usage: bootstrap_gcp_state.sh PROJECT_ID REGION STATE_BUCKET}"
STATE_BUCKET="\${3:?usage: bootstrap_gcp_state.sh PROJECT_ID REGION STATE_BUCKET}"

gcloud config set project "$PROJECT_ID"
gcloud services enable storage.googleapis.com

if gcloud storage buckets describe "gs://$STATE_BUCKET" >/dev/null 2>&1; then
  echo "Terraform state bucket already exists: gs://$STATE_BUCKET"
else
  gcloud storage buckets create "gs://$STATE_BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access
fi

gcloud storage buckets update "gs://$STATE_BUCKET" --versioning

echo "State bucket ready: gs://$STATE_BUCKET"

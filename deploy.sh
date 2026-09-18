#!/usr/bin/env bash
set -euo pipefail
: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to your approved project ID}"
region="${CLOUD_RUN_REGION:-asia-southeast1}"
gcloud run deploy doors-are-closing --source . --project "$GOOGLE_CLOUD_PROJECT" \
  --region "$region" --allow-unauthenticated --cpu 2 --memory 4Gi \
  --concurrency 1 --timeout 300 --max-instances 3 --min-instances 0

#!/bin/sh
set -e
# Train automatically on first start if no model exists in the mounted volume
if [ ! -f models/churn_model.joblib ]; then
  echo "[entrypoint] No model found -> running training pipeline..."
  python -m src.pipelines.training_pipeline
fi
exec "$@"

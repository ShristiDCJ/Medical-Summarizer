#!/bin/bash
# Startup script for Render deployment
# This script runs before the application starts

# Create artifacts directory if it doesn't exist
mkdir -p /opt/render/project/src/artifacts

# If you're using cloud storage (S3, GCS, etc.), download artifacts here
# Example for S3:
# aws s3 cp s3://your-bucket/model.pt /opt/render/project/src/artifacts/model.pt
# aws s3 cp s3://your-bucket/tokenizer.json /opt/render/project/src/artifacts/tokenizer.json

# Example for GCS:
# gsutil cp gs://your-bucket/model.pt /opt/render/project/src/artifacts/model.pt
# gsutil cp gs://your-bucket/tokenizer.json /opt/render/project/src/artifacts/tokenizer.json

echo "Startup script completed"


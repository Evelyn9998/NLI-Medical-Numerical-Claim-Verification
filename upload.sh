#!/bin/bash
# Upload project files to SLURM cluster
# Usage: bash upload.sh

set -e

SERVER="mewa8567@olympus.dsv.su.se"
REMOTE="~/my_project/"

echo "Uploading to ${SERVER}:${REMOTE} ..."

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='wandb/' \
    --exclude='baseline_reslut/' \
    --exclude='*.ipynb_checkpoints' \
    --exclude='results/' \
    baseline_test/ \
    data/train_306.csv \
    requirements.txt \
    slurm-run.sh \
    check-status.sh \
    "${SERVER}:${REMOTE}"

echo ""
echo "Upload done. Verifying remote files:"
ssh "${SERVER}" "ls -lh ${REMOTE} && echo '---' && ls -lh ${REMOTE}baseline_test/ && echo '---' && ls -lh ${REMOTE}data/"

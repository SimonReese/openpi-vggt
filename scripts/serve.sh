#!/bin/bash
# Print cuda devices available
echo "Executing on GPU ${CUDA_VISIBLE_DEVICES}"

# Variables
WORKDIR="/home/peraro/source/openpi-vggt"
CONFIG_NAME="pi05_rlbench"
CHECKPOINT_DIR="${WORKDIR}/checkpoints/pi05_rlbench/pi05_rlbench/29999"

# Disable preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run openpi/scripts/serve_policy.py \
    --port 4900 \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CHECKPOINT_DIR}" 
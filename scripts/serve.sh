#!/bin/bash
# Print cuda devices available
echo "Executing on GPU ${CUDA_VISIBLE_DEVICES}"

# Setup project path
# Dir of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Dir of project
WORKDIR="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"
# Variables
CONFIG_NAME="pi05_rlbench"
CHECKPOINT_DIR="${WORKDIR}/checkpoints/pi05_rlbench/pi05_rlbench_vggt_arrange/29999"

# Disable preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run openpi/scripts/serve_policy.py \
    --port 4900 \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CHECKPOINT_DIR}" 

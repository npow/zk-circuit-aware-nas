#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --no-cache-dir torchvision==0.21.0 huggingface_hub
cd /workspace/repo
export PYTHONPATH="src:.${PYTHONPATH:+:${PYTHONPATH}}"
exec python gpu_validation_experiment.py "$@"

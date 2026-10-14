#!/usr/bin/env bash
#SBATCH --job-name=gokturk-3p3b-qlora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gokturk-qlora-%j.out

# EVREN Slurm partition, account, container image and mount policy are account-specific.
# Add the portal-provided --partition/--account options at sbatch time if your project requires them.
# This template asks for one GPU; train_qlora_h200.py checks that the allocated device is H200.
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

TRAIN_CONFIG="${GOKTURK_TRAIN_CONFIG:-configs/train_h200.yaml}"
TRAIN_DATA="${GOKTURK_TRAIN_DATA:-data/gokturk33b_teacher.accepted.jsonl}"
OUTPUT_DIR="${GOKTURK_TRAIN_OUTPUT:-outputs/gokturk-3.3b-h200}"

if [[ -n "${EVREN_APPTAINER_IMAGE:-}" ]]; then
  command -v apptainer >/dev/null || { echo "apptainer command not found" >&2; exit 2; }
  CONTAINER_PYTHON="${EVREN_CONTAINER_PYTHON:-python}"
  apptainer exec --nv --bind "$REPO_ROOT:$REPO_ROOT" --pwd "$REPO_ROOT" \
    "$EVREN_APPTAINER_IMAGE" \
    "$CONTAINER_PYTHON" scripts/train_qlora_h200.py \
      --config "$TRAIN_CONFIG" --train-data "$TRAIN_DATA" --output "$OUTPUT_DIR"
else
  HOST_PYTHON="${GOKTURK_PYTHON:-python}"
  if [[ "$HOST_PYTHON" == "python" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
    HOST_PYTHON="$REPO_ROOT/.venv/bin/python"
  fi
  "$HOST_PYTHON" scripts/train_qlora_h200.py \
    --config "$TRAIN_CONFIG" --train-data "$TRAIN_DATA" --output "$OUTPUT_DIR"
fi

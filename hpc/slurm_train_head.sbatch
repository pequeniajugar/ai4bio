#!/usr/bin/env bash
#SBATCH --job-name=aging-head
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts}"
FEATURE_ARCHIVE="${FEATURE_ARCHIVE:-${OUTPUT_DIR}/aging_loci.alphagenome_features.npz}"

mkdir -p "${OUTPUT_DIR}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

aging-train-head \
  --features "${FEATURE_ARCHIVE}" \
  --output-model "${OUTPUT_DIR}/aging_linear_head.npz" \
  --metrics "${OUTPUT_DIR}/aging_linear_head.metrics.json" \
  --learning-rate 0.001 \
  --l2 0.0001 \
  --epochs 300 \
  --batch-size 128 \
  --patience 30 \
  --seed 17

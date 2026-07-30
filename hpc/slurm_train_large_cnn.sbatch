#!/usr/bin/env bash
#SBATCH --job-name=aging-cnn-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
PREPARED_DATASET="${PREPARED_DATASET:-${PROJECT_ROOT}/data/processed/aging_loci.hg38.tsv}"
LARGE_CNN_OUTPUT_DIR="${LARGE_CNN_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/cnn-large}"
CNN_CACHE_DIR="${CNN_CACHE_DIR:-${SCRATCH_ROOT:-${PROJECT_ROOT}/.cache}/cnn-large}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CNN_CACHE_DIR}/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${CNN_CACHE_DIR}/jax}"

mkdir -p \
  "${LARGE_CNN_OUTPUT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

python -u -m aging_alphagenome.cnn \
  --dataset "${PREPARED_DATASET}" \
  --output-model "${LARGE_CNN_OUTPUT_DIR}/aging_201bp_large_cnn.npz" \
  --metrics "${LARGE_CNN_OUTPUT_DIR}/aging_201bp_large_cnn.metrics.json" \
  --test-predictions "${LARGE_CNN_OUTPUT_DIR}/aging_201bp_large_cnn.test_predictions.tsv" \
  --architecture large \
  --sequence-length 201 \
  --epochs 120 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --weight-decay 0.0005 \
  --dropout 0.40 \
  --patience 20 \
  --internal-validation-fraction 0.10 \
  --seed 17 \
  --require-gpu

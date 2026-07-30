#!/usr/bin/env bash
#SBATCH --job-name=aging-cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
PREPARED_DATASET="${PREPARED_DATASET:-${PROJECT_ROOT}/data/processed/aging_loci.hg38.tsv}"
CNN_OUTPUT_DIR="${CNN_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/cnn}"
CNN_CACHE_DIR="${CNN_CACHE_DIR:-${SCRATCH_ROOT:-${PROJECT_ROOT}/.cache}/cnn}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CNN_CACHE_DIR}/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${CNN_CACHE_DIR}/jax}"

mkdir -p \
  "${CNN_OUTPUT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

python -u -m aging_alphagenome.cnn \
  --dataset "${PREPARED_DATASET}" \
  --output-model "${CNN_OUTPUT_DIR}/aging_201bp_cnn.npz" \
  --metrics "${CNN_OUTPUT_DIR}/aging_201bp_cnn.metrics.json" \
  --test-predictions "${CNN_OUTPUT_DIR}/aging_201bp_cnn.test_predictions.tsv" \
  --sequence-length 201 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.25 \
  --patience 15 \
  --internal-validation-fraction 0.10 \
  --seed 17 \
  --require-gpu

#!/usr/bin/env bash
#SBATCH --job-name=aging-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ALPHAGENOME_CACHE="${ALPHAGENOME_CACHE:-${PROJECT_ROOT}/.cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts}"
PREPARED_DATASET="${PREPARED_DATASET:-${PROJECT_ROOT}/data/processed/aging_loci.hg38.tsv}"

: "${ALPHAGENOME_CHECKPOINT_DIR:?Set ALPHAGENOME_CHECKPOINT_DIR to the fold-0 checkpoint.}"

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${ALPHAGENOME_CACHE}/jax_compilation}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ALPHAGENOME_CACHE}/xdg}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

mkdir -p "${OUTPUT_DIR}" "${JAX_COMPILATION_CACHE_DIR}" "${XDG_CACHE_HOME}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

python -u -m aging_alphagenome.features \
  --dataset "${PREPARED_DATASET}" \
  --checkpoint "${ALPHAGENOME_CHECKPOINT_DIR}" \
  --output "${OUTPUT_DIR}/aging_loci.alphagenome_features.npz" \
  --feature-mode combined \
  --biological-window 201 \
  --batch-size 1

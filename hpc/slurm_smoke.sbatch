#!/usr/bin/env bash
#SBATCH --job-name=alphagenome-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ALPHAGENOME_CACHE="${ALPHAGENOME_CACHE:-${PROJECT_ROOT}/.cache}"

: "${ALPHAGENOME_CHECKPOINT_DIR:?Set ALPHAGENOME_CHECKPOINT_DIR to the downloaded fold checkpoint.}"

export HF_HOME="${HF_HOME:-${ALPHAGENOME_CACHE}/huggingface}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${ALPHAGENOME_CACHE}/jax_compilation}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ALPHAGENOME_CACHE}/xdg}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

mkdir -p \
  "${HF_HOME}" \
  "${JAX_COMPILATION_CACHE_DIR}" \
  "${XDG_CACHE_HOME}"

source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

echo "Job ID:       ${SLURM_JOB_ID:-not-running-under-slurm}"
echo "Node:         $(hostname)"
echo "Project:      ${PROJECT_ROOT}"
echo "Environment:  ${CONDA_ENV_NAME:-${ALPHAGENOME_ENV:-}}"
echo "Checkpoint:   ${ALPHAGENOME_CHECKPOINT_DIR}"
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python -u "${PROJECT_ROOT}/scripts/hpc/smoke_test.py" \
  --tsv "${PROJECT_ROOT}/RS_PDL50.wgs.rediportal.vcf.isec.tsv" \
  --require-gpu \
  --checkpoint "${ALPHAGENOME_CHECKPOINT_DIR}" \
  --run-inference

echo "Finished UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

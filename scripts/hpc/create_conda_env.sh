#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-aging-alphagenome}"
SCRATCH_ROOT="${SCRATCH_ROOT:-}"
JAX_CUDA_VARIANT="${JAX_CUDA_VARIANT:-cuda12}"
ALPHAGENOME_CLIENT_VERSION="${ALPHAGENOME_CLIENT_VERSION:-0.7.0}"
ALPHAGENOME_RESEARCH_REF="${ALPHAGENOME_RESEARCH_REF:-c5b51606a410b34c3e64d870ea4e034c3c0ca976}"
ALPHAGENOME_RESEARCH_SOURCE="${ALPHAGENOME_RESEARCH_SOURCE:-}"

case "${JAX_CUDA_VARIANT}" in
  cuda12|cuda13|cuda12-local|cuda13-local|cpu)
    ;;
  *)
    echo "Unsupported JAX_CUDA_VARIANT=${JAX_CUDA_VARIANT}" >&2
    echo "Use cuda12, cuda13, cuda12-local, cuda13-local, or cpu." >&2
    exit 2
    ;;
esac

# Keep Conda and pip artifacts off small home quotas when a scratch root is
# provided. Explicit CONDA_* or PIP_* values always take precedence.
if [[ -n "${SCRATCH_ROOT}" ]]; then
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH_ROOT}/conda/pkgs}"
  export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${SCRATCH_ROOT}/conda/envs}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip-cache}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg-cache}"
  mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load your site's Miniconda or Anaconda module." >&2
  exit 2
fi

# Make conda activate available in this non-interactive shell. Some site
# activation hooks reference optional variables without guarding them; keep
# Bash nounset disabled while Conda evaluates those hooks.
nounset_enabled=0
case "$-" in
  *u*) nounset_enabled=1; set +u ;;
esac
eval "$(conda shell.bash hook)"
if (( nounset_enabled )); then
  set -u
fi

existing_envs="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"
if printf '%s\n' "${existing_envs}" | grep -Fxq "${CONDA_ENV_NAME}"; then
  conda env update \
    --name "${CONDA_ENV_NAME}" \
    --file "${PROJECT_ROOT}/environment.yml"
else
  conda env create \
    --name "${CONDA_ENV_NAME}" \
    --file "${PROJECT_ROOT}/environment.yml"
fi

# Conda activation runs site-provided activate.d scripts, which may not be
# nounset-safe (for example, Qt hooks on some clusters).
nounset_enabled=0
case "$-" in
  *u*) nounset_enabled=1; set +u ;;
esac
conda activate "${CONDA_ENV_NAME}"
if (( nounset_enabled )); then
  set -u
fi
python -m pip install --upgrade pip setuptools wheel

if [[ "${JAX_CUDA_VARIANT}" == "cpu" ]]; then
  python -m pip install --upgrade jax
else
  python -m pip install --upgrade "jax[${JAX_CUDA_VARIANT}]"
fi

python -m pip install --upgrade "alphagenome==${ALPHAGENOME_CLIENT_VERSION}"

if [[ -n "${ALPHAGENOME_RESEARCH_SOURCE}" ]]; then
  if [[ ! -f "${ALPHAGENOME_RESEARCH_SOURCE}/pyproject.toml" ]]; then
    echo "ALPHAGENOME_RESEARCH_SOURCE is not an AlphaGenome research checkout:" >&2
    echo "  ${ALPHAGENOME_RESEARCH_SOURCE}" >&2
    exit 2
  fi
  RESEARCH_SPEC="${ALPHAGENOME_RESEARCH_SOURCE}"
else
  RESEARCH_SPEC="git+https://github.com/google-deepmind/alphagenome_research.git@${ALPHAGENOME_RESEARCH_REF}"
fi

# Reinstall from source so pyBigWig can build correctly on the cluster.
PIP_NO_BINARY=pyBigWig python -m pip install --upgrade "${RESEARCH_SPEC}"
python -m pip install --editable "${PROJECT_ROOT}"

python -m pip check
python -m pip freeze > "${CONDA_PREFIX}/requirements.freeze.txt"

echo
echo "AlphaGenome Conda environment is ready."
echo "Environment: ${CONDA_ENV_NAME} (${CONDA_PREFIX})"
echo "Python:      $(python --version 2>&1)"
echo "JAX target:  ${JAX_CUDA_VARIANT}"
echo "Activate:    conda activate ${CONDA_ENV_NAME}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ALPHAGENOME_ENV="${ALPHAGENOME_ENV:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
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

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python was not found. Load a Python 3.11+ module first." >&2
    exit 2
  fi
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' || {
  echo "Python 3.11+ is required; found ${PYTHON_VERSION}." >&2
  exit 2
}

mkdir -p "${ALPHAGENOME_ENV}"
if [[ ! -x "${ALPHAGENOME_ENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${ALPHAGENOME_ENV}"
fi

PYTHON="${ALPHAGENOME_ENV}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ "${JAX_CUDA_VARIANT}" == "cpu" ]]; then
  "${PYTHON}" -m pip install --upgrade jax
else
  "${PYTHON}" -m pip install --upgrade "jax[${JAX_CUDA_VARIANT}]"
fi

"${PYTHON}" -m pip install "alphagenome==${ALPHAGENOME_CLIENT_VERSION}"
"${PYTHON}" -m pip install --upgrade "huggingface_hub>=0.30"

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

# The upstream fine-tuning notebook builds pyBigWig locally. This avoids
# platform-specific wheel issues seen on some HPC Linux distributions.
PIP_NO_BINARY=pyBigWig "${PYTHON}" -m pip install "${RESEARCH_SPEC}"
"${PYTHON}" -m pip install --editable "${PROJECT_ROOT}"

"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze > "${ALPHAGENOME_ENV}/requirements.freeze.txt"

echo
echo "AlphaGenome environment is ready."
echo "Environment: ${ALPHAGENOME_ENV}"
echo "Python:      ${PYTHON_VERSION}"
echo "JAX target:  ${JAX_CUDA_VARIANT}"
echo "Activate:    source \"${ALPHAGENOME_ENV}/bin/activate\""
echo
echo "Next, run:"
echo "  python \"${PROJECT_ROOT}/scripts/hpc/smoke_test.py\" --data-only"

#!/usr/bin/env bash
# Source this file from Slurm jobs. It supports both Conda and venv installs.

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV_NAME is set but conda is not available. Load Conda first." >&2
    return 2
  fi
# Some site activation hooks reference optional variables without guarding
# them. Preserve the caller's nounset setting while Conda evaluates the hooks.
  nounset_enabled=0
  case "$-" in
    *u*) nounset_enabled=1; set +u ;;
  esac
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
  if (( nounset_enabled )); then
    set -u
  fi
else
  ALPHAGENOME_ENV="${ALPHAGENOME_ENV:-}"
  if [[ -z "${ALPHAGENOME_ENV}" || ! -x "${ALPHAGENOME_ENV}/bin/python" ]]; then
    echo "Set CONDA_ENV_NAME or provide a valid ALPHAGENOME_ENV venv." >&2
    return 2
  fi
  source "${ALPHAGENOME_ENV}/bin/activate"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "No Python interpreter is available after environment activation." >&2
  return 2
fi

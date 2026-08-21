#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_ROOT}/data/reference}"
HG38_FASTA="${HG38_FASTA:-${REFERENCE_DIR}/GRCh38.p13.genome.fa}"
HG38_URL="${HG38_URL:-https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa}"
ALPHAGENOME_ENV="${ALPHAGENOME_ENV:-${PROJECT_ROOT}/.venv}"

mkdir -p "${REFERENCE_DIR}"

echo "Downloading GRCh38.p13/hg38 to ${HG38_FASTA}"
curl \
  --fail \
  --location \
  --retry 5 \
  --continue-at - \
  --output "${HG38_FASTA}" \
  "${HG38_URL}"

if command -v samtools >/dev/null 2>&1; then
  samtools faidx "${HG38_FASTA}"
elif command -v python >/dev/null 2>&1 \
  && python -c 'import pyfaidx' >/dev/null 2>&1; then
  # When Conda is active, use its pyfaidx installation. This also supports
  # clusters that do not provide samtools on the login/data-transfer node.
  python -c \
    'import sys; from pyfaidx import Fasta; Fasta(sys.argv[1], rebuild=True).close()' \
    "${HG38_FASTA}"
elif [[ -x "${ALPHAGENOME_ENV}/bin/python" ]]; then
  "${ALPHAGENOME_ENV}/bin/python" -c \
    'import sys; from pyfaidx import Fasta; Fasta(sys.argv[1], rebuild=True).close()' \
    "${HG38_FASTA}"
else
  echo "Cannot index the FASTA: load samtools or create ALPHAGENOME_ENV." >&2
  exit 2
fi

echo "Reference ready: ${HG38_FASTA}"
echo "Index ready:     ${HG38_FASTA}.fai"

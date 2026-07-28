#!/usr/bin/env bash
#SBATCH --job-name=aging-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ALPHAGENOME_ENV="${ALPHAGENOME_ENV:-${PROJECT_ROOT}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"

: "${HG38_FASTA:?Set HG38_FASTA to the indexed GRCh38.p13 FASTA.}"

if [[ ! -x "${ALPHAGENOME_ENV}/bin/python" ]]; then
  echo "Environment not found: ${ALPHAGENOME_ENV}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
source "${ALPHAGENOME_ENV}/bin/activate"

python -u -m aging_alphagenome.data \
  --input-tsv "${PROJECT_ROOT}/RS_PDL50.wgs.rediportal.vcf.isec.tsv" \
  --reference-fasta "${HG38_FASTA}" \
  --output-tsv "${OUTPUT_DIR}/aging_loci.hg38.tsv" \
  --manifest "${OUTPUT_DIR}/aging_loci.hg38.manifest.json" \
  --biological-window 201 \
  --model-window 2048 \
  --negative-ratio 1 \
  --validation-fraction 0.20 \
  --seed 17

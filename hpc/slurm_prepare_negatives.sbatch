#!/usr/bin/env bash
#SBATCH --job-name=aging-negatives
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"
NEGATIVE_SAMPLES="${NEGATIVE_SAMPLES:-1000000}"

: "${HG38_FASTA:?Set HG38_FASTA to the indexed GRCh38.p13 FASTA.}"

mkdir -p "${OUTPUT_DIR}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

python -u -m aging_alphagenome.negative_data \
  --train-tsv "${PROJECT_ROOT}/RS_PDL50_train_80.tsv" \
  --validation-tsv "${PROJECT_ROOT}/RS_PDL50_test_20.tsv" \
  --reference-fasta "${HG38_FASTA}" \
  --output-tsv "${OUTPUT_DIR}/aging_negatives_16384bp.tsv.gz" \
  --mask-bed "${OUTPUT_DIR}/aging_mask_200bp.bed" \
  --manifest "${OUTPUT_DIR}/aging_negatives_16384bp.manifest.json" \
  --number "${NEGATIVE_SAMPLES}" \
  --mask-window 200 \
  --region-length 16384 \
  --max-n-fraction 0.05 \
  --seed 17

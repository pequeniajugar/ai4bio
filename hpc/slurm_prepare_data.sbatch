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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/processed}"

: "${HG38_FASTA:?Set HG38_FASTA to the indexed GRCh38.p13 FASTA.}"

mkdir -p "${OUTPUT_DIR}"
source "${PROJECT_ROOT}/scripts/hpc/activate_env.sh"

python -u -m aging_alphagenome.data \
  --train-tsv "${PROJECT_ROOT}/RS_PDL50_train_80.tsv" \
  --validation-tsv "${PROJECT_ROOT}/RS_PDL50_test_20.tsv" \
  --reference-fasta "${HG38_FASTA}" \
  --output-tsv "${OUTPUT_DIR}/aging_loci.hg38.tsv" \
  --manifest "${OUTPUT_DIR}/aging_loci.hg38.manifest.json" \
  --biological-window 201 \
  --model-window 2048 \
  --negative-ratio 1 \
  --seed 17

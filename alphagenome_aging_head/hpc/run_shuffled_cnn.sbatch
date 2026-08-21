#!/usr/bin/env bash
#SBATCH --job-name=cnn_shuffled
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/hm2991/alphagenome_SFT/alphagenome_aging_head}"
ENV_BIN="${ENV_BIN:-/scratch/hm2991/conda/envs/aging-alphagenome/bin}"
DATASET="${DATASET:-$PROJECT_ROOT/artifacts/gc_test/aging_loci.original.shuffled.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/gc_test/cnn_shuffled}"

cd "$PROJECT_ROOT"

if [[ ! -f "$PROJECT_ROOT/src/aging_alphagenome/cnn.py" ]]; then
  echo "Missing current CNN source: $PROJECT_ROOT/src/aging_alphagenome/cnn.py" >&2
  exit 1
fi
if [[ ! -f "$DATASET" ]]; then
  echo "Missing shuffled dataset: $DATASET" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${DEBUG_NUMERICS:-0}" == "1" ]]; then
  export JAX_DEBUG_NANS=True
  export JAX_DEBUG_INFS=True
fi

echo "=== GPU ==="
nvidia-smi

echo "=== Python and imported source ==="
"$ENV_BIN/python" --version
"$ENV_BIN/python" -c \
  'import aging_alphagenome.cnn as cnn; import jax; print("CNN source:", cnn.__file__); print("JAX:", jax.__version__); print("Devices:", jax.devices())'

"$ENV_BIN/python" -m aging_alphagenome.cnn \
  --dataset "$DATASET" \
  --output-model "$OUTPUT_DIR/model.npz" \
  --metrics "$OUTPUT_DIR/metrics.json" \
  --test-predictions "$OUTPUT_DIR/test_predictions.tsv" \
  --architecture large \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --dropout 0.40 \
  --weight-decay 0.0005 \
  --seed 17 \
  --require-gpu

echo "=== RESULTS ==="
cat "$OUTPUT_DIR/metrics.json"

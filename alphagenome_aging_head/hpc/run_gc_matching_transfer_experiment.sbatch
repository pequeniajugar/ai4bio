#!/usr/bin/env bash
#SBATCH --job-name=cnn_gc_transfer
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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/artifacts/gc_matching_transfer}"
MATCHED_DATASET="${MATCHED_DATASET:-$PROJECT_ROOT/artifacts/gc_test/aging_loci.original.tsv}"
TRAIN_TSV="${TRAIN_TSV:-$PROJECT_ROOT/RS_PDL50_train_80.tsv}"
VALIDATION_TSV="${VALIDATION_TSV:-$PROJECT_ROOT/RS_PDL50_test_20.tsv}"
: "${HG38_FASTA:?Set/export HG38_FASTA to the indexed GRCh38/hg38 FASTA before sbatch.}"

DATA_DIR="$EXPERIMENT_ROOT/datasets"
GC_UNMATCHED_DATASET="$DATA_DIR/aging_loci.gc_unmatched_reference_only.tsv"
AUDIT_MATCHED="$EXPERIMENT_ROOT/audit_gc_matched"
AUDIT_UNMATCHED="$EXPERIMENT_ROOT/audit_gc_unmatched"

CNN_SEED="${CNN_SEED:-17}"
NEGATIVE_SEED="${NEGATIVE_SEED:-17}"
CNN_EPOCHS="${CNN_EPOCHS:-100}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-64}"
CNN_LEARNING_RATE="${CNN_LEARNING_RATE:-1e-4}"
CNN_DROPOUT="${CNN_DROPOUT:-0.40}"
CNN_WEIGHT_DECAY="${CNN_WEIGHT_DECAY:-0.0005}"

# By default reuse the already-completed matched CNN from the dinucleotide
# experiment, which used the exact same training hyperparameters. Set
# RETRAIN_MATCHED=1 to force a fresh matched-model training in this job.
RETRAIN_MATCHED="${RETRAIN_MATCHED:-0}"
EXISTING_MATCHED_MODEL="${EXISTING_MATCHED_MODEL:-$PROJECT_ROOT/artifacts/dinucleotide_shuffle_experiment/cnn_original/model.npz}"

cd "$PROJECT_ROOT"
mkdir -p "$EXPERIMENT_ROOT" "$DATA_DIR" hpc/logs
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${DEBUG_NUMERICS:-0}" == "1" ]]; then
  export JAX_DEBUG_NANS=True
  export JAX_DEBUG_INFS=True
fi

for path in "$MATCHED_DATASET" "$TRAIN_TSV" "$VALIDATION_TSV" "$HG38_FASTA"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

train_one() {
  local name="$1"
  local dataset="$2"
  local output_dir="$EXPERIMENT_ROOT/cnn_${name}"
  mkdir -p "$output_dir"
  echo "=== TRAIN: $name ==="
  "$ENV_BIN/python" -m aging_alphagenome.cnn \
    --dataset "$dataset" \
    --output-model "$output_dir/model.npz" \
    --metrics "$output_dir/metrics.json" \
    --test-predictions "$output_dir/test_predictions.tsv" \
    --architecture large \
    --epochs "$CNN_EPOCHS" \
    --batch-size "$CNN_BATCH_SIZE" \
    --learning-rate "$CNN_LEARNING_RATE" \
    --dropout "$CNN_DROPOUT" \
    --weight-decay "$CNN_WEIGHT_DECAY" \
    --seed "$CNN_SEED" \
    --require-gpu
}

score_one() {
  local train_name="$1"
  local test_name="$2"
  local model="$3"
  local dataset="$4"
  local output_dir="$EXPERIMENT_ROOT/score_train_${train_name}_test_${test_name}"
  mkdir -p "$output_dir"
  echo "=== SCORE train=$train_name test=$test_name ==="
  "$ENV_BIN/python" -m aging_alphagenome.cnn_score \
    --dataset "$dataset" \
    --model "$model" \
    --metrics "$output_dir/metrics.json" \
    --test-predictions "$output_dir/test_predictions.tsv" \
    --batch-size "$CNN_BATCH_SIZE" \
    --require-gpu
}

echo "=== GPU / environment ==="
nvidia-smi
"$ENV_BIN/python" --version
"$ENV_BIN/python" -c 'import aging_alphagenome.data as d, aging_alphagenome.cnn as c, jax; print("data:", d.__file__); print("cnn:", c.__file__); print("JAX:", jax.__version__); print("devices:", jax.devices())'

echo "=== 1. Generate GC-unmatched controls ==="
# reference_only is deliberate: negatives are still random same-chromosome
# samples, but we retain center-base matching and remove ONLY the GC criterion.
"$ENV_BIN/python" -m aging_alphagenome.data \
  --train-tsv "$TRAIN_TSV" \
  --validation-tsv "$VALIDATION_TSV" \
  --reference-fasta "$HG38_FASTA" \
  --output-tsv "$GC_UNMATCHED_DATASET" \
  --biological-window 201 \
  --model-window 2048 \
  --negative-ratio 1 \
  --matching-strategy reference_only \
  --positive-exclusion-radius 2048 \
  --negative-min-distance 201 \
  --max-n-fraction 0.05 \
  --seed "$NEGATIVE_SEED"

echo "=== 2. Audit composition in BOTH datasets ==="
"$ENV_BIN/python" -m aging_alphagenome.composition_audit \
  --dataset "$MATCHED_DATASET" \
  --output-dir "$AUDIT_MATCHED" \
  --seed "$NEGATIVE_SEED"
"$ENV_BIN/python" -m aging_alphagenome.composition_audit \
  --dataset "$GC_UNMATCHED_DATASET" \
  --output-dir "$AUDIT_UNMATCHED" \
  --seed "$NEGATIVE_SEED"

echo "=== 3. Obtain matched CNN ==="
if [[ "$RETRAIN_MATCHED" == "1" || ! -f "$EXISTING_MATCHED_MODEL" ]]; then
  train_one "gc_matched" "$MATCHED_DATASET"
  MATCHED_MODEL="$EXPERIMENT_ROOT/cnn_gc_matched/model.npz"
else
  MATCHED_MODEL="$EXISTING_MATCHED_MODEL"
  echo "Reusing matched checkpoint: $MATCHED_MODEL"
fi

echo "=== 4. Train CNN on GC-unmatched negatives ==="
train_one "gc_unmatched" "$GC_UNMATCHED_DATASET"
GC_UNMATCHED_MODEL="$EXPERIMENT_ROOT/cnn_gc_unmatched/model.npz"

echo "=== 5. Full 2x2 transfer matrix ==="
score_one "gc_matched" "gc_matched" "$MATCHED_MODEL" "$MATCHED_DATASET"
score_one "gc_matched" "gc_unmatched" "$MATCHED_MODEL" "$GC_UNMATCHED_DATASET"
score_one "gc_unmatched" "gc_matched" "$GC_UNMATCHED_MODEL" "$MATCHED_DATASET"
score_one "gc_unmatched" "gc_unmatched" "$GC_UNMATCHED_MODEL" "$GC_UNMATCHED_DATASET"

echo "=== 6. Validate and summarize ==="
"$ENV_BIN/python" -m aging_alphagenome.negative_sampling_transfer_summary \
  --matched-dataset "$MATCHED_DATASET" \
  --unmatched-dataset "$GC_UNMATCHED_DATASET" \
  --matched-audit "$AUDIT_MATCHED/composition_audit.json" \
  --unmatched-audit "$AUDIT_UNMATCHED/composition_audit.json" \
  --matched-on-matched "$EXPERIMENT_ROOT/score_train_gc_matched_test_gc_matched/metrics.json" \
  --matched-on-unmatched "$EXPERIMENT_ROOT/score_train_gc_matched_test_gc_unmatched/metrics.json" \
  --unmatched-on-matched "$EXPERIMENT_ROOT/score_train_gc_unmatched_test_gc_matched/metrics.json" \
  --unmatched-on-unmatched "$EXPERIMENT_ROOT/score_train_gc_unmatched_test_gc_unmatched/metrics.json" \
  --output-dir "$EXPERIMENT_ROOT/summary"

echo "=== FINAL: composition comparison ==="
cat "$EXPERIMENT_ROOT/summary/composition_comparison.tsv"
echo "=== FINAL: 2x2 transfer matrix ==="
cat "$EXPERIMENT_ROOT/summary/transfer_matrix.tsv"
echo "=== FINAL: summary JSON ==="
cat "$EXPERIMENT_ROOT/summary/summary.json"

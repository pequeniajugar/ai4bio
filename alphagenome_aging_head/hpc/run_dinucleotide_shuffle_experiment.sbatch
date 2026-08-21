#!/usr/bin/env bash
#SBATCH --job-name=cnn_dinuc_shuffle
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
DATASET="${DATASET:-$PROJECT_ROOT/artifacts/gc_test/aging_loci.original.tsv}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/artifacts/dinucleotide_shuffle_experiment}"
SHUFFLE_SEEDS="${SHUFFLE_SEEDS:-17 23 41 59 73}"
MONO_SEED="${MONO_SEED:-17}"
CNN_SEED="${CNN_SEED:-17}"
CNN_EPOCHS="${CNN_EPOCHS:-100}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-64}"
CNN_LEARNING_RATE="${CNN_LEARNING_RATE:-1e-4}"
CNN_DROPOUT="${CNN_DROPOUT:-0.40}"
CNN_WEIGHT_DECAY="${CNN_WEIGHT_DECAY:-0.0005}"

cd "$PROJECT_ROOT"
mkdir -p "$EXPERIMENT_ROOT" hpc/logs
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${DEBUG_NUMERICS:-0}" == "1" ]]; then
  export JAX_DEBUG_NANS=True
  export JAX_DEBUG_INFS=True
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Missing source dataset: $DATASET" >&2
  exit 1
fi

read -r -a SHUFFLE_SEED_ARRAY <<< "$SHUFFLE_SEEDS"

echo "=== GPU ==="
nvidia-smi

echo "=== Environment ==="
"$ENV_BIN/python" --version
"$ENV_BIN/python" -c \
  'import aging_alphagenome.cnn as cnn; import aging_alphagenome.composition as c; import jax; print("CNN source:", cnn.__file__); print("Composition source:", c.__file__); print("JAX:", jax.__version__); print("Devices:", jax.devices())'

echo "=== 1. Generate exact shuffle controls ==="
"$ENV_BIN/python" -m aging_alphagenome.shuffle_controls \
  --dataset "$DATASET" \
  --output-dir "$EXPERIMENT_ROOT/datasets" \
  --mono-seed "$MONO_SEED" \
  --dinucleotide-seeds "${SHUFFLE_SEED_ARRAY[@]}"

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
  local name="$1"
  local dataset="$2"
  local output_dir="$EXPERIMENT_ROOT/fixed_model_${name}"
  mkdir -p "$output_dir"
  echo "=== FIXED ORIGINAL MODEL SCORE: $name ==="
  "$ENV_BIN/python" -m aging_alphagenome.cnn_score \
    --dataset "$dataset" \
    --model "$EXPERIMENT_ROOT/cnn_original/model.npz" \
    --metrics "$output_dir/metrics.json" \
    --test-predictions "$output_dir/test_predictions.tsv" \
    --batch-size "$CNN_BATCH_SIZE" \
    --require-gpu
}

# Keep CNN initialization/training seed fixed for every retraining run.  Only
# the sequence randomization seed changes.
echo "=== 2. Retrain on original control ==="
train_one "original" "$DATASET"

echo "=== 3. Retrain on mononucleotide control ==="
MONO_DATASET="$EXPERIMENT_ROOT/datasets/aging_loci.mono_shuffled.seed_${MONO_SEED}.tsv"
train_one "mono_seed_${MONO_SEED}" "$MONO_DATASET"

echo "=== 4. Retrain on five exact-dinucleotide controls ==="
for SEED in "${SHUFFLE_SEED_ARRAY[@]}"; do
  DINUC_DATASET="$EXPERIMENT_ROOT/datasets/aging_loci.dinuc_shuffled.seed_${SEED}.tsv"
  train_one "dinuc_seed_${SEED}" "$DINUC_DATASET"
done

echo "=== 5. Inference-only perturbation with the fixed original CNN ==="
score_one "original" "$DATASET"
score_one "mono_seed_${MONO_SEED}" "$MONO_DATASET"
for SEED in "${SHUFFLE_SEED_ARRAY[@]}"; do
  DINUC_DATASET="$EXPERIMENT_ROOT/datasets/aging_loci.dinuc_shuffled.seed_${SEED}.tsv"
  score_one "dinuc_seed_${SEED}" "$DINUC_DATASET"
done

echo "=== 6. Summarize ==="
"$ENV_BIN/python" -m aging_alphagenome.shuffle_experiment_summary \
  --root "$EXPERIMENT_ROOT" \
  --output-tsv "$EXPERIMENT_ROOT/summary.tsv"

echo "=== FINAL SUMMARY ==="
cat "$EXPERIMENT_ROOT/summary.tsv"
cat "$EXPERIMENT_ROOT/summary.json"

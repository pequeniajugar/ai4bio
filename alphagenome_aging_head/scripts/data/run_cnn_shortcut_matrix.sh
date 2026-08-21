#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/gc_shortcut_stress_test}"
CNN_ARCHITECTURE="${CNN_ARCHITECTURE:-small}"
CNN_EPOCHS="${CNN_EPOCHS:-100}"
CNN_BATCH_SIZE="${CNN_BATCH_SIZE:-64}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

GPU_FLAG=()
if [[ "$REQUIRE_GPU" == "1" ]]; then
  GPU_FLAG+=(--require-gpu)
fi

shopt -s nullglob
SEED_DIRS=("$OUTPUT_ROOT"/seed_*)
if [[ ${#SEED_DIRS[@]} -eq 0 ]]; then
  echo "No seed_* directories found under $OUTPUT_ROOT" >&2
  exit 1
fi

for SEED_DIR in "${SEED_DIRS[@]}"; do
  SEED_NAME="$(basename "$SEED_DIR")"
  SEED="${SEED_NAME#seed_}"
  for STRATEGY in random reference_only gc_tolerance gc_exact composition; do
    for VARIANT in original shuffled; do
      if [[ "$VARIANT" == "original" ]]; then
        DATASET="$SEED_DIR/aging_loci.${STRATEGY}.tsv"
      else
        DATASET="$SEED_DIR/aging_loci.${STRATEGY}.mono_shuffled.tsv"
      fi
      if [[ ! -f "$DATASET" ]]; then
        echo "Skipping missing dataset: $DATASET" >&2
        continue
      fi

      ARTIFACT_DIR="$SEED_DIR/cnn_${STRATEGY}_${VARIANT}"
      mkdir -p "$ARTIFACT_DIR"
      echo "=== seed=$SEED strategy=$STRATEGY variant=$VARIANT ==="
      aging-train-cnn \
        --dataset "$DATASET" \
        --output-model "$ARTIFACT_DIR/model.npz" \
        --metrics "$ARTIFACT_DIR/metrics.json" \
        --test-predictions "$ARTIFACT_DIR/test_predictions.tsv" \
        --architecture "$CNN_ARCHITECTURE" \
        --epochs "$CNN_EPOCHS" \
        --batch-size "$CNN_BATCH_SIZE" \
        --seed "$SEED" \
        "${GPU_FLAG[@]}"
    done
  done
done

#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${HG38_FASTA:?Set HG38_FASTA to GRCh38.p13.genome.fa}"

TRAIN_TSV="${TRAIN_TSV:-$PROJECT_ROOT/RS_PDL50_train_80.tsv}"
VALIDATION_TSV="${VALIDATION_TSV:-$PROJECT_ROOT/RS_PDL50_test_20.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/gc_shortcut_stress_test}"
SEEDS="${SEEDS:-17}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-50000}"

mkdir -p "$OUTPUT_ROOT"

for SEED in $SEEDS; do
  SEED_ROOT="$OUTPUT_ROOT/seed_${SEED}"
  mkdir -p "$SEED_ROOT"
  for STRATEGY in random reference_only gc_tolerance gc_exact composition; do
    DATASET="$SEED_ROOT/aging_loci.${STRATEGY}.tsv"
    AUDIT_DIR="$SEED_ROOT/audit_${STRATEGY}"
    SHUFFLED="$SEED_ROOT/aging_loci.${STRATEGY}.mono_shuffled.tsv"

    echo "=== seed=$SEED: preparing $STRATEGY negatives ==="
    aging-prepare-data \
      --train-tsv "$TRAIN_TSV" \
      --validation-tsv "$VALIDATION_TSV" \
      --reference-fasta "$HG38_FASTA" \
      --output-tsv "$DATASET" \
      --biological-window 201 \
      --model-window 2048 \
      --negative-ratio 1 \
      --matching-strategy "$STRATEGY" \
      --gc-tolerance 0.05 \
      --gc-count-tolerance 0 \
      --base-fraction-tolerance 0.02 \
      --dinucleotide-l1-tolerance 0.15 \
      --max-attempts "$MAX_ATTEMPTS" \
      --seed "$SEED"

    echo "=== seed=$SEED: auditing $STRATEGY dataset ==="
    aging-audit-composition \
      --dataset "$DATASET" \
      --output-dir "$AUDIT_DIR" \
      --write-mononucleotide-shuffled "$SHUFFLED" \
      --seed "$SEED"
  done
done

cat <<EOF

Prepared all shortcut-control datasets under:
  $OUTPUT_ROOT

For repeated negative sampling, for example:
  SEEDS="17 23 41 59 73" bash "$PROJECT_ROOT/scripts/data/run_gc_shortcut_stress_test.sh"

Next, train the existing CNN on each original dataset and its mono-shuffled
copy using aging-train-cnn. A high score on shuffled data or on the
composition-only baselines is evidence that low-order composition is enough
to solve the benchmark.
EOF

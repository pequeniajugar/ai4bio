# AlphaGenome aging-locus classifier

This repository now contains the complete first-pass pipeline requested by
Professor Wang:

1. validate the supplied positive loci against hg38;
2. extract a 201 bp sequence (`±100 bp`) around every locus;
3. construct matched random genomic controls;
4. make a leakage-resistant 80/20 chromosome split;
5. extract frozen AlphaGenome trunk embeddings on an HPC GPU;
6. train a binary linear classification head; and
7. score new loci with the trained head.

No additional “HPC specifications” are required to use the code. Cluster
account names, Slurm partitions, and module names are site-specific values
that can be supplied when jobs are submitted.

Here, “HPC specs” only means those cluster-specific values. The scientific
specification is now fixed by the supervisor's reply: hg38, a 201 bp biological
window, randomly sampled controls, and an 80/20 evaluation split. If your
cluster uses Slurm, the only values you normally need to fill in are
`SLURM_ACCOUNT` and the CPU/GPU partition names; a template is provided at
[`hpc/site.env.example`](hpc/site.env.example). You do not need to know these
values to continue developing or testing the Python code locally.

## Modeling definition

The first-pass target is:

> distinguish known aging-associated loci from matched, randomly sampled hg38
> loci and use the learned score to prioritize unseen candidate sites.

This is technically a **positive-unlabeled** problem. Label `0` means a
randomly sampled, matched hg38 locus; it does not prove that the locus has no
aging function. Metrics must be interpreted with that limitation.

### Sequence lengths

- The biological region is exactly 201 bp centered on the locus.
- The AlphaGenome trunk receives 2,048 bp centered on the same locus.
- The head uses the 1 bp embedding at the center plus a weighted pool of the
  128 bp embeddings overlapping the central 201 bp.

The larger model input is intentional. The released AlphaGenome architecture
is designed for power-of-two inputs and its pairwise trunk representation
starts at 2,048 bp. Only the requested central 201 bp is pooled as the primary
classification region.

### Random controls

For every positive, the default pipeline samples one control that:

- is on the same chromosome;
- has the same hg38 reference base, preserving the `A>G` / `T>C` composition;
- is within 0.05 local GC fraction of the positive;
- has no more than 5% ambiguous bases;
- is more than 2,048 bp from a known positive; and
- is at least 201 bp from another sampled control.

Controls remain paired with their source positive in `pair_id`.

### Split

The split holds out whole chromosomes instead of randomly splitting rows.
This prevents overlapping or nearby sequence contexts from appearing in both
sets. With the current 2,132 positives and seed 17, the automatic holdout is
`chr1,chr8,chr18,chr22`: 426 positives, or 19.98%.

Each control is sampled on the positive's chromosome, so both classes remain
balanced in the train and validation partitions.

## Repository map

- `src/aging_alphagenome/data.py`: hg38 validation, sequence extraction,
  matched sampling, and grouped splitting.
- `src/aging_alphagenome/features.py`: frozen AlphaGenome embedding
  extraction.
- `src/aging_alphagenome/head.py`: linear-head training, evaluation, and
  scoring.
- `scripts/data/download_hg38.sh`: resumable GRCh38.p13 download and indexing.
- `scripts/hpc/`: environment, checkpoint, and smoke-test utilities.
- `hpc/`: Slurm jobs for each pipeline stage.
- `tests/`: deterministic tests for sampling, splitting, metrics, and training.

## HPC requirements

- Linux x86_64 and Python 3.11+.
- Slurm, or equivalent commands adapted to another scheduler.
- One NVIDIA H100 80 GB for checkpoint validation and embedding extraction.
- NVIDIA driver 525+ for the default CUDA 12 JAX wheels.
- At least 128 GB host RAM for AlphaGenome GPU jobs.
- Shared storage visible from login and compute nodes.
- Internet access on a login/data-transfer node for installation and downloads.

The official research implementation is pinned to commit
`c5b51606a410b34c3e64d870ea4e034c3c0ca976`, with AlphaGenome client `0.7.0`.

## 1. Install the environment

Load the equivalent Python, Git, and compiler modules provided by the cluster:

```bash
module purge
module load python/3.11 git gcc

export PROJECT_ROOT=/shared/path/to/alphagenome-SFT
export ALPHAGENOME_ENV=/shared/path/to/envs/alphagenome
export PIP_CACHE_DIR=/shared/path/to/cache/pip
export JAX_CUDA_VARIANT=cuda12

bash "$PROJECT_ROOT/scripts/hpc/bootstrap_env.sh"
source "$ALPHAGENOME_ENV/bin/activate"
```

The bootstrap installs this repository in editable mode and records the
resolved environment in
`$ALPHAGENOME_ENV/requirements.freeze.txt`. It is safe to rerun.

Use `JAX_CUDA_VARIANT=cuda13` only with driver 580+. If cluster policy requires
site-installed CUDA/cuDNN libraries, use `cuda12-local` or `cuda13-local`.

Run local tests:

```bash
cd "$PROJECT_ROOT"
python -m unittest discover -s tests -v
python scripts/hpc/smoke_test.py --data-only
```

## 2. Download and index hg38

The downloader uses the same GRCh38.p13 FASTA referenced by AlphaGenome:

```bash
export REFERENCE_DIR=/shared/path/to/references/hg38
export HG38_FASTA="$REFERENCE_DIR/GRCh38.p13.genome.fa"

bash "$PROJECT_ROOT/scripts/data/download_hg38.sh"
```

The data-preparation stage checks every TSV `REF` allele against this FASTA and
stops if they do not match. This prevents silently using the wrong assembly.

## 3. Download the fold-0 AlphaGenome checkpoint

Accept the non-commercial model terms at
[`google/alphagenome-fold-0`](https://huggingface.co/google/alphagenome-fold-0),
then use a read-only Hugging Face token:

```bash
source "$ALPHAGENOME_ENV/bin/activate"
export HF_TOKEN='your-read-only-token'
export ALPHAGENOME_CHECKPOINT_DIR=/shared/path/to/weights/alphagenome-fold-0

python "$PROJECT_ROOT/scripts/hpc/download_weights.py" \
  --model-version fold-0 \
  --output-dir "$ALPHAGENOME_CHECKPOINT_DIR"

unset HF_TOKEN
```

Do not store the token in this repository or a Slurm job. Fold 0 is used for
initial evaluation; `all-folds` should be reserved for final inference after
the methodology is fixed.

## 4. Verify AlphaGenome on a GPU

```bash
export PROJECT_ROOT ALPHAGENOME_ENV ALPHAGENOME_CHECKPOINT_DIR
export ALPHAGENOME_CACHE=/shared/path/to/cache/alphagenome

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_H100_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_smoke.sbatch"
```

A successful log ends with `ALPHAGENOME HPC SMOKE TEST PASSED`.

## 5. Prepare training data

Submit the CPU preparation job:

```bash
export PROJECT_ROOT ALPHAGENOME_ENV HG38_FASTA
export OUTPUT_DIR=/shared/path/to/aging-project/data

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_prepare_data.sbatch"
```

This creates:

- `aging_loci.hg38.tsv`: 2,132 positives and 2,132 matched controls, including
  the 201 bp and 2,048 bp sequences.
- `aging_loci.hg38.manifest.json`: input hash, reference metadata, sampling
  settings, holdout chromosomes, counts, and label semantics.

The equivalent direct command is:

```bash
aging-prepare-data \
  --input-tsv "$PROJECT_ROOT/RS_PDL50.wgs.rediportal.vcf.isec.tsv" \
  --reference-fasta "$HG38_FASTA" \
  --output-tsv "$OUTPUT_DIR/aging_loci.hg38.tsv" \
  --biological-window 201 \
  --model-window 2048 \
  --negative-ratio 1 \
  --validation-fraction 0.20 \
  --seed 17
```

## 6. Extract frozen AlphaGenome features

```bash
export PROJECT_ROOT ALPHAGENOME_ENV ALPHAGENOME_CHECKPOINT_DIR
export PREPARED_DATASET="$OUTPUT_DIR/aging_loci.hg38.tsv"
export OUTPUT_DIR=/shared/path/to/aging-project/artifacts

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_H100_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_extract_features.sbatch"
```

The output `aging_loci.alphagenome_features.npz` contains a compact feature
vector for every example, labels, split assignments, coordinates, and IDs.
The AlphaGenome trunk remains frozen; only the small downstream head is
trained.

For a short pipeline check, add `--limit 8` to the feature command in a copy of
the job file. Do not train or report metrics from a limited archive.

## 7. Train and evaluate the head

This stage is small and runs on a CPU node:

```bash
export PROJECT_ROOT ALPHAGENOME_ENV
export OUTPUT_DIR=/shared/path/to/aging-project/artifacts
export FEATURE_ARCHIVE="$OUTPUT_DIR/aging_loci.alphagenome_features.npz"

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_train_head.sbatch"
```

Outputs:

- `aging_linear_head.npz`: weights, bias, training-set normalization, and
  decision threshold.
- `aging_linear_head.metrics.json`: train and held-out-chromosome loss,
  accuracy, balanced accuracy, AUROC, and average precision.

The linear head is the interpretable baseline. A nonlinear MLP should only be
considered after this baseline, negative sampling, and validation design are
reviewed.

## 8. Score feature archives

After extracting AlphaGenome features for candidate loci with the same window
and feature settings:

```bash
aging-score-head \
  --features candidate_features.npz \
  --model "$OUTPUT_DIR/aging_linear_head.npz" \
  --output-tsv candidate_aging_scores.tsv
```

The `aging_score` is a prioritization score under the sampled-control training
distribution, not a calibrated probability that a locus biologically causes
aging.

## Scheduler customization

The repository cannot know cluster-specific names. Supply account and
partition at `sbatch` time, or add them to local copies of the job files.
If the site does not support `#SBATCH --gres=gpu:1`, replace it with the local
equivalent, often `#SBATCH --gpus=1` or `#SBATCH --gres=gpu:h100:1`.

## Troubleshooting

- **REF mismatch during preparation:** verify that the FASTA is GRCh38/hg38
  with `chr`-prefixed contigs. Do not bypass this validation.
- **A control cannot be sampled:** increase `--max-attempts` first; only then
  consider widening `--gc-tolerance`.
- **JAX lists only CPU:** confirm `nvidia-smi` works inside the allocation and
  that the environment contains the appropriate JAX CUDA plugin.
- **Wrong CUDA libraries:** pip CUDA wheels generally work best without a
  cluster CUDA module in `LD_LIBRARY_PATH`; `*-local` installs require the
  exact site CUDA/cuDNN modules instead.
- **Out of GPU memory:** request an H100 80 GB and keep batch size at 1.
- **Hugging Face 401/403:** accept the gated model terms with the same account
  that owns `HF_TOKEN`.

## Scientific decisions still worth revisiting

- Whether the prediction unit should ultimately be an editing locus or an
  aggregated gene.
- Whether random controls should additionally match genic region, repeat
  class, mappability, and RNA-editing database membership.
- Whether the 201 bp pooled region should be compared against 301 bp and
  401 bp regions while keeping the AlphaGenome input fixed.
- Whether validation should follow AlphaGenome's official fold intervals in
  addition to the whole-chromosome holdout.

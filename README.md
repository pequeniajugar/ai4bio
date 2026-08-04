# AlphaGenome aging-locus classifier

The currently available source dataset contains positive aging-associated loci
only. The repository supports two negative-control constructions:

- a large coordinate-only corpus of random 16,384 bp hg38 regions for
  AlphaGenome training; and
- a small one-to-one matched-control dataset for the existing 201 bp CNN
  baseline.

For the first binary baseline, the small matched-control pipeline:

1. reads only the first two TSV columns (chromosome and 1-based position);
2. ignores REF, ALT, gene, repeat, and every other annotation column;
3. extracts a 201 bp sequence (`±100 bp`) around every hg38 locus;
4. samples one matched random hg38 context as an assumed negative per positive;
5. preserves the supplied 80/20 chromosome-held-out split;
6. first trains a compact CNN directly on the 201 bp sequences; and
7. optionally compares it with a linear head on frozen AlphaGenome features.

No additional “HPC specifications” are required to use the code. Cluster
account names, Slurm partitions, and module names are site-specific values
that can be supplied when jobs are submitted.

Here, “HPC specs” only means those cluster-specific values. The current
scientific inputs are hg38, a centered `±100 bp` biological window, two
coordinate-only positive TSVs, and their existing 80/20 split.

## Modeling definition

Every supplied row has label `1`. For the baseline, each sampled random context
receives label `0`. These are assumed negatives, not experimentally confirmed
non-aging loci, so the resulting score is a ranking under this sampled-control
definition rather than a calibrated probability of aging function.

### Sequence lengths

- Each row in the large negative corpus marks exactly 16,384 bp.
- A large negative region is eligible only when its full 16,384 bp span does
  not touch any masked 200 bp aging context.
- The biological region is exactly 201 bp centered on the locus.
- The CNN receives only this 201 bp sequence.
- The AlphaGenome trunk receives 2,048 bp centered on the same locus.
- The head uses the 1 bp embedding at the center plus a weighted pool of the
  128 bp embeddings overlapping the central 201 bp.

The larger model input is a technical requirement of the released local
AlphaGenome trunk, whose smallest supported input in this implementation is
2,048 bp. The locus is centered in that tensor. The head receives the center
1 bp embedding and a weighted pool of 128 bp embeddings overlapping only the
central 201 bp, although those embeddings can incorporate information from the
wider 2,048 bp context.

### Random controls

For each positive, the pipeline samples one random hg38 context on the same
chromosome, with the same center reference base and similar central-window GC
content. It excludes known positive neighborhoods and keeps the control in the
same train or validation split as its paired positive. No source annotation
columns are used for this matching.

### Split

`RS_PDL50_train_80.tsv` contains 1,706 positive loci on 17 chromosomes.
`RS_PDL50_test_20.tsv` contains 426 positive loci on 4 held-out chromosomes.
There is no coordinate overlap between the files.

## Repository map

- `src/aging_alphagenome/data.py`: coordinate-only hg38 sequence extraction,
  random-control sampling, and preservation of the supplied split.
- `src/aging_alphagenome/negative_data.py`: streaming construction of a large
  masked, random 16,384 bp negative-region corpus.
- `src/aging_alphagenome/cnn.py`: compact 201 bp CNN training, internal
  early-stopping split, final held-out evaluation, and prediction export.
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

### Conda alternative

For a Conda-managed HPC environment, load your site's Miniconda/Anaconda
module, set a name, and run:

```bash
module load miniconda  # use your site's module name
export PROJECT_ROOT=/shared/path/to/alphagenome-SFT
export CONDA_ENV_NAME=aging-alphagenome
export SCRATCH_ROOT=/scratch/YOUR_NETID
export JAX_CUDA_VARIANT=cuda12

bash "$PROJECT_ROOT/scripts/hpc/create_conda_env.sh"
conda activate "$CONDA_ENV_NAME"
```

When home-directory quota is limited, set `SCRATCH_ROOT` to your cluster
scratch directory. The setup then places Conda package archives, the Conda
environment, and pip's cache under `${SCRATCH_ROOT}`.

The Conda job scripts detect `CONDA_ENV_NAME` automatically. Do not set both
`CONDA_ENV_NAME` and `ALPHAGENOME_ENV`; use one environment style per shell.

Both setup scripts install this repository in editable mode and record the
resolved environment in `requirements.freeze.txt` inside the active
environment. It is safe to rerun.

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

The data-preparation stage derives every sequence directly from this FASTA.
Only the first two source columns are read; TSV `REF`, `ALT`, and annotations
are deliberately ignored.

## 3. Build the large 16,384 bp negative corpus

The large-corpus generator first centers an exact 200 bp mask on every known
aging locus in both supplied split files. It then samples unique starts
uniformly from the hg38 intervals for which the **entire** 16,384 bp region
does not overlap a mask. Regions with more than 5% non-ACGT sequence are
rejected.

By default, the job writes one million coordinate-only rows. Keeping sequence
out of this file is intentional: one million uncompressed 16,384 bp strings
alone would occupy at least 16.4 GB. Sequence can be fetched from the recorded
hg38 coordinates during training, or included explicitly for a smaller run
with `--include-sequence`.

```bash
export PROJECT_ROOT HG38_FASTA
export OUTPUT_DIR=/shared/path/to/aging-project/data/processed
export NEGATIVE_SAMPLES=1000000

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_prepare_negatives.sbatch"
```

The equivalent direct command is:

```bash
aging-prepare-negatives \
  --train-tsv "$PROJECT_ROOT/RS_PDL50_train_80.tsv" \
  --validation-tsv "$PROJECT_ROOT/RS_PDL50_test_20.tsv" \
  --reference-fasta "$HG38_FASTA" \
  --output-tsv "$OUTPUT_DIR/aging_negatives_16384bp.tsv.gz" \
  --mask-bed "$OUTPUT_DIR/aging_mask_200bp.bed" \
  --manifest "$OUTPUT_DIR/aging_negatives_16384bp.manifest.json" \
  --number 1000000 \
  --mask-window 200 \
  --region-length 16384 \
  --seed 17
```

This creates:

- `aging_negatives_16384bp.tsv.gz`: compact AlphaGenome interval rows with
  label 0, unique starts, and chromosome-held-out split assignments;
- `aging_mask_200bp.bed`: the merged 200 bp aging-context mask; and
- `aging_negatives_16384bp.manifest.json`: hashes, parameters, sampling-space
  size, rejection counts, and per-chromosome output counts.

Negative intervals may overlap one another; this keeps the available corpus
large. They cannot cross the train/validation boundary because the split is by
chromosome. These rows are still **assumed negatives**: masking all currently
known aging loci cannot rule out undiscovered aging-associated sequence.

## 4. Download the fold-0 AlphaGenome checkpoint

Accept the non-commercial model terms at
[`google/alphagenome-fold-0`](https://huggingface.co/google/alphagenome-fold-0),
then activate whichever environment you created and enter a read-only Hugging
Face token without putting it in shell history:

```bash
# Use the activation command for the environment you created:
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  conda activate "$CONDA_ENV_NAME"              # Conda setup
else
  source "$ALPHAGENOME_ENV/bin/activate"        # venv setup
fi

read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
export ALPHAGENOME_CHECKPOINT_DIR=/shared/path/to/weights/alphagenome-fold-0

python "$PROJECT_ROOT/scripts/hpc/download_weights.py" \
  --model-version fold-0 \
  --output-dir "$ALPHAGENOME_CHECKPOINT_DIR"

unset HF_TOKEN
```

The downloader authenticates with `HF_TOKEN` for this process only; a separate
`huggingface-cli login` is not required. Do not store the token in this
repository, `site.env`, or a Slurm job. Fold 0 is used for initial evaluation;
`all-folds` should be reserved for final inference after the methodology is
fixed.

## 5. Verify AlphaGenome on a GPU

```bash
export PROJECT_ROOT ALPHAGENOME_CHECKPOINT_DIR
export ALPHAGENOME_CACHE=/shared/path/to/cache/alphagenome

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_H100_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_smoke.sbatch"
```

A successful log ends with `ALPHAGENOME HPC SMOKE TEST PASSED`.

## 6. Prepare the small matched-control training data

Submit the CPU preparation job:

```bash
export PROJECT_ROOT HG38_FASTA
export OUTPUT_DIR=/shared/path/to/aging-project/data

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_prepare_data.sbatch"
```

This creates:

- `aging_loci.hg38.tsv`: 2,132 positive loci and 2,132 paired random controls
  with their 201 bp and 2,048 bp hg38 sequences and train/validation
  assignments.
- `aging_loci.hg38.manifest.json`: input hashes, reference metadata, counts,
  split information, and the coordinate-only column policy.

The equivalent direct command is:

```bash
aging-prepare-data \
  --train-tsv "$PROJECT_ROOT/RS_PDL50_train_80.tsv" \
  --validation-tsv "$PROJECT_ROOT/RS_PDL50_test_20.tsv" \
  --reference-fasta "$HG38_FASTA" \
  --output-tsv "$OUTPUT_DIR/aging_loci.hg38.tsv" \
  --biological-window 201 \
  --model-window 2048 \
  --negative-ratio 1 \
  --seed 17
```

## 7. Train and test the 201 bp CNN baseline

This run does not use AlphaGenome or its checkpoint. It reads only
`biological_sequence`, the 201 bp hg38 context created in step 6. Ten percent
of the supplied 80% training pairs are reserved internally for early stopping.
Positive/control pairs stay together. The supplied chromosome-held-out 20%
split is evaluated only after model selection.

```bash
export PROJECT_ROOT=/scratch/hm2991/alphagenome_SFT
export PREPARED_DATASET=/scratch/hm2991/aging-project/data/processed/aging_loci.hg38.tsv
export CNN_OUTPUT_DIR=/scratch/hm2991/aging-project/artifacts/cnn

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_train_cnn.sbatch"
```

The roughly 142,000-parameter model has three convolution layers, global
mean/max pooling, a 128-unit hidden layer, dropout, reverse-complement
augmentation, and early stopping. It writes:

- `aging_201bp_cnn.npz`: best model parameters;
- `aging_201bp_cnn.metrics.json`: train, internal-validation, and final-test
  AUROC, average precision, loss, and accuracy; and
- `aging_201bp_cnn.test_predictions.tsv`: one score for every held-out
  positive or random-control locus.

To inspect the final report:

```bash
python -m json.tool \
  "$CNN_OUTPUT_DIR/aging_201bp_cnn.metrics.json"
```

These test metrics measure whether 201 bp local sequence distinguishes the
known loci from this particular random-control construction. They do not prove
that the model recognizes aging biology, because the controls are assumed
negatives and may contain unknown aging-related loci.

### Larger-capacity CNN comparison

After establishing the small baseline, the large preset can test whether more
capacity and a wider receptive field improve internal-validation performance:

| Preset | Convolution channels | Parameters | Final receptive field |
| --- | --- | ---: | ---: |
| `small` | 64, 96, 128 | 141,601 | 46 bp |
| `large` | 128, 256, 256, 384, 384, 512 | 3,762,305 | 190 bp |

The last three large-model convolutions use dilation factors 2, 2, and 4. Its
job also increases dropout to 0.40 and weight decay to 0.0005 because the
training dataset is small relative to the model.

```bash
export PROJECT_ROOT=/scratch/hm2991/alphagenome_SFT
export PREPARED_DATASET=/scratch/hm2991/aging-project/data/processed/aging_loci.hg38.tsv
export LARGE_CNN_OUTPUT_DIR=/scratch/hm2991/aging-project/artifacts/cnn-large

sbatch "$PROJECT_ROOT/hpc/slurm_train_large_cnn.sbatch"
```

This writes `aging_201bp_large_cnn.npz`,
`aging_201bp_large_cnn.metrics.json`, and
`aging_201bp_large_cnn.test_predictions.tsv` without overwriting the small
model. Both presets use the same seed and pair-grouped internal split, making
the comparison controlled. Model selection still uses internal-validation
loss; the held-out test split is evaluated only after the best epoch is fixed.

## 8. Extract frozen AlphaGenome features (optional comparison)

```bash
export PROJECT_ROOT ALPHAGENOME_CHECKPOINT_DIR
export PREPARED_DATASET="$OUTPUT_DIR/aging_loci.hg38.tsv"
export OUTPUT_DIR=/shared/path/to/aging-project/artifacts

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_H100_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_extract_features.sbatch"
```

The output `aging_loci.alphagenome_features.npz` contains a compact feature
vector for every example, labels, split assignments, coordinates, and IDs.
The AlphaGenome trunk remains frozen.

For a short pipeline check, add `--limit 8` to the feature command in a copy of
the job file. Do not train or report metrics from a limited archive.

## 9. Train and evaluate the AlphaGenome linear head

```bash
export PROJECT_ROOT
export OUTPUT_DIR=/shared/path/to/aging-project/artifacts
export FEATURE_ARCHIVE="$OUTPUT_DIR/aging_loci.alphagenome_features.npz"

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  "$PROJECT_ROOT/hpc/slurm_train_head.sbatch"
```

This writes `aging_linear_head.npz` and
`aging_linear_head.metrics.json`. Metrics measure separation from the sampled
random controls, not separation from experimentally verified non-aging loci.

## Scheduler customization

The repository cannot know cluster-specific names. Supply account and
partition at `sbatch` time, or add them to local copies of the job files.
If the site does not support `#SBATCH --gres=gpu:1`, replace it with the local
equivalent, often `#SBATCH --gpus=1` or `#SBATCH --gres=gpu:h100:1`.

## Troubleshooting

- **Coordinate missing or out of bounds:** verify that the FASTA is GRCh38/hg38
  with `chr`-prefixed contigs and that TSV positions are 1-based.
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
- How robust the classifier is across repeated random-control samples, and
  whether a scientifically stronger candidate-locus background is available.
- Whether the 201 bp pooled region should be compared against 301 bp and
  401 bp regions while keeping the AlphaGenome input fixed.
- Whether validation should follow AlphaGenome's official fold intervals in
  addition to the whole-chromosome holdout.

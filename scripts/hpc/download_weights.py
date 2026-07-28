#!/usr/bin/env python3
"""Download a gated AlphaGenome checkpoint to shared HPC storage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPOSITORIES = {
    "fold-0": "google/alphagenome-fold-0",
    "fold-1": "google/alphagenome-fold-1",
    "fold-2": "google/alphagenome-fold-2",
    "fold-3": "google/alphagenome-fold-3",
    "all-folds": "google/alphagenome-all-folds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official AlphaGenome Orbax weights. Accept the model "
            "terms in a browser before running this command."
        )
    )
    parser.add_argument(
        "--model-version",
        choices=sorted(REPOSITORIES),
        default="fold-0",
        help="Fold 0 is the recommended starting checkpoint for fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Persistent shared directory visible from compute nodes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print(
            "huggingface_hub is missing. Activate the AlphaGenome environment.",
            file=sys.stderr,
        )
        return 2

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN is not set. Export a read-only Hugging Face token after "
            "accepting the AlphaGenome model terms.",
            file=sys.stderr,
        )
        return 2

    repository = REPOSITORIES[args.model_version]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        account = HfApi().whoami(token=token)
        print(f"Authenticated to Hugging Face as {account.get('name', 'unknown')}.")
        print(f"Downloading {repository} to {output_dir}")
        resolved_path = snapshot_download(
            repo_id=repository,
            local_dir=output_dir,
            token=token,
        )
    except HfHubHTTPError as exc:
        print(
            f"Download failed for {repository}: {exc}\n"
            "Confirm that this Hugging Face account accepted the model terms "
            "and that HF_TOKEN has read access.",
            file=sys.stderr,
        )
        return 1

    metadata = output_dir / "_CHECKPOINT_METADATA"
    manifest = output_dir / "manifest.ocdbt"
    if not metadata.is_file() or not manifest.is_file():
        print(
            "The download finished but required Orbax checkpoint metadata is "
            f"missing under {output_dir}.",
            file=sys.stderr,
        )
        return 1

    print(f"Checkpoint ready: {Path(resolved_path).resolve()}")
    print("You may now unset HF_TOKEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

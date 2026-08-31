#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-download model weights into the workspace (e.g. onto Google Drive).

    python -m structure_finder.setup_models --drive --engine all

Fetching weights ahead of time matters on Colab: the workspace is where
``HF_HOME`` / ``PYSTOW_HOME`` / ``TORCH_HOME`` are pointed, so if the workspace
sits on Drive the ~2 GB of weights survive runtime restarts instead of being
re-downloaded every session.

What each engine pulls:

``molclassifier``  the PatCID classification checkpoint from
                   ``huggingface.co/ds4sd/MolClassifier`` (~180 MB)
``molgrapher``     the keypoint detector and node classifier from
                   ``huggingface.co/ds4sd/MolGrapher``, plus PaddleOCR's
                   detection/recognition models (~500 MB)
``decimer``        the DECIMER V2 SavedModel from Zenodo (~1 GB)
``decimer-seg``    the DECIMER-Segmentation Mask R-CNN weights (~250 MB)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from .storage import ensure_molclassifier_checkpoint, resolve_workspace

ENGINES = ("molclassifier", "molgrapher", "decimer", "decimer-seg")


def fetch_molclassifier(workspace) -> str:
    path = ensure_molclassifier_checkpoint(workspace)
    return f"MolClassifier checkpoint -> {path}"


def fetch_molgrapher(workspace) -> str:
    from molgrapher.models.molgrapher_model import MolgrapherModel

    # Constructing the model downloads the weights into HF_HOME.
    MolgrapherModel({"visualize": False, "clean": False, "force_cpu": True})
    return f"MolGrapher weights -> {workspace.models / 'hf'}"


def fetch_decimer(workspace) -> str:
    import DECIMER  # noqa: F401  (import triggers the download)

    return f"DECIMER V2 weights -> {workspace.models / 'pystow'}"


def fetch_decimer_segmentation(workspace) -> str:
    from decimer_segmentation import get_model

    get_model()
    return "DECIMER-Segmentation Mask R-CNN weights downloaded"


FETCHERS = {
    "molclassifier": fetch_molclassifier,
    "molgrapher": fetch_molgrapher,
    "decimer": fetch_decimer,
    "decimer-seg": fetch_decimer_segmentation,
}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structure_finder.setup_models",
        description="Download model weights into the structure_finder workspace.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=list(ENGINES) + ["all"],
        default=[],
        help="Which weights to fetch (repeatable). Default: all.",
    )
    parser.add_argument("--workspace", help="Workspace directory.")
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Mount Google Drive and use MyDrive/structure_finder.",
    )
    args = parser.parse_args(argv)

    workspace = resolve_workspace(workspace=args.workspace, use_drive=args.drive)
    print(f"Workspace: {workspace.root}")

    engines = args.engine or ["all"]
    if "all" in engines:
        engines = list(ENGINES)

    failures = 0
    for engine in engines:
        print(f"\n== {engine} ==")
        try:
            print(FETCHERS[engine](workspace))
        except Exception as error:
            failures += 1
            print(f"FAILED: {error}", file=sys.stderr)
            print(
                "  (Install the corresponding package first; see "
                "requirements-structure-finder.txt)",
                file=sys.stderr,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

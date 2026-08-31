#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace and model-cache management (local directory or Google Drive).

Every heavy artefact used by this tool - MolGrapher / MolClassifier /
DECIMER weights, and the per-document extraction cache - is written under a
single *workspace* directory:

    <workspace>/
        models/          # MolClassifier checkpoint, DECIMER-Segmentation weights
        models/hf/       # HuggingFace hub cache  (MolGrapher weights land here)
        models/pystow/   # pystow home           (DECIMER V2 weights land here)
        models/torch/    # torch hub cache
        cache/           # per-document extraction results (.json)
        outputs/         # reports and annotated pages

Pointing the workspace at a mounted Google Drive folder therefore persists both
the models and the (expensive) document extractions across Colab sessions.

IMPORTANT: :func:`configure_model_home` sets ``HF_HOME`` / ``PYSTOW_HOME`` /
``TORCH_HOME``.  Those are read by the respective libraries *at import time*, so
it must run before any engine is imported.  All engines in this package import
their dependencies lazily for exactly this reason.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_WORKSPACE_NAME = "structure_finder"

# Model weights published alongside the PatCID paper.
MOLCLASSIFIER_CHECKPOINT_URL = (
    "https://huggingface.co/ds4sd/MolClassifier/resolve/main/models/molclassifier_model.chpt"
)
MOLCLASSIFIER_CHECKPOINT_NAME = "molclassifier_model.chpt"


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_google_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive.

    Returns the path of ``MyDrive`` on success, ``None`` when Drive is not
    available (e.g. running outside Colab).
    """
    my_drive = Path(mount_point) / "MyDrive"
    if my_drive.exists():
        return my_drive
    if not in_colab():
        return None
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return my_drive if my_drive.exists() else None


@dataclass
class Workspace:
    """Filesystem layout for models, caches and outputs."""

    root: Path

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def prepare(self) -> "Workspace":
        for path in (self.root, self.models, self.cache, self.outputs):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def configure_model_home(self) -> None:
        """Redirect third-party model caches into the workspace.

        Must be called *before* importing DECIMER, MolGrapher or torch.
        """
        hf_home = self.models / "hf"
        pystow_home = self.models / "pystow"
        torch_home = self.models / "torch"
        for path in (hf_home, pystow_home, torch_home):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("PYSTOW_HOME", str(pystow_home))
        os.environ.setdefault("TORCH_HOME", str(torch_home))

    def molclassifier_checkpoint(self) -> Path:
        return self.models / MOLCLASSIFIER_CHECKPOINT_NAME


def resolve_workspace(
    workspace: Optional[str] = None,
    use_drive: bool = False,
    drive_subdir: str = DEFAULT_WORKSPACE_NAME,
) -> Workspace:
    """Pick a workspace directory.

    Priority: explicit ``workspace`` > Google Drive (when ``use_drive``) >
    ``$STRUCTURE_FINDER_HOME`` > ``~/.structure_finder``.
    """
    if workspace:
        root = Path(workspace).expanduser()
    elif use_drive:
        my_drive = mount_google_drive()
        if my_drive is None:
            raise RuntimeError(
                "Google Drive is not available. Run inside Colab, or pass an "
                "explicit --workspace directory (e.g. a locally synced Drive folder)."
            )
        root = my_drive / drive_subdir
    elif os.environ.get("STRUCTURE_FINDER_HOME"):
        root = Path(os.environ["STRUCTURE_FINDER_HOME"]).expanduser()
    else:
        root = Path.home() / f".{DEFAULT_WORKSPACE_NAME}"
    ws = Workspace(root.resolve()).prepare()
    ws.configure_model_home()
    return ws


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    """Download ``url`` to ``destination`` unless it already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(destination)
    return destination


def ensure_molclassifier_checkpoint(workspace: Workspace, force: bool = False) -> Path:
    """Fetch the MolClassifier checkpoint (PatCID classification model)."""
    return download_file(
        MOLCLASSIFIER_CHECKPOINT_URL,
        workspace.molclassifier_checkpoint(),
        force=force,
    )

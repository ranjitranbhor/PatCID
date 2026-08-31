#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 of the PatCID pipeline: classify each detected image.

MolClassifier sorts chemical images into three classes:

``Clean``    a molecular structure - the only class that is worth handing to
             the recognition model
``Markush``  a Markush structure (generic structure with variable R groups); a
             SMILES cannot faithfully represent it, so these are recorded and
             flagged rather than matched
``Trash``    background / a segmentation error

Reported precision and recall are 93.4% / 84.6% on D2C-RND (PatCID, Table 3).

When the checkpoint is unavailable the pipeline falls back to
:class:`PassthroughClassifier`, which labels everything ``Clean`` - recognition
then simply runs on every region.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

CLEAN_LABELS = {"clean", "molecular structure", "molecule"}
MARKUSH_LABELS = {"markush", "markush structure"}
TRASH_LABELS = {"trash", "background"}


def load_molclassifier(checkpoint: str, device: str = "cpu"):
    """Load MolClassifier's Mask R-CNN from a checkpoint.

    Uses ``mol_classifier.classifier.ImageSeg`` from the MolClassifier repo,
    which reads the class list out of the checkpoint itself.
    """
    try:
        from mol_classifier.classifier import ImageSeg
    except ImportError as error:  # pragma: no cover - import guard
        raise ImportError(
            "MolClassifier is not importable. Install it with:\n"
            "  pip install 'molclassifier @ git+https://github.com/DS4SD/MolClassifier'\n"
            "or clone the repo and add it to PYTHONPATH."
        ) from error

    model = ImageSeg(device=device, checkpoint=checkpoint)
    model.set_model()
    model.load_model()
    return model


class BaseClassifier:
    name = "base"

    def classify(self, images: Sequence[np.ndarray]) -> List[tuple]:
        """Return ``(label, score)`` per input image."""
        raise NotImplementedError


class PassthroughClassifier(BaseClassifier):
    """Label everything ``Clean``; used when no checkpoint is available."""

    name = "passthrough"

    def classify(self, images: Sequence[np.ndarray]) -> List[tuple]:
        return [("Clean", 1.0) for _ in images]


class MolClassifier(BaseClassifier):
    """Classify already-cropped chemical images with MolClassifier.

    The underlying model is a detector, so it is applied to each crop and the
    highest-scoring detection covering the crop provides the label.  A crop with
    no detection is labelled ``Trash``.
    """

    name = "molclassifier"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        batch_size: int = 4,
        margin: int = 24,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.margin = margin
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = load_molclassifier(self.checkpoint, device=self.device)
        return self._model

    def classify(self, images: Sequence[np.ndarray]) -> List[tuple]:
        from PIL import Image

        if not images:
            return []
        model = self._load()
        # White padding keeps the depiction away from the image border, which is
        # closer to the full-page context the detector was trained on.
        padded = [
            Image.fromarray(
                np.pad(
                    image,
                    ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
                    mode="constant",
                    constant_values=255,
                )
            )
            for image in images
        ]
        annotations = model.evaluate(padded, batch_size=self.batch_size)
        results: List[tuple] = []
        for annotation in annotations:
            detections = annotation.get("annotations", [])
            if not detections:
                results.append(("Trash", 0.0))
                continue
            best = max(detections, key=lambda item: item.get("score", 0.0))
            results.append((best.get("label", "Trash"), float(best.get("score", 0.0))))
        return results


def normalise_label(label: Optional[str]) -> str:
    """Map a raw class name onto ``Clean`` / ``Markush`` / ``Trash``."""
    if not label:
        return "Trash"
    lowered = label.strip().lower()
    if lowered in CLEAN_LABELS:
        return "Clean"
    if lowered in MARKUSH_LABELS:
        return "Markush"
    if lowered in TRASH_LABELS:
        return "Trash"
    return label


def build_classifier(
    engine: str, checkpoint: Optional[str] = None, device: str = "cpu"
) -> BaseClassifier:
    """Instantiate a classifier by name (``molclassifier`` or ``none``)."""
    engine = (engine or "none").lower()
    if engine in {"none", "passthrough", "off"}:
        return PassthroughClassifier()
    if engine == "molclassifier":
        if not checkpoint:
            raise ValueError(
                "MolClassifier requires a checkpoint. Run "
                "`python -m structure_finder.setup_models --engine molclassifier`."
            )
        return MolClassifier(checkpoint=checkpoint, device=device)
    raise ValueError(f"Unknown classifier {engine!r}.")

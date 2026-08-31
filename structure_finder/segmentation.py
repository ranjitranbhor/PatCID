#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 of the PatCID pipeline: locate chemical-structure depictions on a page.

Three interchangeable engines are provided:

``decimer``       DECIMER-Segmentation (Mask R-CNN, Rajan et al.) - the
                  segmenter PatCID uses; 88.0% precision / 86.3% recall on
                  D2C-RND (Table 3 of the PatCID paper).
``molclassifier`` PatCID's own MolClassifier.  It is a Mask R-CNN with the
                  classes ``Clean`` / ``Markush`` / ``Trash``, so it detects
                  *and* labels regions in a single pass.
``heuristic``     A dependency-free connected-component segmenter.  It needs no
                  model download and is useful for smoke tests and for pages
                  where a single structure fills the sheet, but its quality is
                  far below the two learned segmenters - do not use it for
                  production recall.

All engines return :class:`Region` objects with pixel bounding boxes in the
coordinate frame of the page image handed to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SEGMENTER_ENGINES = ("decimer", "molclassifier", "heuristic")


@dataclass
class Region:
    """A candidate chemical-structure depiction cropped from a page."""

    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in page pixels
    image: np.ndarray  # RGB crop, uint8
    score: float = 1.0
    label: Optional[str] = None  # set by the classification step
    label_score: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


class BaseSegmenter:
    name = "base"

    def segment(self, page_image: np.ndarray) -> List[Region]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DECIMER-Segmentation
# ---------------------------------------------------------------------------


class DecimerSegmenter(BaseSegmenter):
    """Wrapper around ``decimer_segmentation.segment_chemical_structures``."""

    name = "decimer"

    def __init__(self, expand: bool = True) -> None:
        self.expand = expand
        self._segment_fn = None

    def _load(self):
        if self._segment_fn is None:
            try:
                from decimer_segmentation import segment_chemical_structures
            except ImportError as error:  # pragma: no cover - import guard
                raise ImportError(
                    "DECIMER-Segmentation is not installed. "
                    "Install it with: pip install decimer-segmentation"
                ) from error
            self._segment_fn = segment_chemical_structures
        return self._segment_fn

    def segment(self, page_image: np.ndarray) -> List[Region]:
        segment_fn = self._load()
        try:
            segments, bboxes = segment_fn(
                page_image, expand=self.expand, return_bboxes=True
            )
        except Exception as error:
            raise RuntimeError(
                "DECIMER-Segmentation failed. Its Mask R-CNN weights are "
                "downloaded from zenodo.org on first use, so this is usually a "
                "network problem. Pre-download them with:\n"
                "  python -m structure_finder.setup_models --engine decimer-seg\n"
                "Or fall back to '--segmenter heuristic' (lower recall).\n"
                f"Underlying error: {error}"
            ) from error
        regions: List[Region] = []
        for segment, bbox in zip(segments, bboxes):
            # DECIMER-Segmentation returns (y0, x0, y1, x1).
            y0, x0, y1, x1 = (int(value) for value in bbox)
            if segment is None or segment.size == 0:
                continue
            regions.append(
                Region(bbox=(x0, y0, x1, y1), image=segment, score=1.0)
            )
        return regions


# ---------------------------------------------------------------------------
# MolClassifier as a detector (PatCID's classification model)
# ---------------------------------------------------------------------------


class MolClassifierSegmenter(BaseSegmenter):
    """Use MolClassifier's Mask R-CNN for detection *and* classification.

    MolClassifier is trained on full document pages and emits one box per
    detected chemical image together with its class (``Clean`` = molecular
    structure, ``Markush`` = Markush structure, ``Trash`` = background).  Using
    it as the segmenter collapses PatCID's steps 1 and 2 into a single forward
    pass, which is the fastest CPU-only configuration.
    """

    name = "molclassifier"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        score_threshold: float = 0.5,
        padding: int = 8,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.score_threshold = score_threshold
        self.padding = padding
        self._model = None

    def _load(self):
        if self._model is None:
            from .classification import load_molclassifier

            self._model = load_molclassifier(self.checkpoint, device=self.device)
        return self._model

    def segment(self, page_image: np.ndarray) -> List[Region]:
        from PIL import Image

        model = self._load()
        annotations = model.evaluate([Image.fromarray(page_image)], batch_size=1)
        if not annotations:
            return []
        height, width = page_image.shape[:2]
        regions: List[Region] = []
        for annotation in annotations[0].get("annotations", []):
            score = float(annotation.get("score", 0.0))
            if score < self.score_threshold:
                continue
            x0, y0, x1, y1 = (int(round(value)) for value in annotation["bbox"])
            x0 = max(0, x0 - self.padding)
            y0 = max(0, y0 - self.padding)
            x1 = min(width, x1 + self.padding)
            y1 = min(height, y1 + self.padding)
            if x1 <= x0 or y1 <= y0:
                continue
            regions.append(
                Region(
                    bbox=(x0, y0, x1, y1),
                    image=page_image[y0:y1, x0:x1].copy(),
                    score=score,
                    label=annotation.get("label"),
                    label_score=score,
                )
            )
        return _sort_reading_order(regions)


# ---------------------------------------------------------------------------
# Heuristic fallback (no model weights required)
# ---------------------------------------------------------------------------


class HeuristicSegmenter(BaseSegmenter):
    """Connected-component segmenter used when no model weights are available.

    Ink pixels are dilated so that the bonds, atom labels and captions of one
    depiction merge into a single blob, then blobs are filtered by size, aspect
    ratio and ink density.  This finds isolated, reasonably large depictions;
    it will over-segment dense patent pages, so it is a fallback, not a
    replacement for DECIMER-Segmentation.
    """

    name = "heuristic"

    def __init__(
        self,
        min_area_fraction: float = 0.004,
        max_area_fraction: float = 0.85,
        min_side_px: int = 60,
        padding: int = 12,
        dilation_fraction: float = 0.012,
    ) -> None:
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.min_side_px = min_side_px
        self.padding = padding
        self.dilation_fraction = dilation_fraction

    def segment(self, page_image: np.ndarray) -> List[Region]:
        import cv2

        height, width = page_image.shape[:2]
        page_area = height * width
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        # Otsu on the inverted image: ink becomes foreground (255).
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel_size = max(3, int(self.dilation_fraction * min(height, width)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        merged = cv2.dilate(binary, kernel, iterations=1)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
        regions: List[Region] = []
        for index in range(1, count):
            x, y, box_w, box_h, _ = stats[index]
            area = box_w * box_h
            if area < self.min_area_fraction * page_area:
                continue
            if area > self.max_area_fraction * page_area:
                continue
            if box_w < self.min_side_px or box_h < self.min_side_px:
                continue
            aspect = box_w / float(box_h)
            if aspect > 12 or aspect < 1 / 12:
                continue  # a text line, not a depiction

            component = (labels[y : y + box_h, x : x + box_w] == index)
            ink = binary[y : y + box_h, x : x + box_w][component]
            density = float(np.count_nonzero(ink)) / max(1, ink.size)
            if density < 0.02 or density > 0.75:
                continue

            x0 = max(0, x - self.padding)
            y0 = max(0, y - self.padding)
            x1 = min(width, x + box_w + self.padding)
            y1 = min(height, y + box_h + self.padding)
            regions.append(
                Region(
                    bbox=(x0, y0, x1, y1),
                    image=page_image[y0:y1, x0:x1].copy(),
                    score=float(density),
                )
            )
        return _sort_reading_order(regions)


# ---------------------------------------------------------------------------


def _sort_reading_order(
    regions: List[Region], same_row_tolerance: int = 50
) -> List[Region]:
    """Sort regions top-to-bottom then left-to-right, as PatCID does."""
    if not regions:
        return regions
    ordered = sorted(regions, key=lambda region: region.bbox[1])
    rows: List[List[Region]] = [[ordered[0]]]
    for region in ordered[1:]:
        if abs(region.bbox[1] - rows[-1][-1].bbox[1]) < same_row_tolerance:
            rows[-1].append(region)
        else:
            rows.append([region])
    result: List[Region] = []
    for row in rows:
        result.extend(sorted(row, key=lambda region: region.bbox[0]))
    return result


def build_segmenter(engine: str, **kwargs) -> BaseSegmenter:
    """Instantiate a segmenter by name."""
    engine = engine.lower()
    if engine == "decimer":
        return DecimerSegmenter(expand=kwargs.get("expand", True))
    if engine == "molclassifier":
        checkpoint = kwargs.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "The 'molclassifier' segmenter requires a checkpoint path. "
                "Run `python -m structure_finder.setup_models --engine molclassifier`."
            )
        return MolClassifierSegmenter(
            checkpoint=checkpoint,
            device=kwargs.get("device", "cpu"),
            score_threshold=kwargs.get("score_threshold", 0.5),
        )
    if engine == "heuristic":
        return HeuristicSegmenter()
    raise ValueError(f"Unknown segmenter {engine!r}. Choose from {SEGMENTER_ENGINES}.")

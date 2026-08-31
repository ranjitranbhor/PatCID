#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3 of the PatCID pipeline: convert a structure depiction into SMILES.

Engines
-------
``molgrapher``  MolGrapher (Morin et al., ICCV 2023) - keypoint detector plus a
                GNN over a candidate "supergraph".  PatCID's choice: it runs on
                CPU roughly twice as fast as DECIMER and degrades gracefully on
                large molecules.  Batched.
``decimer``     DECIMER Image Transformer 2.x (EfficientNet-V2 encoder +
                transformer decoder).  Scored 67.2% vs MolGrapher's 63.0% on
                D2C-RND (PatCID, Table 3), so it is a good complement.  One
                image at a time, and it returns per-token confidences.
``ensemble``    Runs both and keeps the higher-confidence prediction, preferring
                whichever engine produces a structure that matches the query.

Every engine yields :class:`Prediction` objects with a SMILES string, a
confidence in ``[0, 1]`` and the engine name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

RECOGNIZER_ENGINES = ("molgrapher", "decimer", "ensemble")


@dataclass
class Prediction:
    """One SMILES prediction for one depiction."""

    smiles: Optional[str]
    confidence: float
    engine: str

    def as_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "confidence": round(float(self.confidence), 4),
            "recognition_engine": self.engine,
        }


class BaseRecognizer:
    name = "base"

    def predict(self, images: Sequence[np.ndarray]) -> List[Prediction]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MolGrapher
# ---------------------------------------------------------------------------


class MolgrapherRecognizer(BaseRecognizer):
    """Wrapper around ``molgrapher.models.molgrapher_model.MolgrapherModel``."""

    name = "molgrapher"

    def __init__(
        self,
        force_cpu: bool = False,
        chunk_size: int = 64,
        node_classifier_variant: str = "gc_no_stereo_model",
        assign_stereo: bool = False,
        extra_args: Optional[dict] = None,
    ) -> None:
        self.args = {
            "force_cpu": force_cpu,
            "chunk_size": chunk_size,
            "node_classifier_variant": node_classifier_variant,
            "assign_stereo": assign_stereo,
            "visualize": False,
            "visualize_rdkit": False,
            "clean": False,
            "save_mol_folder": "",
        }
        if extra_args:
            self.args.update(extra_args)
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from molgrapher.models.molgrapher_model import MolgrapherModel
            except ImportError as error:  # pragma: no cover - import guard
                raise ImportError(
                    "MolGrapher is not installed. Install it with:\n"
                    "  pip install -e '.[cpu]'   # inside a MolGrapher checkout\n"
                    "or: pip install 'molgrapher @ git+https://github.com/DS4SD/MolGrapher'"
                ) from error
            try:
                self._model = MolgrapherModel(self.args)
            except Exception as error:
                raise RuntimeError(
                    "MolGrapher failed to load. Its weights are fetched from "
                    "huggingface.co/ds4sd/MolGrapher on first use, so this is "
                    "usually a network or cache problem. Pre-download them with:\n"
                    "  python -m structure_finder.setup_models --engine molgrapher\n"
                    f"Underlying error: {error}"
                ) from error
        return self._model

    def predict(self, images: Sequence[np.ndarray]) -> List[Prediction]:
        from PIL import Image

        if not images:
            return []
        model = self._load()
        pil_images = [Image.fromarray(image) for image in images]
        annotations = model.predict_batch(pil_images)

        predictions: List[Prediction] = [
            Prediction(None, 0.0, self.name) for _ in images
        ]
        for index, annotation in enumerate(annotations):
            if index >= len(predictions):
                break
            smiles = annotation.get("smi") if isinstance(annotation, dict) else None
            confidence = (
                float(annotation.get("conf", 0.0)) if isinstance(annotation, dict) else 0.0
            )
            predictions[index] = Prediction(smiles or None, confidence, self.name)
        return predictions


# ---------------------------------------------------------------------------
# DECIMER Image Transformer
# ---------------------------------------------------------------------------


class DecimerRecognizer(BaseRecognizer):
    """Wrapper around ``DECIMER.predict_SMILES``."""

    name = "decimer"

    def __init__(self, hand_drawn: bool = False) -> None:
        self.hand_drawn = hand_drawn
        self._predict = None

    def _load(self):
        if self._predict is None:
            try:
                from DECIMER import predict_SMILES
            except ImportError as error:  # pragma: no cover - import guard
                raise ImportError(
                    "DECIMER is not installed. Install it with: pip install decimer"
                ) from error
            except Exception as error:
                # Importing DECIMER downloads its SavedModel from Zenodo.
                raise RuntimeError(
                    "DECIMER failed to load its model. The weights are fetched "
                    "from zenodo.org on first import, so this is usually a "
                    "network problem. Pre-download them with:\n"
                    "  python -m structure_finder.setup_models --engine decimer\n"
                    f"Underlying error: {error}"
                ) from error
            self._predict = predict_SMILES
        return self._predict

    def predict(self, images: Sequence[np.ndarray]) -> List[Prediction]:
        predict_smiles = self._load()
        predictions: List[Prediction] = []
        for image in images:
            try:
                smiles, token_confidences = predict_smiles(
                    image, confidence=True, hand_drawn=self.hand_drawn
                )
            except Exception:
                predictions.append(Prediction(None, 0.0, self.name))
                continue
            confidence = _mean_token_confidence(token_confidences)
            predictions.append(Prediction(smiles or None, confidence, self.name))
        return predictions


def _mean_token_confidence(token_confidences) -> float:
    """Average DECIMER's per-token confidences into a single score."""
    try:
        values = [float(value) for _, value in token_confidences]
    except (TypeError, ValueError):
        return 0.0
    if not values:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class EnsembleRecognizer(BaseRecognizer):
    """Run several recognizers and keep the best prediction per image.

    "Best" means: a prediction that matches the query wins over one that does
    not (matching is what the user asked for, and a match is strong evidence
    the reading is correct); among equals, the higher confidence wins.
    """

    name = "ensemble"

    def __init__(self, recognizers: Sequence[BaseRecognizer], matcher=None) -> None:
        self.recognizers = list(recognizers)
        self.matcher = matcher

    def predict(self, images: Sequence[np.ndarray]) -> List[Prediction]:
        if not images:
            return []
        per_engine = [recognizer.predict(images) for recognizer in self.recognizers]
        best: List[Prediction] = []
        for index in range(len(images)):
            candidates = [
                engine_predictions[index]
                for engine_predictions in per_engine
                if index < len(engine_predictions)
            ]
            candidates = [candidate for candidate in candidates if candidate.smiles]
            if not candidates:
                best.append(Prediction(None, 0.0, self.name))
                continue

            def rank(prediction: Prediction):
                matched = (
                    self.matcher.match(prediction.smiles).matched
                    if self.matcher is not None
                    else False
                )
                return (not matched, -prediction.confidence)

            best.append(sorted(candidates, key=rank)[0])
        return best


def build_recognizer(engine: str, matcher=None, **kwargs) -> BaseRecognizer:
    """Instantiate a recognizer by name."""
    engine = engine.lower()
    if engine == "molgrapher":
        return MolgrapherRecognizer(
            force_cpu=kwargs.get("force_cpu", False),
            chunk_size=kwargs.get("chunk_size", 64),
            node_classifier_variant=kwargs.get(
                "node_classifier_variant", "gc_no_stereo_model"
            ),
        )
    if engine == "decimer":
        return DecimerRecognizer(hand_drawn=kwargs.get("hand_drawn", False))
    if engine == "ensemble":
        return EnsembleRecognizer(
            [
                MolgrapherRecognizer(
                    force_cpu=kwargs.get("force_cpu", False),
                    chunk_size=kwargs.get("chunk_size", 64),
                ),
                DecimerRecognizer(hand_drawn=kwargs.get("hand_drawn", False)),
            ],
            matcher=matcher,
        )
    raise ValueError(
        f"Unknown recognizer {engine!r}. Choose from {RECOGNIZER_ENGINES}."
    )

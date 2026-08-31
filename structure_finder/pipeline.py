#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestration: documents in, structure hits out.

The pipeline is a direct implementation of PatCID's Figure 2 ingestion pipeline
applied to user-supplied documents, with a matching stage bolted on:

    document -> page images -> segmentation -> classification -> recognition
             -> SMILES -> RDKit match against the query -> hits

Extraction (everything up to and including recognition) is *query independent*,
so it is cached per document under ``<workspace>/cache``.  Searching a second
structure across the same documents therefore costs milliseconds.  The cached
record mirrors PatCID's ``patcid_patent_to_molecules_*.jsonl`` schema: a list of
``figures``, each with a page, a bounding box, a class and a SMILES.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .classification import build_classifier, normalise_label
from .documents import Document, Page
from .matching import MoleculeMatcher, ParsedQuery, canonical_smiles
from .recognition import Prediction, build_recognizer
from .segmentation import Region, build_segmenter
from .storage import Workspace

CACHE_VERSION = 3


@dataclass
class Figure:
    """One chemical image found in a document, with its recognised structure."""

    page: int
    bbox: List[int]  # [x0, y0, x1, y1] in page pixels
    page_width: int
    page_height: int
    dpi: int
    detection_score: float
    figure_class: str
    class_score: Optional[float]
    smiles: Optional[str]
    canonical_smiles: Optional[str]
    recognition_confidence: float
    recognition_engine: Optional[str]
    source: str = "render"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DocumentExtraction:
    """Everything the query-independent stages found in one document."""

    document: Dict[str, object]
    figures: List[Figure] = field(default_factory=list)
    page_count: int = 0
    text_smiles: List[str] = field(default_factory=list)
    text_inchikeys: List[str] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    cache_version: int = CACHE_VERSION

    def as_dict(self) -> Dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "document": self.document,
            "page_count": self.page_count,
            "config": self.config,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "text_smiles": self.text_smiles,
            "text_inchikeys": self.text_inchikeys,
            "figures": [figure.as_dict() for figure in self.figures],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DocumentExtraction":
        return cls(
            document=payload["document"],
            figures=[Figure(**figure) for figure in payload.get("figures", [])],
            page_count=int(payload.get("page_count", 0)),
            text_smiles=list(payload.get("text_smiles", [])),
            text_inchikeys=list(payload.get("text_inchikeys", [])),
            config=dict(payload.get("config", {})),
            runtime_seconds=float(payload.get("runtime_seconds", 0.0)),
            cache_version=int(payload.get("cache_version", 0)),
        )


@dataclass
class Hit:
    """A figure whose recognised structure matches the query."""

    document: str
    document_path: str
    page: int
    bbox: List[int]
    page_width: int
    page_height: int
    figure_class: str
    smiles: Optional[str]
    match_mode: Optional[str]
    similarity: float
    recognition_confidence: float
    recognition_engine: Optional[str]
    query_label: str
    evidence: str = "image"  # "image" | "text"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class PipelineConfig:
    """Engine selection and thresholds for one run."""

    segmenter: str = "decimer"
    classifier: str = "molclassifier"
    recognizer: str = "molgrapher"
    dpi: int = 300
    device: str = "cpu"
    min_recognition_confidence: float = 0.0
    keep_classes: Sequence[str] = ("Clean",)
    recognize_markush: bool = False
    scan_text: bool = True
    molclassifier_checkpoint: Optional[str] = None
    batch_size: int = 16
    pages: Optional[List[int]] = None

    def fingerprint(self) -> Dict[str, object]:
        """Config fields that invalidate the extraction cache when changed."""
        return {
            "segmenter": self.segmenter,
            "classifier": self.classifier,
            "recognizer": self.recognizer,
            "dpi": self.dpi,
            "keep_classes": sorted(self.keep_classes),
            "recognize_markush": self.recognize_markush,
            "pages": self.pages,
        }


# ---------------------------------------------------------------------------
# Text scan: molecules written out as SMILES / InChI / InChIKey in the text
# ---------------------------------------------------------------------------

_INCHIKEY_RE = re.compile(r"\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b")
_INCHI_RE = re.compile(r"InChI=1S?/[^\s,;\)\]]+")
# Conservative: a run of SMILES-legal characters, at least 6 long, containing a
# ring closure, a branch or a bond symbol - avoids matching ordinary words.
_SMILES_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9])((?=[^\s]{6,})[A-Za-z0-9@\+\-\[\]\(\)=#\\/%\.]{6,})(?![A-Za-z0-9])"
)


def _balanced_candidate(candidate: str) -> Optional[str]:
    """Trim a raw text match down to a parseable SMILES, or give up.

    Prose puts molecules inside brackets and sentences - "(SMILES: CCO)." - so
    the raw match often carries unbalanced trailing punctuation.  Drop trailing
    characters until the parentheses/brackets balance and RDKit accepts it.
    """
    from rdkit import Chem

    trimmed = candidate
    for _ in range(6):
        if not trimmed or len(trimmed) < 4:
            return None
        if trimmed.count("(") == trimmed.count(")") and trimmed.count(
            "["
        ) == trimmed.count("]"):
            mol = Chem.MolFromSmiles(trimmed, sanitize=True)
            if mol is not None:
                return trimmed
        trimmed = trimmed.rstrip(".,;:")
        if trimmed and trimmed[-1] in ")]":
            trimmed = trimmed[:-1]
        else:
            return None
    return None


def scan_text_for_structures(text: str, max_candidates: int = 20000):
    """Find explicit SMILES / InChI / InChIKey strings in document text.

    Patents and papers frequently print the structure as text next to (or
    instead of) the drawing.  This is cheap, needs no model, and complements the
    image pipeline.  Returns ``(canonical_smiles_list, inchikeys)``.
    """
    from rdkit import Chem

    if not text:
        return [], []

    inchikeys = sorted(set(_INCHIKEY_RE.findall(text)))

    smiles_found: List[str] = []
    seen: set = set()

    for inchi in _INCHI_RE.findall(text)[:2000]:
        try:
            mol = Chem.MolFromInchi(inchi)
        except Exception:
            mol = None
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if smiles not in seen:
                seen.add(smiles)
                smiles_found.append(smiles)

    for index, match in enumerate(_SMILES_CANDIDATE_RE.finditer(text)):
        if index > max_candidates:
            break
        raw = match.group(1)
        if not any(character in raw for character in "()=#[]123456789"):
            continue
        if raw.isdigit() or raw.isalpha():
            continue
        candidate = _balanced_candidate(raw)
        if candidate is None:
            continue
        mol = Chem.MolFromSmiles(candidate, sanitize=True)
        if mol is None or mol.GetNumHeavyAtoms() < 4:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles in seen:
            continue
        seen.add(smiles)
        smiles_found.append(smiles)

    return smiles_found, inchikeys


# ---------------------------------------------------------------------------


class StructureFinder:
    """Find a query structure inside a set of documents."""

    def __init__(
        self,
        workspace: Workspace,
        config: Optional[PipelineConfig] = None,
        matcher: Optional[MoleculeMatcher] = None,
        verbose: bool = True,
    ) -> None:
        self.workspace = workspace
        self.config = config or PipelineConfig()
        self.matcher = matcher
        self.verbose = verbose
        self._segmenter = None
        self._classifier = None
        self._recognizer = None

    # -- lazily built engines --------------------------------------------------

    @property
    def segmenter(self):
        if self._segmenter is None:
            self._segmenter = build_segmenter(
                self.config.segmenter,
                checkpoint=self.config.molclassifier_checkpoint,
                device=self.config.device,
            )
        return self._segmenter

    @property
    def classifier(self):
        if self._classifier is None:
            engine = self.config.classifier
            # The MolClassifier segmenter already emits a class per region.
            if self.config.segmenter == "molclassifier" and engine == "molclassifier":
                engine = "none"
            self._classifier = build_classifier(
                engine,
                checkpoint=self.config.molclassifier_checkpoint,
                device=self.config.device,
            )
        return self._classifier

    @property
    def recognizer(self):
        if self._recognizer is None:
            self._recognizer = build_recognizer(
                self.config.recognizer,
                matcher=self.matcher,
                force_cpu=self.config.device == "cpu",
                chunk_size=self.config.batch_size,
            )
        return self._recognizer

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[structure-finder] {message}", flush=True)

    # -- extraction (query independent, cached) --------------------------------

    def _cache_path(self, document: Document) -> Path:
        import hashlib

        fingerprint = json.dumps(self.config.fingerprint(), sort_keys=True)
        key = hashlib.sha256(
            (document.content_hash() + fingerprint).encode("utf-8")
        ).hexdigest()[:32]
        return self.workspace.cache / f"{document.path.stem[:60]}.{key}.json"

    def extract(self, document: Document, use_cache: bool = True) -> DocumentExtraction:
        """Run segmentation + classification + recognition over one document."""
        cache_path = self._cache_path(document)
        if use_cache and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                if payload.get("cache_version") == CACHE_VERSION:
                    self._log(f"{document.name}: loaded from cache")
                    return DocumentExtraction.from_dict(payload)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        started = time.time()
        extraction = DocumentExtraction(
            document=document.metadata(),
            config=self.config.fingerprint(),
        )

        text_parts: List[str] = []
        page_index = 0
        for page in document.pages(self.config.pages):
            page_index += 1
            if page.text:
                text_parts.append(page.text)
            figures = self._process_page(page)
            extraction.figures.extend(figures)
            self._log(
                f"{document.name} page {page.number}: "
                f"{len(figures)} chemical image(s)"
            )
        extraction.page_count = page_index

        if self.config.scan_text:
            text = "\n".join(text_parts) or document.full_text()
            smiles_list, inchikeys = scan_text_for_structures(text)
            extraction.text_smiles = smiles_list
            extraction.text_inchikeys = inchikeys

        extraction.runtime_seconds = time.time() - started
        document.cleanup()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(extraction.as_dict(), indent=1))
        return extraction

    def _process_page(self, page: Page) -> List[Figure]:
        regions: List[Region] = self.segmenter.segment(page.image)
        if not regions:
            return []

        # Classify regions that the segmenter did not already label.
        unlabelled = [region for region in regions if region.label is None]
        if unlabelled:
            labels = self.classifier.classify([region.image for region in unlabelled])
            for region, (label, score) in zip(unlabelled, labels):
                region.label = label
                region.label_score = score

        for region in regions:
            region.label = normalise_label(region.label)

        keep = set(self.config.keep_classes)
        if self.config.recognize_markush:
            keep = keep | {"Markush"}
        to_recognize = [region for region in regions if region.label in keep]

        predictions: List[Prediction] = []
        for start in range(0, len(to_recognize), self.config.batch_size):
            batch = to_recognize[start : start + self.config.batch_size]
            predictions.extend(self.recognizer.predict([r.image for r in batch]))

        prediction_by_region = {
            id(region): prediction
            for region, prediction in zip(to_recognize, predictions)
        }

        figures: List[Figure] = []
        for region in regions:
            if region.label == "Trash":
                continue
            prediction = prediction_by_region.get(id(region))
            smiles = prediction.smiles if prediction else None
            figures.append(
                Figure(
                    page=page.number,
                    bbox=[int(value) for value in region.bbox],
                    page_width=page.width,
                    page_height=page.height,
                    dpi=page.dpi,
                    detection_score=round(float(region.score), 4),
                    figure_class=region.label or "Clean",
                    class_score=(
                        round(float(region.label_score), 4)
                        if region.label_score is not None
                        else None
                    ),
                    smiles=smiles,
                    canonical_smiles=canonical_smiles(smiles) if smiles else None,
                    recognition_confidence=(
                        round(float(prediction.confidence), 4) if prediction else 0.0
                    ),
                    recognition_engine=prediction.engine if prediction else None,
                    source=page.source,
                )
            )
        return figures

    # -- search (query dependent, cheap) ---------------------------------------

    def search(
        self,
        documents: Iterable[Document],
        matcher: Optional[MoleculeMatcher] = None,
        use_cache: bool = True,
    ) -> Dict[str, object]:
        """Extract structures from ``documents`` and match them to the query."""
        matcher = matcher or self.matcher
        if matcher is None:
            raise ValueError("A MoleculeMatcher (query) is required to search.")
        self.matcher = matcher

        hits: List[Hit] = []
        extractions: List[DocumentExtraction] = []
        for document in documents:
            self._log(f"processing {document.name}")
            extraction = self.extract(document, use_cache=use_cache)
            extractions.append(extraction)
            hits.extend(self._match_extraction(extraction, matcher))

        hits.sort(
            key=lambda hit: (
                hit.document,
                hit.page,
                -hit.similarity,
            )
        )
        return {
            "queries": [query.as_dict() for query in matcher.queries],
            "match_modes": list(matcher.modes),
            "similarity_threshold": matcher.similarity_threshold,
            "config": self.config.fingerprint(),
            "documents": [
                {
                    "filename": extraction.document["filename"],
                    "path": extraction.document["path"],
                    "pages": extraction.page_count,
                    "chemical_images": len(extraction.figures),
                    "recognised_structures": sum(
                        1 for figure in extraction.figures if figure.smiles
                    ),
                    "markush_images": sum(
                        1
                        for figure in extraction.figures
                        if figure.figure_class == "Markush"
                    ),
                    "hits": sum(
                        1
                        for hit in hits
                        if hit.document == extraction.document["filename"]
                    ),
                    "runtime_seconds": round(extraction.runtime_seconds, 2),
                }
                for extraction in extractions
            ],
            "hits": [hit.as_dict() for hit in hits],
            "extractions": extractions,
        }

    def _match_extraction(
        self, extraction: DocumentExtraction, matcher: MoleculeMatcher
    ) -> List[Hit]:
        hits: List[Hit] = []
        filename = str(extraction.document["filename"])
        path = str(extraction.document["path"])

        for figure in extraction.figures:
            if not figure.smiles:
                continue
            if figure.recognition_confidence < self.config.min_recognition_confidence:
                continue
            result = matcher.match(figure.canonical_smiles or figure.smiles)
            if not result.matched:
                continue
            hits.append(
                Hit(
                    document=filename,
                    document_path=path,
                    page=figure.page,
                    bbox=figure.bbox,
                    page_width=figure.page_width,
                    page_height=figure.page_height,
                    figure_class=figure.figure_class,
                    smiles=figure.canonical_smiles or figure.smiles,
                    match_mode=result.mode,
                    similarity=result.similarity,
                    recognition_confidence=figure.recognition_confidence,
                    recognition_engine=figure.recognition_engine,
                    query_label=result.query_label,
                    evidence="image",
                )
            )

        # Text evidence: the structure spelled out as SMILES/InChI in the text.
        for smiles in extraction.text_smiles:
            result = matcher.match(smiles)
            if not result.matched:
                continue
            hits.append(
                Hit(
                    document=filename,
                    document_path=path,
                    page=0,
                    bbox=[0, 0, 0, 0],
                    page_width=0,
                    page_height=0,
                    figure_class="Text",
                    smiles=smiles,
                    match_mode=result.mode,
                    similarity=result.similarity,
                    recognition_confidence=1.0,
                    recognition_engine="text-scan",
                    query_label=result.query_label,
                    evidence="text",
                )
            )

        query_keys = {
            query.inchikey_no_stereo
            for query in matcher.queries
            if query.inchikey_no_stereo
        }
        for key in extraction.text_inchikeys:
            if key in query_keys or key.split("-")[0] in {
                query.skeleton for query in matcher.queries if query.skeleton
            }:
                hits.append(
                    Hit(
                        document=filename,
                        document_path=path,
                        page=0,
                        bbox=[0, 0, 0, 0],
                        page_width=0,
                        page_height=0,
                        figure_class="Text",
                        smiles=None,
                        match_mode="exact" if key in query_keys else "connectivity",
                        similarity=1.0,
                        recognition_confidence=1.0,
                        recognition_engine="text-scan-inchikey",
                        query_label=key,
                        evidence="text",
                    )
                )
        return hits


def resolve_query(
    queries: Sequence[str],
    modes: Sequence[str],
    similarity_threshold: float = 0.85,
) -> MoleculeMatcher:
    """Build a :class:`MoleculeMatcher` from raw query strings."""
    from .matching import parse_query

    parsed: List[ParsedQuery] = [parse_query(query) for query in queries]
    return MoleculeMatcher(
        parsed, modes=modes, similarity_threshold=similarity_threshold
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""structure_finder - find a chemical structure inside PDF / Word documents.

The pipeline follows PatCID (Morin et al., *Nature Communications* 15, 6532,
2024), Figure 2:

    1. Segmentation   - DECIMER-Segmentation locates chemical images on a page
    2. Classification - MolClassifier labels them Clean / Markush / Trash
    3. Recognition    - MolGrapher (or DECIMER) converts them to SMILES

and adds a fourth step for the task at hand:

    4. Matching       - RDKit compares each recognised structure with the
                        user's query SMILES (stereo-insensitive InChIKey, as
                        PatCID's own evaluation does, plus looser modes)

Quick start
-----------
>>> from structure_finder import find_structure
>>> results = find_structure(
...     documents=["patent.pdf", "report.docx"],
...     smiles="CC(=O)Oc1ccccc1C(=O)O",
... )
>>> for hit in results["hits"]:
...     print(hit["document"], hit["page"], hit["match_mode"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .documents import DEFAULT_DPI, Document, collect_documents
from .matching import MATCH_MODES, MoleculeMatcher, parse_query
from .pipeline import Figure, Hit, PipelineConfig, StructureFinder, resolve_query
from .report import annotate_hits, format_summary, save_all
from .storage import Workspace, ensure_molclassifier_checkpoint, resolve_workspace

__version__ = "1.0.0"

__all__ = [
    "find_structure",
    "StructureFinder",
    "PipelineConfig",
    "MoleculeMatcher",
    "Document",
    "Figure",
    "Hit",
    "Workspace",
    "collect_documents",
    "resolve_workspace",
    "resolve_query",
    "parse_query",
    "format_summary",
    "annotate_hits",
    "save_all",
    "MATCH_MODES",
    "DEFAULT_DPI",
]


def find_structure(
    documents: Sequence[str],
    smiles,
    workspace: Optional[str] = None,
    use_drive: bool = False,
    segmenter: str = "decimer",
    classifier: str = "molclassifier",
    recognizer: str = "molgrapher",
    match_modes: Sequence[str] = ("exact", "connectivity", "tautomer"),
    similarity_threshold: float = 0.85,
    dpi: int = DEFAULT_DPI,
    device: str = "cpu",
    pages: Optional[List[int]] = None,
    scan_text: bool = True,
    recognize_markush: bool = False,
    use_cache: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """One-call API: search ``smiles`` across ``documents``.

    Args:
        documents: Paths to PDF / DOCX / image files, or directories.
        smiles: A query SMILES, or a list of them. Prefix a string with
            ``smarts:`` to search a substructure pattern instead.
        workspace: Directory holding models, cache and outputs.
        use_drive: Mount Google Drive and use ``MyDrive/structure_finder``.
        segmenter: ``decimer`` | ``molclassifier`` | ``heuristic``.
        classifier: ``molclassifier`` | ``none``.
        recognizer: ``molgrapher`` | ``decimer`` | ``ensemble``.
        match_modes: Any of ``exact``, ``connectivity``, ``tautomer``,
            ``substructure``, ``similarity``.
        pages: 1-based page selection applied to every document.

    Returns:
        A dict with ``queries``, ``documents``, ``hits`` and ``extractions``.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    if isinstance(documents, (str, Path)):
        # Iterating a bare string yields characters, so wrap it rather than
        # searching for documents named "c", "o", "n", ...
        documents = [documents]

    space = resolve_workspace(workspace=workspace, use_drive=use_drive)

    checkpoint = None
    if "molclassifier" in {segmenter, classifier}:
        try:
            checkpoint = str(ensure_molclassifier_checkpoint(space))
        except Exception:
            checkpoint = None
            if classifier == "molclassifier":
                classifier = "none"
            if segmenter == "molclassifier":
                segmenter = "heuristic"

    matcher = resolve_query(smiles, match_modes, similarity_threshold)
    config = PipelineConfig(
        segmenter=segmenter,
        classifier=classifier,
        recognizer=recognizer,
        dpi=dpi,
        device=device,
        scan_text=scan_text,
        recognize_markush=recognize_markush,
        molclassifier_checkpoint=checkpoint,
        pages=pages,
    )
    finder = StructureFinder(
        workspace=space, config=config, matcher=matcher, verbose=verbose
    )
    return finder.search(collect_documents(list(documents), dpi=dpi), use_cache=use_cache)

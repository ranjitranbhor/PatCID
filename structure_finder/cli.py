#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point.

    python -m structure_finder --smiles "CC(=O)Oc1ccccc1C(=O)O" report.pdf notes.docx

Run ``python -m structure_finder --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .documents import DEFAULT_DPI, collect_documents
from .matching import MATCH_MODES
from .pipeline import PipelineConfig, StructureFinder, resolve_query
from .recognition import RECOGNIZER_ENGINES
from .report import format_summary, save_all
from .segmentation import SEGMENTER_ENGINES
from .storage import ensure_molclassifier_checkpoint, resolve_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structure_finder",
        description=(
            "Find a chemical structure (given as SMILES) inside PDF and Word "
            "documents, using the PatCID pipeline: DECIMER-Segmentation -> "
            "MolClassifier -> MolGrapher / DECIMER -> RDKit matching."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "documents",
        nargs="+",
        help="PDF / DOCX / image files, or directories to scan recursively.",
    )
    query_group = parser.add_argument_group("query")
    query_group.add_argument(
        "-s",
        "--smiles",
        action="append",
        default=[],
        help=(
            "Query structure as SMILES. Repeat for several queries. "
            "Prefix with 'smarts:' to give a SMARTS substructure pattern."
        ),
    )
    query_group.add_argument(
        "--smiles-file",
        help="Text file with one query SMILES per line.",
    )
    query_group.add_argument(
        "--match-mode",
        action="append",
        choices=list(MATCH_MODES),
        default=[],
        help=(
            "Matching criteria to apply (repeatable). "
            "Default: exact, connectivity, tautomer."
        ),
    )
    query_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Tanimoto threshold for the 'similarity' match mode.",
    )

    engine_group = parser.add_argument_group("engines")
    engine_group.add_argument(
        "--segmenter",
        choices=list(SEGMENTER_ENGINES),
        default="decimer",
        help="Chemical-image detector. 'heuristic' needs no model weights.",
    )
    engine_group.add_argument(
        "--classifier",
        choices=["molclassifier", "none"],
        default="molclassifier",
        help="Clean/Markush/Trash classification of detected images.",
    )
    engine_group.add_argument(
        "--recognizer",
        choices=list(RECOGNIZER_ENGINES),
        default="molgrapher",
        help="Optical chemical structure recognition model.",
    )
    engine_group.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for the torch-based models.",
    )
    engine_group.add_argument(
        "--batch-size", type=int, default=16, help="Images per recognition batch."
    )
    engine_group.add_argument(
        "--recognize-markush",
        action="store_true",
        help=(
            "Also run recognition on Markush structures. Their SMILES is not "
            "chemically faithful, so matches are indicative only."
        ),
    )

    doc_group = parser.add_argument_group("documents")
    doc_group.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Page rendering resolution.",
    )
    doc_group.add_argument(
        "--pages",
        help="Restrict to these 1-based pages, e.g. '1-10,15'.",
    )
    doc_group.add_argument(
        "--no-text-scan",
        action="store_true",
        help="Skip searching the document text for literal SMILES/InChI strings.",
    )
    doc_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard recognitions below this confidence before matching.",
    )

    storage_group = parser.add_argument_group("storage")
    storage_group.add_argument(
        "--workspace",
        help=(
            "Directory for models, cache and outputs. "
            "Default: $STRUCTURE_FINDER_HOME or ~/.structure_finder."
        ),
    )
    storage_group.add_argument(
        "--drive",
        action="store_true",
        help="Mount Google Drive and use MyDrive/structure_finder as workspace.",
    )
    storage_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-extract documents even if a cached extraction exists.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "-o",
        "--output-dir",
        help="Where to write the report. Default: <workspace>/outputs/<timestamp>.",
    )
    output_group.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not render annotated pages for the matches.",
    )
    output_group.add_argument(
        "--quiet", action="store_true", help="Suppress progress logging."
    )
    return parser


def parse_pages(spec: Optional[str]) -> Optional[List[int]]:
    """Parse a page selection like ``'1-10,15,20-22'``."""
    if not spec:
        return None
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages)) or None


def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = list(args.smiles)
    if args.smiles_file:
        text = Path(args.smiles_file).expanduser().read_text()
        queries.extend(
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return queries


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    queries = load_queries(args)
    if not queries:
        parser.error("At least one query is required (--smiles or --smiles-file).")

    # Resolve the workspace first: it redirects HF_HOME / PYSTOW_HOME / TORCH_HOME
    # before any model library gets imported.
    workspace = resolve_workspace(workspace=args.workspace, use_drive=args.drive)
    if not args.quiet:
        print(f"[structure-finder] workspace: {workspace.root}")

    checkpoint = None
    if "molclassifier" in {args.segmenter, args.classifier}:
        try:
            checkpoint = str(ensure_molclassifier_checkpoint(workspace))
        except Exception as error:
            print(
                f"[structure-finder] WARNING: could not obtain the MolClassifier "
                f"checkpoint ({error}). Continuing without classification.",
                file=sys.stderr,
            )
            if args.classifier == "molclassifier":
                args.classifier = "none"
            if args.segmenter == "molclassifier":
                args.segmenter = "heuristic"

    modes = args.match_mode or ["exact", "connectivity", "tautomer"]
    try:
        matcher = resolve_query(queries, modes, args.similarity_threshold)
    except ValueError as error:
        parser.error(str(error))

    config = PipelineConfig(
        segmenter=args.segmenter,
        classifier=args.classifier,
        recognizer=args.recognizer,
        dpi=args.dpi,
        device=args.device,
        min_recognition_confidence=args.min_confidence,
        recognize_markush=args.recognize_markush,
        scan_text=not args.no_text_scan,
        molclassifier_checkpoint=checkpoint,
        batch_size=args.batch_size,
        pages=parse_pages(args.pages),
    )

    try:
        documents = collect_documents(args.documents, dpi=args.dpi)
    except FileNotFoundError as error:
        parser.error(str(error))
    if not documents:
        parser.error("No supported documents found in the given inputs.")

    finder = StructureFinder(
        workspace=workspace, config=config, matcher=matcher, verbose=not args.quiet
    )
    results = finder.search(documents, use_cache=not args.no_cache)

    print()
    print(format_summary(results))

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else workspace.outputs / time_stamp()
    )
    written = save_all(results, output_dir, annotate=not args.no_annotate)
    print()
    print(f"Report written to {output_dir}")
    for key, value in written.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} file(s)")
        else:
            print(f"  {key}: {value}")

    return 0 if results["hits"] else 2


def time_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())

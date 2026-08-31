#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reporting: JSON, CSV, a console summary, and annotated page images.

PatCID's distinguishing feature over SureChEMBL / Google Patents is that it
keeps the *provenance* of a molecule - the exact page and bounding box where it
was drawn.  This module preserves that: every hit can be rendered back onto its
page, highlighted the way ``PatCID/src/display.py`` does it (blue for the query
molecule, red for Markush structures).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HIT_CSV_COLUMNS = [
    "document",
    "page",
    "evidence",
    "match_mode",
    "similarity",
    "figure_class",
    "smiles",
    "recognition_confidence",
    "recognition_engine",
    "query_label",
    "bbox",
    "document_path",
]

QUERY_COLOR = (0, 122, 255)
MARKUSH_COLOR = (239, 25, 25)
OTHER_COLOR = (140, 140, 140)


def write_json(results: Dict[str, object], path: Path) -> Path:
    """Write the full result document (without the bulky extraction objects)."""
    payload = {key: value for key, value in results.items() if key != "extractions"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_extractions(results: Dict[str, object], path: Path) -> Path:
    """Write per-document extractions as JSONL, in PatCID's schema shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for extraction in results.get("extractions", []):
            handle.write(json.dumps(extraction.as_dict()) + "\n")
    return path


def write_csv(results: Dict[str, object], path: Path) -> Path:
    """Write one row per hit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIT_CSV_COLUMNS)
        writer.writeheader()
        for hit in results.get("hits", []):
            row = {column: hit.get(column) for column in HIT_CSV_COLUMNS}
            row["bbox"] = ",".join(str(value) for value in hit.get("bbox", []))
            writer.writerow(row)
    return path


def format_summary(results: Dict[str, object]) -> str:
    """Human-readable console summary."""
    lines: List[str] = []
    queries = results.get("queries", [])
    lines.append("Query structure(s):")
    for query in queries:
        lines.append(f"  - {query['query']}")
        if query.get("canonical_smiles") and query["canonical_smiles"] != query["query"]:
            lines.append(f"      canonical (no stereo): {query['canonical_smiles']}")
        if query.get("inchikey_no_stereo"):
            lines.append(f"      InChIKey (no stereo):  {query['inchikey_no_stereo']}")
    lines.append(f"Match modes: {', '.join(results.get('match_modes', []))}")
    lines.append("")

    lines.append("Documents processed:")
    header = (
        f"  {'document':<44}{'pages':>6}{'images':>8}{'SMILES':>8}"
        f"{'Markush':>9}{'hits':>6}{'skip':>6}{'sec':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for document in results.get("documents", []):
        lines.append(
            f"  {str(document['filename'])[:43]:<44}"
            f"{document['pages']:>6}"
            f"{document['chemical_images']:>8}"
            f"{document['recognised_structures']:>8}"
            f"{document['markush_images']:>9}"
            f"{document['hits']:>6}"
            f"{document.get('failed_pages', 0):>6}"
            f"{document['runtime_seconds']:>8.1f}"
        )
    lines.append("")

    skipped = sum(d.get("failed_pages", 0) for d in results.get("documents", []))
    if skipped:
        lines.append(
            f"NOTE: {skipped} page(s) were skipped after segmentation errors; "
            "they are listed as failed_pages in the extractions file."
        )
        lines.append("")

    hits = results.get("hits", [])
    if not hits:
        lines.append("NO MATCH: the query structure was not found in these documents.")
        total_images = sum(
            document["chemical_images"] for document in results.get("documents", [])
        )
        if total_images == 0:
            lines.append(
                "  (No chemical images were detected at all - check the segmenter "
                "engine and that the documents contain structure depictions.)"
            )
        return "\n".join(lines)

    lines.append(f"FOUND {len(hits)} match(es):")
    for hit in hits:
        location = (
            f"page {hit['page']} bbox={hit['bbox']}"
            if hit["evidence"] == "image"
            else "document text"
        )
        lines.append(
            f"  * {hit['document']} - {location}\n"
            f"      match={hit['match_mode']} similarity={hit['similarity']} "
            f"class={hit['figure_class']} conf={hit['recognition_confidence']}"
        )
        if hit.get("smiles"):
            lines.append(f"      found SMILES: {hit['smiles']}")
    return "\n".join(lines)


def _extraction_dpi(extraction, page_number: int, default: int = 300) -> int:
    """DPI at which ``page_number`` was rendered during extraction."""
    if extraction is None:
        return default
    for figure in extraction.figures:
        if figure.page == page_number:
            return int(figure.dpi)
    return int(extraction.document.get("dpi", default))


def _load_font(size: int = 40):
    """A readable TrueType font if one is available, else PIL's default."""
    from PIL import ImageFont

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate_hits(
    results: Dict[str, object],
    output_dir: Path,
    dpi: int = 150,
    context: str = "page",
) -> List[Path]:
    """Render each image hit back onto its page, with the match highlighted.

    ``context='page'`` writes the whole page with every detected chemical image
    outlined and the matching one(s) in blue; ``context='crop'`` writes only the
    matching crop.
    """
    from PIL import Image, ImageDraw

    from .documents import Document

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    extractions = {
        str(extraction.document["path"]): extraction
        for extraction in results.get("extractions", [])
    }

    by_document_page: Dict[tuple, List[dict]] = {}
    for hit in results.get("hits", []):
        if hit["evidence"] != "image":
            continue
        by_document_page.setdefault((hit["document_path"], hit["page"]), []).append(hit)

    for (document_path, page_number), page_hits in sorted(by_document_page.items()):
        source = Path(document_path)
        if not source.exists():
            continue
        # Bounding boxes are in the pixel frame of the page as it was rendered
        # during extraction, so re-render at exactly that DPI before drawing.
        extraction = extractions.get(document_path)
        render_dpi = _extraction_dpi(extraction, page_number)
        document = Document(path=source, dpi=render_dpi)
        try:
            page = next(iter(document.pages([page_number])), None)
        finally:
            document.cleanup()
        if page is None:
            continue

        image = Image.fromarray(page.image).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        matched_boxes = {tuple(hit["bbox"]) for hit in page_hits}
        if extraction is not None:
            for figure in extraction.figures:
                if figure.page != page_number:
                    continue
                box = tuple(figure.bbox)
                if box in matched_boxes:
                    continue
                colour = MARKUSH_COLOR if figure.figure_class == "Markush" else OTHER_COLOR
                draw.rectangle(list(box), outline=colour + (200,), width=4)

        for hit in page_hits:
            box = list(hit["bbox"])
            draw.rectangle(box, outline=QUERY_COLOR + (255,), width=10)
            draw.rectangle(box, fill=QUERY_COLOR + (48,))
            label = f"MATCH ({hit['match_mode']})"
            draw.text(
                (box[0] + 12, max(0, box[1] - 52)),
                label,
                fill=QUERY_COLOR,
                font=_load_font(40),
            )

        if context == "crop":
            box = page_hits[0]["bbox"]
            image = image.crop(box)

        scale = dpi / float(page.dpi)
        if 0 < scale < 1:
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            )

        destination = output_dir / f"{source.stem}_page{page_number:04d}.png"
        image.save(destination)
        written.append(destination)

    return written


def draw_query(smiles: str, path: Path, size: int = 400) -> Optional[Path]:
    """Save a depiction of the query molecule for the report."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(mol, str(path), size=(size, size))
    return path


def save_all(
    results: Dict[str, object],
    output_dir: Path,
    annotate: bool = True,
    prefix: str = "structure_search",
) -> Dict[str, object]:
    """Write JSON, JSONL, CSV and (optionally) annotated pages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, object] = {
        "json": str(write_json(results, output_dir / f"{prefix}.json")),
        "csv": str(write_csv(results, output_dir / f"{prefix}_hits.csv")),
        "extractions": str(
            write_extractions(results, output_dir / f"{prefix}_extractions.jsonl")
        ),
    }
    if annotate and results.get("hits"):
        pages = annotate_hits(results, output_dir / "annotated")
        written["annotated_pages"] = [str(path) for path in pages]
    return written

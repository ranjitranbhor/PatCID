#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for structure_finder.

These cover everything that does not need downloaded model weights: matching,
document handling, the text scan, caching and reporting.  The recognition step
is exercised through a stub engine, so the orchestration is tested end to end
without MolGrapher / DECIMER present.

Run with:  python -m pytest tests/test_structure_finder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structure_finder.classification import normalise_label  # noqa: E402
from structure_finder.documents import Document, collect_documents  # noqa: E402
from structure_finder.matching import (  # noqa: E402
    MoleculeMatcher,
    canonical_smiles,
    parse_query,
)
from structure_finder.pipeline import (  # noqa: E402
    PipelineConfig,
    StructureFinder,
    scan_text_for_structures,
)
from structure_finder.recognition import BaseRecognizer, Prediction  # noqa: E402
from structure_finder.report import format_summary, save_all  # noqa: E402
from structure_finder.segmentation import HeuristicSegmenter  # noqa: E402
from structure_finder.storage import Workspace  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_exact_match_is_stereo_insensitive():
    """PatCID's convention: compare InChIKeys with stereo removed."""
    matcher = MoleculeMatcher([parse_query("C[C@H](N)C(=O)O")], modes=["exact"])
    assert matcher.match("C[C@@H](N)C(=O)O").matched  # opposite stereo centre
    assert matcher.match("CC(N)C(=O)O").matched  # no stereo at all
    assert not matcher.match(CAFFEINE).matched


def test_exact_match_is_smiles_writing_insensitive():
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    for equivalent in [
        "O=C(C)Oc1ccccc1C(=O)O",
        "OC(=O)c1ccccc1OC(C)=O",
        "CC(=O)OC1=CC=CC=C1C(O)=O",  # kekulised
    ]:
        assert matcher.match(equivalent).matched, equivalent


def test_salt_is_stripped_from_query_and_candidate():
    matcher = MoleculeMatcher([parse_query(ASPIRIN + ".[Na+]")], modes=["exact"])
    assert matcher.match(ASPIRIN).matched


def test_connectivity_mode_ignores_protonation():
    query = parse_query("CC(=O)[O-]")
    strict = MoleculeMatcher([query], modes=["exact"])
    loose = MoleculeMatcher([query], modes=["connectivity"])
    assert not strict.match("CC(=O)O").matched
    assert loose.match("CC(=O)O").matched


def test_substructure_mode_and_smarts_query():
    matcher = MoleculeMatcher([parse_query("smarts:c1ccccc1")], modes=["substructure"])
    assert matcher.match(ASPIRIN).matched
    assert not matcher.match("CCCCO").matched


def test_similarity_mode_respects_threshold():
    query = [parse_query(ASPIRIN)]
    lenient = MoleculeMatcher(query, modes=["similarity"], similarity_threshold=0.3)
    strict = MoleculeMatcher(query, modes=["similarity"], similarity_threshold=0.99)
    salicylic_acid = "OC(=O)c1ccccc1O"
    assert lenient.match(salicylic_acid).matched
    assert not strict.match(salicylic_acid).matched


def test_strictest_mode_is_reported():
    matcher = MoleculeMatcher(
        [parse_query(ASPIRIN)],
        modes=["exact", "connectivity", "substructure", "similarity"],
        similarity_threshold=0.1,
    )
    assert matcher.match(ASPIRIN).mode == "exact"


def test_invalid_smiles_candidate_does_not_crash():
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    assert not matcher.match("not-a-molecule((((").matched
    assert not matcher.match(None).matched


def test_invalid_query_raises():
    with pytest.raises(ValueError):
        parse_query("!!!not a molecule!!!")


def test_canonical_smiles_helper():
    assert canonical_smiles("C[C@H](N)C(=O)O") == canonical_smiles("CC(N)C(=O)O")


# ---------------------------------------------------------------------------
# Text scan
# ---------------------------------------------------------------------------


def test_text_scan_finds_smiles_and_inchikey():
    text = (
        "The title compound (SMILES: CC(=O)Oc1ccccc1C(=O)O) was prepared as "
        "described. Its InChIKey is BSYNRYMUTXBXSQ-UHFFFAOYSA-N and it melts "
        "at 135 degrees. Ordinary prose must not be parsed as a molecule."
    )
    smiles_list, inchikeys = scan_text_for_structures(text)
    assert "BSYNRYMUTXBXSQ-UHFFFAOYSA-N" in inchikeys
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    assert any(matcher.match(smiles).matched for smiles in smiles_list)


def test_text_scan_on_empty_text():
    assert scan_text_for_structures("") == ([], [])


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def _write_pdf_with_molecules(path: Path, smiles_per_page):
    """Render molecules with RDKit and place them on PDF pages via PyMuPDF."""
    import pymupdf
    from rdkit import Chem
    from rdkit.Chem import Draw

    pdf = pymupdf.open()
    for page_smiles in smiles_per_page:
        page = pdf.new_page(width=595, height=842)  # A4 at 72 dpi
        for index, smiles in enumerate(page_smiles):
            mol = Chem.MolFromSmiles(smiles)
            image = Draw.MolToImage(mol, size=(400, 400))
            png_path = path.parent / f"_tmp_{abs(hash(smiles)) % 10**8}_{index}.png"
            image.save(png_path)
            top = 60 + index * 330
            page.insert_image(
                pymupdf.Rect(90, top, 90 + 300, top + 300), filename=str(png_path)
            )
            png_path.unlink()
        page.insert_text((50, 800), "Example page with chemical structures.")
    pdf.save(path)
    pdf.close()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small document set: aspirin on page 1, caffeine on page 2."""
    directory = tmp_path_factory.mktemp("corpus")
    pdf_path = directory / "example_patent.pdf"
    _write_pdf_with_molecules(pdf_path, [[ASPIRIN], [CAFFEINE]])

    text_pdf = directory / "text_only.pdf"
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), f"Compound 1 has the structure {IMATINIB}")
    doc.save(text_pdf)
    doc.close()
    return directory


def test_collect_documents_filters_by_suffix(corpus, tmp_path):
    (corpus / "notes.txt").write_text("ignore me")
    documents = collect_documents([str(corpus)])
    names = sorted(document.name for document in documents)
    assert names == ["example_patent.pdf", "text_only.pdf"]


def test_pdf_pages_render_at_requested_dpi(corpus):
    document = Document(path=corpus / "example_patent.pdf", dpi=150)
    pages = list(document.pages())
    assert len(pages) == 2
    assert pages[0].number == 1
    # A4 at 150 dpi is about 1240 x 1754 px.
    assert 1200 < pages[0].width < 1280
    assert pages[0].image.shape[2] == 3
    assert "Example page" in pages[0].text


def test_page_selection(corpus):
    document = Document(path=corpus / "example_patent.pdf", dpi=100)
    pages = list(document.pages([2]))
    assert [page.number for page in pages] == [2]


def test_document_full_text_finds_written_smiles(corpus):
    document = Document(path=corpus / "text_only.pdf")
    assert "Compound 1" in document.full_text()


def test_docx_embedded_image_extraction(tmp_path):
    docx = pytest.importorskip("docx")
    from rdkit import Chem
    from rdkit.Chem import Draw

    image_path = tmp_path / "aspirin.png"
    Draw.MolToImage(Chem.MolFromSmiles(ASPIRIN), size=(400, 400)).save(image_path)

    word_path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("Scheme 1. The target compound.")
    document.add_picture(str(image_path))
    document.save(word_path)

    pages = list(Document(path=word_path, dpi=200).pages())
    assert len(pages) >= 1
    assert pages[0].image.ndim == 3


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_heuristic_segmenter_finds_a_drawn_structure(corpus):
    document = Document(path=corpus / "example_patent.pdf", dpi=150)
    page = next(iter(document.pages([1])))
    regions = HeuristicSegmenter().segment(page.image)
    assert regions, "expected at least one candidate region on the page"
    # The depiction occupies a sizeable fraction of the page.
    assert max(region.area for region in regions) > 20000


def test_heuristic_segmenter_on_blank_page():
    blank = np.full((800, 600, 3), 255, dtype=np.uint8)
    assert HeuristicSegmenter().segment(blank) == []


def test_normalise_label():
    assert normalise_label("Clean") == "Clean"
    assert normalise_label("markush structure") == "Markush"
    assert normalise_label(None) == "Trash"


# ---------------------------------------------------------------------------
# End-to-end orchestration with a stub recognizer
# ---------------------------------------------------------------------------


class StubRecognizer(BaseRecognizer):
    """Returns a fixed SMILES per call index, standing in for MolGrapher."""

    name = "stub"

    def __init__(self, smiles_sequence):
        self.smiles_sequence = list(smiles_sequence)
        self.calls = 0

    def predict(self, images):
        predictions = []
        for _ in images:
            smiles = self.smiles_sequence[self.calls % len(self.smiles_sequence)]
            self.calls += 1
            predictions.append(Prediction(smiles, 0.9, self.name))
        return predictions


def _finder(tmp_path, smiles_sequence, matcher, **config_overrides):
    workspace = Workspace(tmp_path / "workspace").prepare()
    config_overrides.setdefault("dpi", 150)
    config = PipelineConfig(
        segmenter="heuristic",
        classifier="none",
        recognizer="stub",
        **config_overrides,
    )
    finder = StructureFinder(workspace, config, matcher=matcher, verbose=False)
    finder._recognizer = StubRecognizer(smiles_sequence)
    return finder


def test_end_to_end_reports_a_hit_with_provenance(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher)
    results = finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])

    assert results["hits"], "aspirin should be found"
    hit = results["hits"][0]
    assert hit["document"] == "example_patent.pdf"
    assert hit["page"] in (1, 2)
    assert hit["match_mode"] == "exact"
    assert len(hit["bbox"]) == 4
    assert hit["bbox"][2] > hit["bbox"][0] and hit["bbox"][3] > hit["bbox"][1]
    assert hit["evidence"] == "image"


def test_end_to_end_reports_no_hit_for_absent_structure(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(IMATINIB)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher, scan_text=False)
    results = finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])
    assert results["hits"] == []
    assert "NO MATCH" in format_summary(results)


def test_text_evidence_hit(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(IMATINIB)], modes=["exact"])
    finder = _finder(tmp_path, ["CCO"], matcher)
    results = finder.search([Document(path=corpus / "text_only.pdf", dpi=150)])
    assert any(hit["evidence"] == "text" for hit in results["hits"])


def test_extraction_cache_is_reused(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    document_path = corpus / "example_patent.pdf"

    finder = _finder(tmp_path, [ASPIRIN], matcher)
    finder.search([Document(path=document_path, dpi=150)])
    first_call_count = finder._recognizer.calls
    assert first_call_count > 0

    # A second search over the same document must not re-run recognition.
    finder2 = _finder(tmp_path, [ASPIRIN], matcher)
    results = finder2.search([Document(path=document_path, dpi=150)])
    assert finder2._recognizer.calls == 0
    assert results["hits"]


def test_cache_is_invalidated_by_config_change(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    document_path = corpus / "example_patent.pdf"
    _finder(tmp_path, [ASPIRIN], matcher).search([Document(path=document_path, dpi=150)])

    other = _finder(tmp_path, [ASPIRIN], matcher, dpi=120)
    other.search([Document(path=document_path, dpi=120)])
    assert other._recognizer.calls > 0


def test_reports_are_written(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher)
    results = finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])

    output_dir = tmp_path / "out"
    written = save_all(results, output_dir, annotate=True)
    assert Path(written["json"]).exists()
    assert Path(written["csv"]).exists()
    assert Path(written["extractions"]).exists()
    assert written.get("annotated_pages")
    for page_path in written["annotated_pages"]:
        assert Path(page_path).stat().st_size > 0


def test_summary_contains_query_inchikey(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher)
    results = finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])
    summary = format_summary(results)
    assert "BSYNRYMUTXBXSQ" in summary
    assert "FOUND" in summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_page_parsing():
    from structure_finder.cli import parse_pages

    assert parse_pages("1-3,7") == [1, 2, 3, 7]
    assert parse_pages(None) is None
    assert parse_pages("2") == [2]


# ---------------------------------------------------------------------------
# numpy 2 compatibility shim (DECIMER-Segmentation)
# ---------------------------------------------------------------------------


def test_numpy2_shim_restores_visible_deprecation_warning(monkeypatch):
    """decimer_segmentation touches np.VisibleDeprecationWarning at import time.

    numpy 2.0 removed it, so without the shim merely importing the package
    raises AttributeError. The shim must restore a warning class that behaves
    the same way for `warnings.filterwarnings`.
    """
    import warnings

    import numpy as np

    import structure_finder.compat as compat

    monkeypatch.setattr(compat, "_APPLIED", False, raising=False)
    monkeypatch.delattr(np, "VisibleDeprecationWarning", raising=False)
    assert not hasattr(np, "VisibleDeprecationWarning")

    assert compat.apply_numpy2_shims() is True
    assert issubclass(np.VisibleDeprecationWarning, UserWarning)

    # The package's only use of it is this call; it must not raise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def test_numpy2_shim_is_idempotent(monkeypatch):
    import structure_finder.compat as compat

    monkeypatch.setattr(compat, "_APPLIED", False, raising=False)
    compat.apply_numpy2_shims()
    assert compat.apply_numpy2_shims() is False


def test_numpy2_shim_noop_on_numpy1(monkeypatch):
    import numpy as np

    import structure_finder.compat as compat

    monkeypatch.setattr(compat, "_APPLIED", False, raising=False)
    monkeypatch.setattr(
        np, "VisibleDeprecationWarning", UserWarning, raising=False
    )
    assert compat.apply_numpy2_shims() is False


# ---------------------------------------------------------------------------
# Input handling: a bare path must not be iterated character by character
# ---------------------------------------------------------------------------


def test_collect_documents_accepts_a_bare_string_path(corpus):
    """A string is iterable, so `for item in inputs` would yield characters.

    Passing one used to fail with "No such file or directory: c", which gives
    the user no idea what went wrong.
    """
    documents = collect_documents(str(corpus))
    assert sorted(d.name for d in documents) == [
        "example_patent.pdf",
        "text_only.pdf",
    ]


def test_collect_documents_accepts_a_bare_path_object(corpus):
    assert collect_documents(corpus)


def test_collect_documents_accepts_a_single_file_path(corpus):
    documents = collect_documents(corpus / "example_patent.pdf")
    assert [d.name for d in documents] == ["example_patent.pdf"]


def test_collect_documents_rejects_non_paths_clearly():
    with pytest.raises(TypeError, match="must be a path"):
        collect_documents([123])


def test_collect_documents_still_accepts_a_list(corpus):
    documents = collect_documents([str(corpus / "text_only.pdf")])
    assert [d.name for d in documents] == ["text_only.pdf"]


# ---------------------------------------------------------------------------
# Keras 3 compatibility (DECIMER-Segmentation weight loading)
# ---------------------------------------------------------------------------


def test_importing_the_package_sets_legacy_keras_env():
    """TensorFlow reads TF_USE_LEGACY_KERAS at ITS import time.

    Importing structure_finder must therefore set it, so that a later
    `import tensorflow` picks up Keras 2 and mrcnn's .h5 loader works.
    """
    import os
    import subprocess

    # A fresh interpreter, so an already-set variable cannot mask a regression.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; os.environ.pop('TF_USE_LEGACY_KERAS', None);"
            "import sys; sys.path.insert(0, %r);" % str(Path(__file__).resolve().parent.parent)
            + "import structure_finder;"
            "print(os.environ.get('TF_USE_LEGACY_KERAS'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "TF_USE_LEGACY_KERAS": ""},
    )
    assert result.stdout.strip().endswith("1"), result.stdout + result.stderr


def test_enable_legacy_keras_returns_false_without_tensorflow(monkeypatch):
    """No TensorFlow installed is not an error - the engine just is not usable."""
    import builtins

    from structure_finder.compat import enable_legacy_keras

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("no tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert enable_legacy_keras() is False


def test_legacy_keras_error_names_the_remedy(monkeypatch):
    """A Keras 3 session that cannot be fixed must say to restart, not just fail."""
    import builtins
    import types

    from structure_finder.compat import LegacyKerasUnavailable, enable_legacy_keras

    fake_tf = types.SimpleNamespace(keras=types.SimpleNamespace(__name__="keras.api"))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            return fake_tf
        if name == "tf_keras":
            return types.SimpleNamespace()  # installed, but not active
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(LegacyKerasUnavailable, match="[Rr]estart"):
        enable_legacy_keras()


# ---------------------------------------------------------------------------
# Per-page segmentation failures must not lose the whole document
# ---------------------------------------------------------------------------


class _FlakySegmenter(HeuristicSegmenter):
    """Raises on one page, behaves normally on the rest."""

    name = "flaky"

    def __init__(self, fail_on_call: int = 1):
        super().__init__()
        self.calls = 0
        self.fail_on_call = fail_on_call

    def segment(self, page_image):
        self.calls += 1
        if self.calls == self.fail_on_call:
            # The shape DECIMER-Segmentation actually fails with: an empty mask
            # reaching np.where(rows)[0][[0, -1]] in get_masked_image_optimized.
            raise RuntimeError("index 0 is out of bounds for axis 0 with size 0")
        return super().segment(page_image)


def test_one_bad_page_does_not_lose_the_document(corpus, tmp_path):
    matcher = MoleculeMatcher([parse_query(CAFFEINE)], modes=["exact"])
    finder = _finder(tmp_path, [CAFFEINE], matcher, scan_text=False)
    finder._segmenter = _FlakySegmenter(fail_on_call=1)   # page 1 blows up

    results = finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])

    extraction = results["extractions"][0]
    assert extraction.failed_pages == [1]
    assert extraction.page_count == 2
    # Page 2 still produced a hit.
    assert results["hits"], "page 2 should still have been processed"
    assert results["documents"][0]["failed_pages"] == 1
    assert "skipped after segmentation errors" in format_summary(results)


def test_every_page_failing_raises_rather_than_reporting_nothing(corpus, tmp_path):
    """All pages failing is a broken setup - do not report a quiet 'no match'."""
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher, scan_text=False)

    class _AlwaysFails(HeuristicSegmenter):
        def segment(self, page_image):
            raise RuntimeError("index 0 is out of bounds for axis 0 with size 0")

    finder._segmenter = _AlwaysFails()
    with pytest.raises(RuntimeError, match="all 2 page"):
        finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])


def test_engine_configuration_errors_are_not_swallowed(corpus, tmp_path):
    """A missing engine must stop the run, not look like 60 bad pages."""
    matcher = MoleculeMatcher([parse_query(ASPIRIN)], modes=["exact"])
    finder = _finder(tmp_path, [ASPIRIN], matcher, scan_text=False)

    class _NotInstalled(HeuristicSegmenter):
        def segment(self, page_image):
            raise ImportError("DECIMER-Segmentation is not installed.")

    finder._segmenter = _NotInstalled()
    with pytest.raises(ImportError, match="not installed"):
        finder.search([Document(path=corpus / "example_patent.pdf", dpi=150)])

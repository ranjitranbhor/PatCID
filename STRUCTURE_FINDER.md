# structure_finder — find a chemical structure inside your own documents

`structure_finder` answers one question:

> **Given a SMILES string and a pile of PDFs / Word documents, where — if
> anywhere — is that structure drawn?**

It runs the [PatCID](https://www.nature.com/articles/s41467-024-50779-y)
document-ingestion pipeline over documents *you* supply, instead of over the
80.7M patent images IBM pre-processed, and then matches every structure it
recovers against your query with RDKit.

```
                  ┌──────────────────────────────────────────────────────┐
 your PDF / DOCX  │  1. SEGMENT      DECIMER-Segmentation (Mask R-CNN)   │
        │         │     ↓            → where are the chemical images?    │
        │         │  2. CLASSIFY     MolClassifier (Mask R-CNN)          │
        └────────▶│     ↓            → Clean / Markush / Trash           │
                  │  3. RECOGNISE    MolGrapher (keypoints + GNN)        │
                  │     ↓            or DECIMER (EfficientNetV2 + Tfmr)  │
                  │                  → SMILES                            │
                  └─────────────────────┬────────────────────────────────┘
                                        ▼
                          4. MATCH  RDKit vs. your query SMILES
                                        ▼
                          hits: document, page, bounding box
```

Steps 1–3 are exactly PatCID's Figure 2 pipeline. Step 4 is what turns it into
a search tool. The reported precision of the full PatCID chain is 54.5% on
D2C-RND and 41.3% on D2C-UNI, with recall 46.0% / 44.5% (PatCID, Table 3) —
read the [Accuracy](#accuracy-what-to-expect) section before trusting a
"not found" result.

---

## Install

Minimum (matching + text scan + heuristic segmenter, no model weights):

```bash
pip install rdkit pymupdf numpy pillow opencv-python python-docx
```

Full pipeline — pick a recognition engine (both work; MolGrapher is ~2× faster
on CPU, DECIMER scored slightly higher on D2C-RND):

```bash
# segmentation (PatCID's choice)
pip install decimer-segmentation

# recognition, option A: MolGrapher
git clone https://github.com/DS4SD/MolGrapher && cd MolGrapher
pip install -e ".[cpu]"          # or ".[gpu]"
bash install_paddleocr.sh

# recognition, option B: DECIMER Image Transformer
pip install decimer

# classification (Clean / Markush / Trash)
git clone https://github.com/DS4SD/MolClassifier
pip install torch torchvision pycocotools albumentations imantics more-itertools
export PYTHONPATH="$PWD/MolClassifier:$PWD/MolClassifier/mol_classifier:$PYTHONPATH"
```

> **MolClassifier PYTHONPATH gotcha.** `mol_classifier/classifier.py` imports
> `albumentations_transforms` as a *top-level* module even though the file lives
> inside the package, so both the repo root **and** `MolClassifier/mol_classifier`
> must be on `PYTHONPATH`. The command above does that.

Word documents keep their page numbers only when LibreOffice is installed
(`apt-get install libreoffice-writer`); without it, `structure_finder` falls
back to reading the images embedded in the `.docx` archive and numbers them
sequentially.

### Google Colab

Open [`notebooks/find_structure_in_documents.ipynb`](./notebooks/find_structure_in_documents.ipynb)
in Colab — it installs everything, mounts Drive, takes uploads and runs the
search. Two runtime facts drive the setup there:

- **Use the DECIMER recognition engine on Colab.** MolGrapher's `setup.py`
  constructs torch wheel URLs pinned to CPython 3.11, so `pip install -e ".[cpu]"`
  fails on Colab's current Python. DECIMER is pip-installable on any runtime and
  actually scored slightly higher than MolGrapher on D2C-RND (67.2% vs 63.0%);
  you only give up some CPU speed. If you do get a 3.11 runtime, the notebook
  installs MolGrapher too.
- **Put the workspace on Drive** (`use_drive=True`). The ~1.5 GB of weights and,
  more importantly, the per-document extraction cache then survive runtime
  restarts.

---

## Use it

### Command line

```bash
python -m structure_finder \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    patents/US9096558.pdf report.docx ./more_documents/
```

```
Query structure(s):
  - CC(=O)Oc1ccccc1C(=O)O
      InChIKey (no stereo):  BSYNRYMUTXBXSQ-UHFFFAOYSA-N
Match modes: exact, connectivity, tautomer

Documents processed:
  document                                     pages  images  SMILES  Markush  hits     sec
  -----------------------------------------------------------------------------------------
  US9096558.pdf                                   58     214      197       11     3    412.6
  report.docx                                      6      12       12        0     0     21.4

FOUND 3 match(es):
  * US9096558.pdf - page 21 bbox=[612, 1840, 1268, 2390]
      match=exact similarity=1.0 class=Clean conf=0.98
      found SMILES: CC(=O)Oc1ccccc1C(=O)O
  ...
```

### Python

```python
from structure_finder import find_structure

results = find_structure(
    documents=["patents/", "report.docx"],
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    use_drive=True,            # Colab: models + cache live on Google Drive
)

for hit in results["hits"]:
    print(hit["document"], "page", hit["page"], hit["match_mode"], hit["bbox"])
```

### Output

Every run writes to `<output-dir>` (default `<workspace>/outputs/<timestamp>/`):

| file | contents |
|---|---|
| `structure_search.json` | queries, per-document statistics, every hit |
| `structure_search_hits.csv` | one row per hit, for a spreadsheet |
| `structure_search_extractions.jsonl` | every chemical image found, with page, bbox, class and SMILES — the same shape as PatCID's `patcid_patent_to_molecules_*.jsonl` |
| `annotated/<doc>_page####.png` | the page, with the match outlined in blue, Markush structures in red and other detections in grey |

The annotated pages are PatCID's provenance feature: you get the exact location
in the document, not just "this document mentions the molecule".

---

## Matching modes

`--match-mode` is repeatable; the default is `exact connectivity tautomer`. The
strictest mode that fires is what gets reported.

| mode | matches when the recognised structure ... | use it for |
|---|---|---|
| `exact` | has the same InChIKey **ignoring stereo-chemistry** | the default; this is PatCID's own evaluation criterion |
| `connectivity` | has the same InChIKey skeleton (first block) | tolerating salt/charge/isotope/stereo differences |
| `tautomer` | has the same RDKit canonical tautomer | keto–enol and similar redraws |
| `substructure` | contains your query (SMILES **or** `smarts:` pattern) | scaffold and Markush-style searches |
| `similarity` | has Tanimoto ≥ `--similarity-threshold` on Morgan(2, 2048) | close analogues; recovering from small OCSR errors |

Stereo-chemistry is stripped on both sides throughout, because MolGrapher's
default node classifier (`gc_no_stereo_model`) and DECIMER do not reliably
recover wedge/hash bonds — comparing stereo would manufacture false negatives.
Salts and solvents are stripped to the largest fragment.

Substructure and similarity searches:

```bash
python -m structure_finder --smiles "smarts:c1ccc2[nH]ccc2c1" docs/   # indole scaffold
python -m structure_finder --smiles "$TARGET" --match-mode similarity \
       --similarity-threshold 0.8 docs/
```

---

## Google Drive for models and cache

Everything heavy lives in one *workspace* directory:

```
<workspace>/
  models/        MolClassifier checkpoint
  models/hf/     HF_HOME       — MolGrapher weights land here
  models/pystow/ PYSTOW_HOME   — DECIMER weights land here
  models/torch/  TORCH_HOME
  cache/         per-document extraction results
  outputs/       reports and annotated pages
```

`--drive` mounts Google Drive and uses `MyDrive/structure_finder`, so on Colab
the ~2 GB of weights and every document you have already processed survive a
runtime restart:

```bash
python -m structure_finder.setup_models --drive --engine all   # once
python -m structure_finder --drive --smiles "$Q" /content/uploads/
```

Outside Colab, point `--workspace` (or `$STRUCTURE_FINDER_HOME`) at any
directory — including a locally synced Drive folder.

### The extraction cache is the important part

Steps 1–3 are slow (a 60-page patent takes minutes on CPU) but they do **not**
depend on the query. Results are cached under `<workspace>/cache/`, keyed by the
document's SHA-256 **and** the engine configuration. So the first search over a
document set is expensive and every subsequent search — a different molecule, a
different match mode — returns in milliseconds. Ingest once, query many times;
that is how PatCID itself is built. `--no-cache` forces re-extraction.

---

## Engine choices

| flag | options | notes |
|---|---|---|
| `--segmenter` | `decimer` (default), `molclassifier`, `heuristic` | `decimer` is DECIMER-Segmentation: 88.0% precision / 86.3% recall on D2C-RND. `molclassifier` detects *and* classifies in one pass (fastest CPU setup). `heuristic` is a connected-component fallback needing no weights — fine for smoke tests, poor on dense patent pages. |
| `--classifier` | `molclassifier` (default), `none` | Filters segmentation errors and separates Markush structures. `none` sends every region to recognition. |
| `--recognizer` | `molgrapher` (default), `decimer`, `ensemble` | `ensemble` runs both and keeps the higher-confidence reading, preferring one that matches the query. Roughly doubles runtime and meaningfully raises recall. |

Other useful flags: `--pages 1-20,35`, `--dpi 400` (small or low-quality
depictions), `--min-confidence 0.7`, `--recognize-markush`, `--device cuda`,
`--no-text-scan`.

---

## The text scan

Alongside the image pipeline, `structure_finder` scans the document *text* for
molecules written out as SMILES, InChI or InChIKey strings, and matches those
too. Such hits are reported with `evidence="text"` and `page=0`. Patents and
papers often print the structure next to the drawing, so this is free recall on
top of the image pipeline — and it is the only thing that works on a document
whose depictions the segmenter misses. Disable with `--no-text-scan`.

---

## Accuracy: what to expect

From the PatCID paper (Tables 3), on its D2C-RND / D2C-UNI benchmarks:

| stage | D2C-RND | D2C-UNI |
|---|---|---|
| DECIMER-Segmentation | 88.0% P / 86.3% R | 81.1% P / 80.8% R |
| MolClassifier | 93.4% P / 84.6% R | 82.9% P / 89.5% R |
| MolGrapher recognition | 63.0% | 57.1% |
| DECIMER recognition | 67.2% | 64.6% |
| **full pipeline** | **54.5% P / 46.0% R** | **41.3% P / 44.5% R** |

Practical consequences:

- **A hit is strong evidence; a miss is weak evidence.** Roughly half of the
  drawn structures are recovered exactly. If the answer matters, re-run with
  `--recognizer ensemble --match-mode exact --match-mode connectivity
  --match-mode similarity --similarity-threshold 0.85` and review the near
  misses in `structure_search_extractions.jsonl` by hand.
- **Recent US-office documents process best.** Older and Asian-Pacific-office
  patents use less standard drawing styles and score noticeably lower (that is
  the whole point of the D2C-UNI benchmark).
- **Markush structures cannot match a concrete SMILES.** A generic structure
  with R groups has no single molecule behind it. They are detected and counted
  in the `Markush` column so you know to look; `--recognize-markush` will
  produce a SMILES for them, but treat it as indicative only. For scaffold
  questions use `--match-mode substructure` with a `smarts:` query.
- **Confidence is a useful filter.** MolGrapher's `conf` and DECIMER's mean
  token confidence are in every record; `--min-confidence 0.8` trades recall for
  precision.

---

## Tests

```bash
python -m pytest tests/test_structure_finder.py -v
```

28 tests covering matching semantics, PDF/DOCX handling, the text scan,
segmentation, caching, reporting and annotation. They use a stub recognizer, so
they run without any downloaded model weights.

---

## Credits

- **PatCID** — Morin, Weber, Meijer, Yu & Staar, *Nature Communications* **15**,
  6532 (2024). The ingestion pipeline this tool re-implements over user documents.
- **MolGrapher** — Morin et al., *ICCV* 2023, pp. 19552–19561.
- **DECIMER Image Transformer / DECIMER-Segmentation** — Rajan, Brinkhaus,
  Sorokina, Zielesny & Steinbeck.
- **MolClassifier** — Weber & Morin, IBM Research (published with PatCID).

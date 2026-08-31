#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chemical matching between a query structure and recognised SMILES.

The PatCID paper counts a molecule as retrieved when the query and the
annotation share an InChIKey *ignoring stereo-chemistry* - the recognition
models used here (MolGrapher's default variant, DECIMER) do not reliably
predict stereo descriptors, so comparing them would only produce false
negatives.  This module implements that convention as the ``exact`` mode and
adds four progressively looser modes:

===============  =========================================================
mode             a recognised structure matches the query when ...
===============  =========================================================
``exact``        ... its stereo-stripped InChIKey equals the query's
``connectivity`` ... its InChIKey *skeleton* (first block) equals the
                 query's, i.e. same heavy-atom connectivity but possibly
                 different protonation / isotopes / stereo
``tautomer``     ... its RDKit canonical tautomer equals the query's
``substructure`` ... it contains the query as a substructure (the query may
                 be given as SMARTS)
``similarity``   ... its Morgan fingerprint Tanimoto to the query is above
                 a threshold (default 0.85)
===============  =========================================================

Modes are ordered from strictest to loosest; :class:`MoleculeMatcher` reports
the strictest mode that fires, so a hit is never over-claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

MATCH_MODES: Sequence[str] = (
    "exact",
    "connectivity",
    "tautomer",
    "substructure",
    "similarity",
)

#: Strictest first - used to rank matches.
_MODE_RANK = {mode: index for index, mode in enumerate(MATCH_MODES)}

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonical_smiles(smiles: str, remove_stereo: bool = True) -> Optional[str]:
    """Canonicalise a SMILES string, optionally dropping stereo-chemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Keep only the largest covalent fragment (drops salts and solvents)."""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) <= 1:
        return mol
    return max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())


def inchikey(mol: Chem.Mol, remove_stereo: bool = True) -> Optional[str]:
    """InChIKey of ``mol``; ``None`` if InChI generation fails."""
    work = Chem.Mol(mol)
    if remove_stereo:
        Chem.RemoveStereochemistry(work)
    try:
        key = Chem.MolToInchiKey(work)
    except Exception:  # pragma: no cover - RDKit InChI failures are rare
        return None
    return key or None


def skeleton_key(mol: Chem.Mol) -> Optional[str]:
    """First InChIKey block: the heavy-atom connectivity hash."""
    key = inchikey(mol, remove_stereo=True)
    return key.split("-")[0] if key else None


@dataclass
class ParsedQuery:
    """A query structure, pre-processed once and reused for every candidate."""

    raw: str
    mol: Chem.Mol
    canonical: str
    inchikey_no_stereo: Optional[str]
    skeleton: Optional[str]
    tautomer_canonical: Optional[str]
    fingerprint: object
    pattern: Chem.Mol
    is_smarts: bool
    label: str

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "query": self.raw,
            "label": self.label,
            "canonical_smiles": self.canonical,
            "inchikey_no_stereo": self.inchikey_no_stereo,
            "skeleton": self.skeleton,
            "is_smarts": self.is_smarts,
        }


@dataclass
class MatchResult:
    """Outcome of comparing one recognised structure with one query."""

    matched: bool
    mode: Optional[str]
    similarity: float
    query_label: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "matched": self.matched,
            "match_mode": self.mode,
            "similarity": round(self.similarity, 4),
            "query_label": self.query_label,
        }


def _canonical_tautomer(mol: Chem.Mol) -> Optional[str]:
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        canonical = enumerator.Canonicalize(mol)
    except Exception:
        return None
    if canonical is None:
        return None
    Chem.RemoveStereochemistry(canonical)
    return Chem.MolToSmiles(canonical)


def parse_query(
    query: str,
    label: Optional[str] = None,
    strip_salts: bool = True,
) -> ParsedQuery:
    """Parse a query given as SMILES or SMARTS.

    A string is treated as SMARTS when it cannot be parsed as SMILES, or when
    it is prefixed with ``smarts:``.
    """
    raw = query.strip()
    is_smarts = False
    if raw.lower().startswith("smarts:"):
        raw_body = raw.split(":", 1)[1].strip()
        is_smarts = True
        mol = Chem.MolFromSmarts(raw_body)
        if mol is None:
            raise ValueError(f"Invalid SMARTS query: {raw_body!r}")
        raw = raw_body
    else:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            mol = Chem.MolFromSmarts(raw)
            is_smarts = mol is not None
        if mol is None:
            raise ValueError(f"Invalid SMILES/SMARTS query: {raw!r}")

    pattern = mol
    if is_smarts:
        # A SMARTS pattern supports substructure search only; the identity-based
        # modes need a concrete molecule, which we cannot derive from SMARTS.
        return ParsedQuery(
            raw=raw,
            mol=mol,
            canonical=raw,
            inchikey_no_stereo=None,
            skeleton=None,
            tautomer_canonical=None,
            fingerprint=None,
            pattern=pattern,
            is_smarts=True,
            label=label or raw,
        )

    if strip_salts:
        mol = largest_fragment(mol)
    Chem.SanitizeMol(mol)
    stereo_free = Chem.Mol(mol)
    Chem.RemoveStereochemistry(stereo_free)

    return ParsedQuery(
        raw=raw,
        mol=mol,
        canonical=Chem.MolToSmiles(stereo_free),
        inchikey_no_stereo=inchikey(mol),
        skeleton=skeleton_key(mol),
        tautomer_canonical=_canonical_tautomer(mol),
        fingerprint=_MORGAN_GENERATOR.GetFingerprint(stereo_free),
        pattern=stereo_free,
        is_smarts=False,
        label=label or Chem.MolToSmiles(stereo_free),
    )


class MoleculeMatcher:
    """Compare recognised SMILES against one or more parsed queries."""

    def __init__(
        self,
        queries: Sequence[ParsedQuery],
        modes: Sequence[str] = ("exact", "connectivity", "tautomer"),
        similarity_threshold: float = 0.85,
        strip_salts: bool = True,
    ) -> None:
        if not queries:
            raise ValueError("At least one query structure is required.")
        unknown = sorted(set(modes) - set(MATCH_MODES))
        if unknown:
            raise ValueError(
                f"Unknown match mode(s): {unknown}. Choose from {list(MATCH_MODES)}."
            )
        self.queries = list(queries)
        self.modes = sorted(set(modes), key=lambda mode: _MODE_RANK[mode])
        self.similarity_threshold = similarity_threshold
        self.strip_salts = strip_salts
        self._cache: Dict[str, List[MatchResult]] = {}

    # -- candidate preparation -------------------------------------------------

    def _prepare(self, smiles: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if self.strip_salts:
            mol = largest_fragment(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        Chem.RemoveStereochemistry(mol)
        return mol

    # -- matching --------------------------------------------------------------

    def _match_one(self, mol: Chem.Mol, query: ParsedQuery) -> MatchResult:
        similarity = 0.0
        if query.fingerprint is not None:
            similarity = TanimotoSimilarity(
                query.fingerprint, _MORGAN_GENERATOR.GetFingerprint(mol)
            )

        for mode in self.modes:
            if mode == "exact" and query.inchikey_no_stereo:
                if inchikey(mol) == query.inchikey_no_stereo:
                    return MatchResult(True, "exact", similarity or 1.0, query.label)
            elif mode == "connectivity" and query.skeleton:
                if skeleton_key(mol) == query.skeleton:
                    return MatchResult(True, "connectivity", similarity, query.label)
            elif mode == "tautomer" and query.tautomer_canonical:
                if _canonical_tautomer(mol) == query.tautomer_canonical:
                    return MatchResult(True, "tautomer", similarity, query.label)
            elif mode == "substructure":
                if mol.HasSubstructMatch(query.pattern):
                    return MatchResult(True, "substructure", similarity, query.label)
            elif mode == "similarity":
                if similarity >= self.similarity_threshold:
                    return MatchResult(True, "similarity", similarity, query.label)

        return MatchResult(False, None, similarity, query.label)

    def match(self, smiles: Optional[str]) -> MatchResult:
        """Best (strictest) match of ``smiles`` across all queries."""
        if not smiles:
            return MatchResult(False, None, 0.0, self.queries[0].label)
        if smiles in self._cache:
            results = self._cache[smiles]
        else:
            mol = self._prepare(smiles)
            if mol is None:
                results = [
                    MatchResult(False, None, 0.0, query.label) for query in self.queries
                ]
            else:
                results = [self._match_one(mol, query) for query in self.queries]
            self._cache[smiles] = results

        def sort_key(result: MatchResult):
            rank = _MODE_RANK.get(result.mode, len(MATCH_MODES)) if result.mode else len(MATCH_MODES)
            return (not result.matched, rank, -result.similarity)

        return sorted(results, key=sort_key)[0]

    def all_matches(self, smiles: Optional[str]) -> List[MatchResult]:
        """Per-query results for ``smiles`` (useful for multi-query reports)."""
        self.match(smiles)
        return self._cache.get(smiles or "", [])

"""
Curated, conservative known-regulator lists for the datasets in raw_data/.

Lists are intentionally small. They contain only genes whose role in the
relevant developmental program is uncontroversial in the primary literature.
We do *not* extend the lists with predictions, downstream targets, or
weakly-associated factors; the goal is overlap-based annotation, not
discovery.

For datasets where gene symbols are absent (e.g. UC epi shipped without a
gene-name file), no list is exposed and overlap analysis is skipped.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

# C. elegans embryogenesis (Packer-style atlas).
# Gene symbols match the conventional WormBase casing used in the Packer
# matrices. We avoid invented orthologue mappings.
CELEGANS_EMBRYOGENESIS = {
    # maternal / early embryo
    "skn-1", "pal-1", "mex-3", "pie-1", "pos-1",
    # endoderm / pharynx / gut
    "end-1", "end-3", "elt-2", "elt-7", "pha-4", "med-1", "med-2",
    # mesoderm / muscle
    "hlh-1", "unc-120", "myo-3", "hnd-1",
    # neural / pan-neural
    "unc-86", "lin-32", "ngn-1", "hlh-2",
    # epidermis
    "elt-1", "elt-3", "lin-26",
}

# Human / mouse pluripotency + early differentiation. Markers usable for
# both Klein mESC (LIF withdrawal) and the hESC RDS dataset.
PLURIPOTENCY_DIFFERENTIATION = {
    # core pluripotency
    "POU5F1", "OCT4", "SOX2", "NANOG", "KLF4", "KLF2", "ESRRB", "TFCP2L1",
    "DNMT3B", "DNMT3A", "ZFP42", "REX1", "DPPA3", "DPPA5",
    # mouse-case variants (Klein dataset uses MGI symbols)
    "Pou5f1", "Sox2", "Nanog", "Klf4", "Klf2", "Esrrb", "Tfcp2l1",
    "Dnmt3b", "Dnmt3a", "Zfp42", "Dppa3", "Dppa5", "Tbx3", "Tbx5",
    "Tdgf1", "Lefty1", "Lefty2", "Fgf4",
    # endoderm
    "GATA6", "GATA4", "SOX17", "FOXA1", "FOXA2",
    "Gata6", "Gata4", "Sox17", "Foxa1", "Foxa2",
    # mesoderm / primitive streak
    "T", "TBXT", "MIXL1", "EOMES", "MESP1", "MESP2",
    "Tbxt", "Mixl1", "Eomes", "Mesp1", "Mesp2",
    # ectoderm / neural
    "PAX6", "SOX1", "NES", "NESTIN",
    "Pax6", "Sox1", "Nes",
    # trophoblast
    "CDX2", "GATA3", "ELF5", "TFAP2C", "KRT7", "HAND1",
    "Cdx2", "Gata3", "Elf5", "Tfap2c", "Krt7", "Hand1",
}

# Intestinal epithelium (UC epi dataset uses MTX with no gene symbols, but
# we keep the list available for any future symbol-annotated UC matrix).
INTESTINAL_EPITHELIUM = {
    # stem / TA
    "LGR5", "OLFM4", "ASCL2", "BMI1", "TERT", "HOPX", "LRIG1",
    # absorptive lineage / enterocyte
    "KRT20", "VIL1", "FABP1", "FABP2", "ALPI", "APOA1", "APOA4", "RBP2",
    # secretory / goblet
    "MUC2", "MUC5AC", "TFF3", "ATOH1", "SPDEF", "GFI1",
    # paneth / antimicrobial
    "DEFA5", "DEFA6", "DEFA1", "DEFA1B", "LYZ", "REG3A", "REG3G", "PLA2G2A",
    # enteroendocrine
    "CHGA", "CHGB", "NEUROG3", "NEUROD1",
    # tuft
    "POU2F3", "DCLK1", "TRPM5",
    # M cell
    "GP2", "SPIB",
}


REGULATOR_SETS = {
    "celegans": CELEGANS_EMBRYOGENESIS,
    "klein": PLURIPOTENCY_DIFFERENTIATION,
    "hESC": PLURIPOTENCY_DIFFERENTIATION,
    "uc_epi": INTESTINAL_EPITHELIUM,
}


def get_regulator_set(dataset: str) -> set:
    return set(REGULATOR_SETS.get(dataset, set()))


def annotate_overlap(genes: Iterable[str], dataset: str):
    """Return the subset of ``genes`` that match the dataset's known set."""
    s = get_regulator_set(dataset)
    if not s:
        return []
    keys = set(genes)
    # case-insensitive match
    s_lower = {x.lower(): x for x in s}
    out = []
    for g in keys:
        if g.lower() in s_lower:
            out.append(g)
    return sorted(out)


__all__ = [
    "CELEGANS_EMBRYOGENESIS",
    "PLURIPOTENCY_DIFFERENTIATION",
    "INTESTINAL_EPITHELIUM",
    "REGULATOR_SETS",
    "get_regulator_set",
    "annotate_overlap",
]

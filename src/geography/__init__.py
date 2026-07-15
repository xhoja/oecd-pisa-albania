"""Within-Albania geographic equity analysis.

Decodes the PISA sampling STRATUM into geographic cells (region band x
urbanicity x sector, see :mod:`src.causal.region`) and reports the weighted
low-proficiency rate for each cell across cycles, with PISA design-based
standard errors (BRR + Rubin over plausible values). Unlike the earthquake
difference-in-differences (:mod:`src.causal.did`), which asks a *causal*
question about one band, this module is descriptive: it maps *where inside
Albania* the low-proficiency crisis concentrates and how each cell moved
through the 2015 -> 2018 -> 2022 trajectory.
"""
from src.geography.equity import (
    atrisk_by_groups,
    gap_brr_rubin,
    region_urbanicity_matrix,
)

__all__ = [
    "atrisk_by_groups",
    "gap_brr_rubin",
    "region_urbanicity_matrix",
]

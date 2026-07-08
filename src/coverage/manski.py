"""
Partial-identification (Manski) bounds on the at-risk rate under incomplete
coverage (target #5).

PISA samples only enrolled 15-year-olds who meet grade and exclusion rules. For
Albania 2022 the sampled population represents about **79%** of all 15-year-olds
(Coverage Index 3 ~ 0.79; ~28 400 of ~36 000 - OECD PISA 2022 Results, Country
Note). The reported at-risk rate is therefore a rate *among the covered*, and the
~21% uncovered - disproportionately out-of-school and disadvantaged youth - are
unobserved. Rather than assume they look like the sample, we report the range of
population at-risk rates consistent with the data (Manski 1989, 2003).

Let ``p`` = observed at-risk rate among the covered, ``c`` = coverage share, and
``u`` = the (unknown) at-risk rate among the uncovered. The population rate is
    P = c * p + (1 - c) * u,     u in [0, 1].

* **Worst-case (no assumptions):** u in [0,1] gives P in [c*p, c*p + (1-c)] -
  width exactly ``1 - c``.
* **Monotone (uncovered no better off):** out-of-school 15-year-olds are, if
  anything, *more* at risk than enrolled ones, so u >= p tightens the lower
  bound to ``p`` itself: P in [p, c*p + (1-c)].
* **Scenario (reweighting):** plug a data-anchored ``u`` - e.g. the at-risk rate
  of the poorest covered SES quintile as a stand-in for the uncovered - to get a
  point value inside the bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bounds:
    lower: float
    upper: float
    assumption: str

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> dict:
        return {"lower": round(self.lower, 4), "upper": round(self.upper, 4),
                "width": round(self.width, 4), "assumption": self.assumption}


def _check(p_obs: float, coverage: float) -> None:
    if not 0.0 <= p_obs <= 1.0:
        raise ValueError(f"p_obs must be in [0,1], got {p_obs}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0,1], got {coverage}")


def worst_case_bounds(p_obs: float, coverage: float) -> Bounds:
    """No-assumption Manski bounds: uncovered rate ranges over [0,1]."""
    _check(p_obs, coverage)
    lo = coverage * p_obs
    hi = coverage * p_obs + (1.0 - coverage)
    return Bounds(lo, hi, "worst-case (u in [0,1])")


def monotone_bounds(p_obs: float, coverage: float, uncovered_worse: bool = True) -> Bounds:
    """Bounds under a monotone assumption on the uncovered group.

    ``uncovered_worse=True`` assumes the uncovered are at least as at-risk as the
    covered (u >= p_obs), tightening the LOWER bound to ``p_obs``. Setting it
    False assumes the reverse (u <= p_obs), tightening the UPPER bound.
    """
    _check(p_obs, coverage)
    wc = worst_case_bounds(p_obs, coverage)
    if uncovered_worse:
        return Bounds(p_obs, wc.upper, "monotone: uncovered >= covered (u >= p)")
    return Bounds(wc.lower, p_obs, "monotone: uncovered <= covered (u <= p)")


def scenario_rate(p_obs: float, coverage: float, uncovered_rate: float) -> float:
    """Population at-risk rate for an explicit assumed uncovered rate ``u``.

    Reweighting interpretation: mix the covered rate with a stand-in rate for the
    uncovered (e.g. the poorest quintile's rate), weighting by coverage share.
    """
    _check(p_obs, coverage)
    if not 0.0 <= uncovered_rate <= 1.0:
        raise ValueError("uncovered_rate must be in [0,1]")
    return coverage * p_obs + (1.0 - coverage) * uncovered_rate


def coverage_sensitivity(p_obs: float, coverages: np.ndarray) -> list[dict]:
    """Worst-case and monotone bounds across a range of coverage assumptions."""
    rows = []
    for c in coverages:
        wc = worst_case_bounds(p_obs, float(c))
        mono = monotone_bounds(p_obs, float(c), uncovered_worse=True)
        rows.append({
            "coverage": round(float(c), 3),
            "wc_lower": round(wc.lower, 4), "wc_upper": round(wc.upper, 4),
            "wc_width": round(wc.width, 4),
            "mono_lower": round(mono.lower, 4), "mono_upper": round(mono.upper, 4),
        })
    return rows

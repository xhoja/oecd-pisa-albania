"""
Scenario-based forecast of Albania's weighted low-proficiency (math < L2) rate for
the next PISA cycle. After COVID delayed PISA 2021 to 2022, the cycle moved to a
4-year cadence, so the next assessment is **2026** (not 2025).

Why scenarios, not a point forecast: there are only **five** cycles and 2022 is a
structural break (the COVID spike), so any single extrapolation is indefensible.
Instead we propagate each cycle's *design-based* uncertainty (the BRR + plausible-
value SE from ``src.data.weights.pv_statistic_brr``) through three transparent
narratives and report predictive intervals, plus a naive all-cycles linear fit as a
reference the reader can see is unreliable.

Scenarios
---------
- **persistence** — the 2022 crisis level holds (2026 ≈ 2022), with an added drift
  term for horizon uncertainty.
- **recovery** — the pre-COVID improving trend (2009–2018 WLS line) resumes, i.e.
  2022 was a transient shock that fully reverses.
- **partial** — a 50/50 blend: the shock half-reverses.
- **naive_linear** *(reference only)* — a weighted line through all five cycles,
  which the 2022 break makes meaningless; shown to demonstrate why we don't use it.

All rates are clipped to [0, 1]. The persistence drift SD is an explicit judgment
parameter, not estimated from data — documented so it can be challenged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRE_COVID = (2009, 2012, 2015, 2018)


def wls_trend(x: np.ndarray, y: np.ndarray, se: np.ndarray, degree: int = 1) -> np.ndarray:
    """Weighted least-squares polynomial fit (weights = 1/SE²). Returns coefficients
    in ``np.vander`` (highest-degree-first) order."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.square(np.asarray(se, float))
    V = np.vander(x, degree + 1)
    WV = V * w[:, None]
    beta = np.linalg.solve(V.T @ WV, WV.T @ y)
    return beta


def _predict(beta: np.ndarray, x: float | np.ndarray) -> np.ndarray:
    degree = len(beta) - 1
    return np.vander(np.atleast_1d(np.asarray(x, float)), degree + 1) @ beta


@dataclass
class ScenarioForecast:
    name: str
    median: float
    lo: float          # 5th percentile
    hi: float          # 95th percentile
    samples: np.ndarray = field(repr=False)


def monte_carlo_forecast(
    cycles: list[int] | np.ndarray,
    rates: list[float] | np.ndarray,
    ses: list[float] | np.ndarray,
    target_year: int = 2026,
    n_sims: int = 20000,
    persistence_drift_sd: float = 0.05,
    seed: int = 42,
) -> dict[str, ScenarioForecast]:
    """Monte-Carlo scenario forecast. Each simulation draws every cycle's true rate
    from ``N(rate, se)`` (design-based sampling distribution), then applies each
    scenario's rule; percentiles over simulations give the predictive interval.

    ``persistence_drift_sd`` is the SD of the extra drift added to the persistence
    scenario over a 4-year horizon (scaled by ``sqrt(horizon/4)``)."""
    cycles = np.asarray(cycles, float)
    rates = np.asarray(rates, float)
    ses = np.asarray(ses, float)
    rng = np.random.default_rng(seed)

    draws = np.clip(rng.normal(rates, ses, size=(n_sims, len(rates))), 0, 1)
    i2022 = int(np.where(cycles == 2022)[0][0])
    horizon = target_year - 2022
    drift = rng.normal(0, persistence_drift_sd * np.sqrt(max(horizon, 1) / 4.0), n_sims)
    persistence = np.clip(draws[:, i2022] + drift, 0, 1)

    pre_mask = np.isin(cycles, PRE_COVID)
    xr, ser = cycles[pre_mask], ses[pre_mask]
    recovery = np.empty(n_sims)
    naive = np.empty(n_sims)
    for s in range(n_sims):
        recovery[s] = _predict(wls_trend(xr, draws[s, pre_mask], ser), target_year)[0]
        naive[s] = _predict(wls_trend(cycles, draws[s], ses), target_year)[0]
    recovery = np.clip(recovery, 0, 1)
    naive = np.clip(naive, 0, 1)
    partial = np.clip(0.5 * persistence + 0.5 * recovery, 0, 1)

    bank = {"persistence": persistence, "recovery": recovery,
            "partial": partial, "naive_linear": naive}
    return {
        k: ScenarioForecast(k, float(np.median(v)),
                            float(np.percentile(v, 5)), float(np.percentile(v, 95)), v)
        for k, v in bank.items()
    }


def scenarios_to_frame(forecast: dict[str, ScenarioForecast], target_year: int = 2026) -> pd.DataFrame:
    """Tidy summary (median + 90% predictive interval) for reporting."""
    rows = [{"scenario": s.name, "target_year": target_year,
             "median": round(s.median, 4), "pi90_low": round(s.lo, 4),
             "pi90_high": round(s.hi, 4)}
            for s in forecast.values()]
    return pd.DataFrame(rows)

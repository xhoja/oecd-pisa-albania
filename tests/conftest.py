"""Shared fixtures: small synthetic PISA-shaped frames (no real data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_pisa_frame(
    n: int = 400,
    n_pv: int = 10,
    n_replicates: int = 4,
    cycle: int = 2022,
    seed: int = 0,
    replicates_equal_base: bool = False,
) -> pd.DataFrame:
    """
    A synthetic frame with the columns the pipeline expects: a final weight,
    Fay replicate weights, `n_pv` math plausible values, a few background
    features, ids and a cycle. Deterministic given `seed`.

    replicates_equal_base: set every replicate weight equal to W_FSTUWT so BRR
        variance is exactly zero (used to pin down SE math in tests).
    """
    rng = np.random.default_rng(seed)
    base_w = rng.uniform(5, 50, n)
    # latent ability -> plausible values (correlated, jittered per draw)
    ability = rng.normal(430, 80, n)
    pv = {f"PV{i}MATH": ability + rng.normal(0, 12, n) for i in range(1, n_pv + 1)}

    df = pd.DataFrame({
        "W_FSTUWT": base_w,
        "ESCS": rng.normal(-0.5, 1.0, n),
        "HOMEPOS": rng.normal(-0.3, 1.0, n),
        "HISCED": rng.integers(0, 8, n).astype(float),
        "HISEI": rng.uniform(15, 90, n),
        "ANXMAT": rng.normal(0.2, 1.0, n),
        "BELONG": rng.normal(0.0, 1.0, n),
        "TEACHSUP": rng.normal(0.0, 1.0, n),
        "GRADE": rng.integers(-1, 2, n).astype(float),
        "GENDER": rng.integers(0, 2, n).astype(float),
        "CYCLE": cycle,
        "COUNTRY": "ALB",
        "CNTSTUID": np.arange(1, n + 1),
        **pv,
    })
    for r in range(1, n_replicates + 1):
        df[f"W_FSTURWT{r}"] = base_w if replicates_equal_base else base_w * rng.uniform(0.8, 1.2, n)
    return df


@pytest.fixture
def pisa_df() -> pd.DataFrame:
    return make_pisa_frame()


@pytest.fixture
def pisa_df_equal_reps() -> pd.DataFrame:
    """Replicate weights identical to the base weight → zero BRR variance."""
    return make_pisa_frame(replicates_equal_base=True)


@pytest.fixture
def two_country_df() -> pd.DataFrame:
    a = make_pisa_frame(n=200, seed=1)
    b = make_pisa_frame(n=300, seed=2)
    b["COUNTRY"] = "EST"
    return pd.concat([a, b], ignore_index=True)

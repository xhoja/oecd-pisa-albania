"""
Leakage-free, fit-in-fold versions of the engineered features.

The plain functions in ``src.features.engineer`` standardize / rank using the
*whole* dataframe they are handed. If you call them once before a CV split, the
test fold's mean, std, and rank distribution leak into the training features.
For descriptive EDA that is fine; for model evaluation it is not.

``EngineeredFeatureBuilder`` is an sklearn transformer: it learns the component
means / stds and the HOMEPOS rank distribution on ``fit`` (the train fold only)
and replays them on ``transform`` (train and test folds). Drop it in as the
first step of a CV pipeline and the engineered columns become leakage-safe.

Interaction terms are built from *centered* components (subtracting the train
mean) so the product term is not confounded with its main effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import structlog

logger = structlog.get_logger(__name__)

SES_COMPONENTS = ["ESCS", "HOMEPOS", "HISCED", "HISEI"]
ICT_COMPONENTS = ["ICTHOME", "ICTSCH", "COMPICT"]


class EngineeredFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Recompute engineered features inside a CV fold using train-fold statistics.

    Parameters
    ----------
    add_composites : build SES_COMPLETE and DIGITAL_READINESS (standardized means)
    add_material_deficit : build MATERIAL_DEFICIT from the train HOMEPOS rank curve
    add_interactions : build centered interaction products
    """

    def __init__(
        self,
        add_composites: bool = True,
        add_material_deficit: bool = True,
        add_interactions: bool = True,
    ):
        self.add_composites = add_composites
        self.add_material_deficit = add_material_deficit
        self.add_interactions = add_interactions

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EngineeredFeatureBuilder requires a DataFrame input")
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        for col in set(SES_COMPONENTS + ICT_COMPONENTS + ["ESCS", "ANXMAT", "BELONG", "TEACHSUP", "GRADE", "GENDER"]):
            if col in X.columns and X[col].notna().sum() > 10:
                self.means_[col] = float(X[col].mean())
                self.stds_[col] = float(X[col].std()) or 1.0
        # HOMEPOS rank curve: store sorted observed values to interpolate pct-rank
        if "HOMEPOS" in X.columns and X["HOMEPOS"].notna().any():
            self.homepos_sorted_ = np.sort(X["HOMEPOS"].dropna().values)
        else:
            self.homepos_sorted_ = None
        # Record interactions that actually have a non-null product on the train
        # fold; building one whose components never co-occur (e.g. ESCS present
        # only pre-2015, ANXMAT only post-2012) yields an all-NaN column that
        # scalers/linear models choke on and the imputer silently drops.
        pairs = {
            "SES_x_ANXIETY": ("ESCS", "ANXMAT"),
            "BELONG_x_TEACHSUP": ("BELONG", "TEACHSUP"),
            "GRADE_x_SES": ("GRADE", "ESCS"),
            "GENDER_x_ANXMAT": ("GENDER", "ANXMAT"),
        }
        self.interactions_ok_ = {
            name: cols for name, cols in pairs.items()
            if all(col in X.columns for col in cols)
            and (X[cols[0]] * X[cols[1]]).notna().any()
        }
        self.feature_names_in_ = list(X.columns)
        return self

    def _z(self, s: pd.Series, col: str) -> pd.Series:
        return (s - self.means_[col]) / self.stds_[col]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        df = X.copy()

        if self.add_composites:
            ses_avail = [c for c in SES_COMPONENTS if c in self.means_]
            if ses_avail:
                df["SES_COMPLETE"] = pd.concat([self._z(df[c], c) for c in ses_avail], axis=1).mean(axis=1)
            ict_avail = [c for c in ICT_COMPONENTS if c in self.means_]
            if ict_avail:
                df["DIGITAL_READINESS"] = pd.concat([self._z(df[c], c) for c in ict_avail], axis=1).mean(axis=1)
            if "ICTHOME" in df.columns and "ICTSCH" in df.columns:
                df["DIGITAL_GAP"] = df["ICTHOME"] - df["ICTSCH"]

        if self.add_material_deficit and self.homepos_sorted_ is not None and "HOMEPOS" in df.columns:
            n = len(self.homepos_sorted_)
            # percentile rank of each value against the TRAIN HOMEPOS distribution
            ranks = np.searchsorted(self.homepos_sorted_, df["HOMEPOS"].values, side="right") / n
            df["MATERIAL_DEFICIT"] = 1.0 - ranks

        if self.add_interactions:
            def c(col: str) -> pd.Series:
                # centered component (mean subtracted) so products aren't confounded
                return df[col] - self.means_.get(col, 0.0)

            for name, (a, b) in self.interactions_ok_.items():
                df[name] = c(a) * c(b)

        return df


class SchoolMeansTransformer(BaseEstimator, TransformerMixin):
    """
    Fold-safe survey-weighted school-mean (compositional) features.

    Production (``engineer.add_school_aggregates``) computes leave-one-out school
    means once over the **full cohort**; the fold-safe ablation
    (``scripts/run_school_features_foldsafe.py``) showed that lift is not a
    leakage artefact (identical delta under per-fold recomputation). This
    transformer packages that per-fold recomputation as a reusable sklearn step
    so a leakage-purist pipeline can drop it in instead of pre-computing the
    aggregates on the whole slice.

    On ``fit`` it learns, from the **train fold only**, each school's weighted sum
    and weight total (per aggregated column), the school sample sizes, and a
    global weighted mean fallback. It emits ``SCH_MEAN_<col>`` and ``SCH_N``:

    - ``fit_transform`` (train rows) → **leave-one-out** weighted mean, excluding
      the row's own contribution, so the feature is a genuine peer measure and
      the row's own label-correlated value is not smuggled in.
    - ``transform`` (test rows) → the **full** train-school weighted mean mapped
      by ``school_col``; schools unseen in train fall back to the global mean and
      ``SCH_N`` = 0 (they had no train students).

    The helper ``school_col`` and ``weight_col`` are **dropped** from the output
    (they are join/weight keys, not model features). Place this *before* the
    median imputer in a CV pipeline; keep ``CNTSCHID`` and ``W_FSTUWT`` in X so
    the transformer can see them.

    Parameters
    ----------
    cols : student indices to aggregate to school level
        (default ESCS/HOMEPOS/ANXMAT/TEACHSUP - the production set).
    school_col : school identifier column (default ``CNTSCHID``).
    weight_col : survey weight column (default ``W_FSTUWT``); absent → unweighted.
    drop_keys : drop ``school_col``/``weight_col`` from the output (default True).
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        school_col: str = "CNTSCHID",
        weight_col: str = "W_FSTUWT",
        drop_keys: bool = True,
    ):
        self.cols = cols
        self.school_col = school_col
        self.weight_col = weight_col
        self.drop_keys = drop_keys

    def _agg_cols(self, X: pd.DataFrame) -> list[str]:
        cols = self.cols if self.cols is not None else ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
        return [c for c in cols if c in X.columns]

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SchoolMeansTransformer requires a DataFrame input")
        self.feature_names_in_ = list(X.columns)
        self.agg_cols_ = self._agg_cols(X)
        self.has_school_ = self.school_col in X.columns
        if not self.has_school_:
            logger.warning("School column not found - transformer is a no-op", col=self.school_col)
            return self

        sch = X[self.school_col]
        w = X[self.weight_col].astype(float) if self.weight_col in X.columns else pd.Series(1.0, index=X.index)

        self.wx_sum_: dict[str, pd.Series] = {}
        self.wv_sum_: dict[str, pd.Series] = {}
        self.global_: dict[str, float] = {}
        for col in self.agg_cols_:
            x = X[col]
            nn = x.notna()
            wx = (w * x).where(nn, 0.0)
            wv = w.where(nn, 0.0)
            self.wx_sum_[col] = wx.groupby(sch).sum()
            self.wv_sum_[col] = wv.groupby(sch).sum()
            tot = float(wv.sum())
            self.global_[col] = float(wx.sum() / tot) if tot > 0 else np.nan
        self.size_ = sch.groupby(sch).size().astype(float)
        return self

    def _full_mean(self, col: str) -> pd.Series:
        wv = self.wv_sum_[col]
        return (self.wx_sum_[col] / wv.where(wv > 0)).fillna(self.global_[col])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        df = X.copy()
        if not self.has_school_:
            return df
        sch = df[self.school_col]
        for col in self.agg_cols_:
            lut = self._full_mean(col)
            df[f"SCH_MEAN_{col}"] = sch.map(lut).fillna(self.global_[col])
        df["SCH_N"] = sch.map(self.size_).fillna(0.0)
        return self._finish(df)

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        self.fit(X, y)
        df = X.copy()
        if not self.has_school_:
            return df
        sch = df[self.school_col]
        w = df[self.weight_col].astype(float) if self.weight_col in df.columns else pd.Series(1.0, index=df.index)
        for col in self.agg_cols_:
            x = df[col]
            nn = x.notna()
            wx_i = (w * x).where(nn, 0.0)
            wv_i = w.where(nn, 0.0)
            # leave-one-out: subtract the row's own contribution from its school total
            num = sch.map(self.wx_sum_[col]).astype(float) - wx_i
            den = sch.map(self.wv_sum_[col]).astype(float) - wv_i
            loo = (num / den.where(den > 0)).fillna(self.global_[col])
            df[f"SCH_MEAN_{col}"] = loo
        df["SCH_N"] = sch.map(self.size_).fillna(0.0)
        return self._finish(df)

    def _finish(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_keys:
            df = df.drop(columns=[c for c in (self.school_col, self.weight_col) if c in df.columns])
        return df

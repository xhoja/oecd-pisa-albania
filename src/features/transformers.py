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

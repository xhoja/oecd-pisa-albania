"""
Tests for the dashboard model export (scripts/export_dashboard_model.py).

The pure helpers (weighted MCC, weighted quantiles) are tested on synthetic
inputs with no real data. The end-to-end bundle check runs only when the export
artefact is present (it needs the real Albania-2022 slice, which is git-ignored),
so it is skipped in a clean checkout / CI.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_dashboard_model.py"
BUNDLE = ROOT / "outputs" / "models" / "dashboard_bundle.joblib"
META = ROOT / "outputs" / "models" / "dashboard_meta.json"


def _load_helpers():
    """Import the two pure helpers without executing main()."""
    spec = importlib.util.spec_from_file_location("_dash_export", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_helpers()


def test_weighted_mcc_perfect_and_useless():
    y = np.array([0, 0, 1, 1])
    w = np.array([1.0, 2.0, 1.0, 3.0])
    assert mod.weighted_mcc(y, y, w) == pytest.approx(1.0)
    # inverted prediction -> -1
    assert mod.weighted_mcc(y, 1 - y, w) == pytest.approx(-1.0)
    # constant prediction -> undefined denom -> 0 by convention
    assert mod.weighted_mcc(y, np.zeros_like(y), w) == 0.0


def test_weighted_mcc_matches_unweighted_when_equal_weights():
    from sklearn.metrics import matthews_corrcoef
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 40)
    p = rng.integers(0, 2, 40)
    w = np.ones(40)
    assert mod.weighted_mcc(y, p, w) == pytest.approx(matthews_corrcoef(y, p), abs=1e-9)


def test_wquantiles_equal_weights_matches_numpy():
    import pandas as pd
    x = pd.Series(np.arange(1.0, 101.0))
    w = pd.Series(np.ones(100))
    got = mod.wquantiles(x, w, [0.25, 0.5, 0.75])
    exp = list(np.quantile(x.to_numpy(), [0.25, 0.5, 0.75]))
    # weighted-quantile convention differs slightly from numpy's; allow a bin
    assert got == pytest.approx(exp, abs=1.5)


def test_wquantiles_weight_shifts_median():
    import pandas as pd
    x = pd.Series([0.0, 10.0])
    # nearly all weight on the high value -> weighted median near 10
    med = mod.wquantiles(x, pd.Series([1.0, 999.0]), [0.5])[0]
    assert med > 9.0


@pytest.mark.skipif(not BUNDLE.exists(), reason="export artefact absent (needs real data)")
def test_bundle_predicts_in_unit_interval():
    import joblib
    import pandas as pd

    bundle = joblib.load(BUNDLE)
    meta = json.loads(META.read_text())
    for key in ("pipeline", "isotonic", "feature_names", "shap_feature_names", "threshold"):
        assert key in bundle
    assert bundle["feature_names"] == meta["feature_names"]
    assert 0.0 < bundle["threshold"] < 1.0

    feats = bundle["feature_names"]
    vals = {f: 0.0 for f in feats}
    for col, spec in meta["sliders"].items():
        vals[col] = float(spec["default"])
    row = pd.DataFrame([[vals[f] for f in feats]], columns=feats)
    raw = float(bundle["pipeline"].predict_proba(row)[0, 1])
    cal = float(bundle["isotonic"].predict([raw])[0])
    assert 0.0 <= raw <= 1.0
    assert 0.0 <= cal <= 1.0


APP = ROOT / "reports" / "dashboard" / "app.py"


@pytest.mark.skipif(not BUNDLE.exists(), reason="export artefact absent (needs real data)")
def test_streamlit_app_runs_end_to_end():
    """Execute the whole app script in-process (unpickle + predict + SHAP render).

    A plain HTTP hit on a running server does NOT trigger the script body, so this
    is the check that actually exercises load_model()/predict/local_shap - it is
    what would have caught the ``No module named 'src'`` unpickle regression.
    """
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.selectbox) > 0
    # changing an answer must re-run the predict/SHAP path without error.
    # options are rendered via format_func, so the raw option value is the
    # integer index (0..n-1); set_value takes that, not the display label.
    sb = at.selectbox[0]
    sb.set_value(len(sb.options) - 1).run()
    assert not at.exception, at.exception
    # language switch must also re-render cleanly
    at.radio[0].set_value("Shqip").run()
    assert not at.exception, at.exception

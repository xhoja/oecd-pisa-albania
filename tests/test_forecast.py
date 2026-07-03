"""Tests for the scenario forecast (src/forecast)."""
import numpy as np

from src.forecast.scenarios import monte_carlo_forecast, scenarios_to_frame, wls_trend


def test_wls_recovers_known_line():
    # Exact line y = 2x + 1; WLS must recover slope 2, intercept 1.
    x = np.array([0, 1, 2, 3], float)
    y = 2 * x + 1
    se = np.ones_like(x)
    beta = wls_trend(x, y, se, degree=1)  # vander order: [slope, intercept]
    np.testing.assert_allclose(beta, [2.0, 1.0], atol=1e-9)


def test_wls_weights_favour_low_se_points():
    # One outlier with a huge SE must barely move the fit.
    x = np.array([0, 1, 2, 3], float)
    y = np.array([1.0, 3.0, 5.0, 99.0])  # last point off the y=2x+1 line
    se = np.array([1.0, 1.0, 1.0, 1e6])   # ...but with enormous uncertainty
    beta = wls_trend(x, y, se, degree=1)
    np.testing.assert_allclose(beta, [2.0, 1.0], atol=1e-3)


def test_scenarios_ordered_and_bounded():
    cycles = [2009, 2012, 2015, 2018, 2022]
    rates = [0.677, 0.607, 0.533, 0.424, 0.739]
    ses = [0.008, 0.005, 0.011, 0.009, 0.004]
    fc = monte_carlo_forecast(cycles, rates, ses, target_year=2025, n_sims=5000, seed=0)
    assert set(fc) == {"persistence", "recovery", "partial", "naive_linear"}
    # recovery (pre-COVID improving trend resumes) must sit well below persistence
    assert fc["recovery"].median < fc["persistence"].median
    # partial blend lies between the two
    assert fc["recovery"].median <= fc["partial"].median <= fc["persistence"].median
    for s in fc.values():
        assert 0.0 <= s.lo <= s.median <= s.hi <= 1.0


def test_persistence_near_2022_level():
    cycles = [2009, 2012, 2015, 2018, 2022]
    rates = [0.677, 0.607, 0.533, 0.424, 0.739]
    ses = [0.008, 0.005, 0.011, 0.009, 0.004]
    fc = monte_carlo_forecast(cycles, rates, ses, n_sims=5000, seed=1)
    assert abs(fc["persistence"].median - 0.739) < 0.03  # anchored to 2022


def test_frame_shape():
    cycles = [2009, 2012, 2015, 2018, 2022]
    rates = [0.677, 0.607, 0.533, 0.424, 0.739]
    ses = [0.008, 0.005, 0.011, 0.009, 0.004]
    fc = monte_carlo_forecast(cycles, rates, ses, n_sims=1000)
    frame = scenarios_to_frame(fc, target_year=2025)
    assert list(frame.columns) == ["scenario", "target_year", "median", "pi90_low", "pi90_high"]
    assert len(frame) == 4

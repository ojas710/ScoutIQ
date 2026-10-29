import numpy as np
import pytest

from src.evaluate import (conformal_quantile, coverage, interval_from_quantile, regression_metrics,
                          valuation_residual)
from src.formatting import format_eur_m, to_millions


def test_currency_conversion_and_format():
    assert to_millions(2_500_000) == 2.5
    assert to_millions(None) is None and to_millions(float("nan")) is None
    assert format_eur_m(12_500_000) == "€12.5m"
    assert format_eur_m(750_000) == "€750k"
    assert format_eur_m(-3_000_000) == "-€3.0m"
    assert format_eur_m(None) == "n/a"


def test_regression_metrics_known_values():
    y = np.array([100.0, 200.0, 400.0]); p = np.array([110.0, 180.0, 400.0])
    m = regression_metrics(y, p)
    assert m["MAE_eur"] == pytest.approx(10.0)
    assert m["RMSE_eur"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert m["R2"] == pytest.approx(1 - 500 / np.sum((y - y.mean()) ** 2))
    assert m["MAPE_pct"] == pytest.approx(100 * np.mean([0.1, 0.1, 0.0]))


def test_metrics_reject_invalid_input():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        regression_metrics([0.0, 1.0], [1.0, 1.0])       # MAPE/log undefined
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_valuation_residual_sign_convention():
    r = valuation_residual([5.0, 10.0], [8.0, 7.0])
    assert r[0] < 0      # observed below model estimate -> potentially undervalued
    assert r[1] > 0
    assert list(r) == [-3.0, 3.0]


def test_conformal_interval_coverage_on_synthetic_noise():
    rng = np.random.default_rng(0)
    pred = np.exp(rng.normal(15, 1, 20000))
    obs = pred * np.exp(rng.normal(0, 0.5, pred.size))
    cal, tst = slice(0, 10000), slice(10000, None)
    q = conformal_quantile(obs[cal], pred[cal], 0.10)
    lo, hi = interval_from_quantile(pred[tst], q)
    assert coverage(obs[tst], lo, hi) == pytest.approx(0.90, abs=0.015)
    assert np.all(lo < pred[tst]) and np.all(hi > pred[tst])

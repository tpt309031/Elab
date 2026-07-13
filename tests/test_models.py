from __future__ import annotations

import numpy as np

from research.model_candidates import (
    HurdleReturnClassifier,
    OrdinalLogisticClassifier,
    QuantileReturnClassifier,
    candidate_model_specs,
    learn_simplex_weights,
    model_availability_rows,
)


def _assert_probabilities(probabilities: np.ndarray, rows: int) -> None:
    assert probabilities.shape == (rows, 3)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)


def test_custom_estimators_emit_three_class_probabilities() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(180, 5))
    returns = 0.012 * values[:, 0] - 0.008 * values[:, 1] + rng.normal(0, 0.012, 180)
    labels = np.select([returns < -0.01, returns > 0.01], [0, 2], default=1)

    ordinal = OrdinalLogisticClassifier(max_iter=300).fit(values, labels)
    hurdle = HurdleReturnClassifier(max_iter=300).fit(values, returns)
    quantile = QuantileReturnClassifier(max_iter=12, min_samples_leaf=8).fit(values, returns)

    _assert_probabilities(ordinal.predict_proba(values[:13]), 13)
    _assert_probabilities(hurdle.predict_proba(values[:13]), 13)
    _assert_probabilities(quantile.predict_proba(values[:13]), 13)


def test_simplex_stacking_is_nonnegative_and_prefers_better_oos_member() -> None:
    labels = np.array([0, 1, 2] * 30, dtype=int)
    strong = np.full((len(labels), 3), 0.05)
    strong[np.arange(len(labels)), labels] = 0.90
    weak = np.roll(strong, 1, axis=1)
    weights, diagnostics = learn_simplex_weights([strong, weak], labels, l2_regularization=0.01)

    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8)
    assert (weights >= 0).all()
    assert weights[0] > weights[1]
    assert diagnostics["loss"] < diagnostics["uniform_loss"]


def test_candidate_registry_keeps_optional_models_explicit() -> None:
    columns = [
        "market_return_1", "market_return_3", "volatility_7", "volatility_21",
        "distance_ma20", "rsi14", "atr14_pct", "volatility_regime_z",
        "trend_regime_score", "return_change_point", "volatility_change_point",
        "index_btc_slope_3", "index_me_slope_3", "gap_index", "astro_volatility_z30",
    ]
    spec_names = {spec.name for spec in candidate_model_specs(columns)}
    status = {row["model"]: row for row in model_availability_rows()}

    required = {
        "Logistic", "Ordinal Logistic", "Random Forest", "HistGradientBoosting",
        "Quantile Boosting", "Hurdle Return", "XGBoost", "LightGBM", "CatBoost", "HMM Regime",
    }
    assert required == set(status)
    for name in required:
        if status[name]["available"]:
            assert name in spec_names

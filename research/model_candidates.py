from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("down", "sideway", "up")
SIDEWAY_LIMIT = 0.01
UP_CORRECT = 0.03
DOWN_CORRECT = -0.03
UP_PARTIAL_MIN = 0.001
DOWN_PARTIAL_MAX = -0.001


@dataclass(frozen=True)
class ModelCandidateSpec:
    name: str
    estimator: object
    columns: tuple[str, ...]
    target_column: str
    family: str


def _normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    output = np.asarray(probabilities, dtype=float)
    output = np.nan_to_num(output, nan=1 / 3, posinf=1 / 3, neginf=1 / 3)
    output = np.clip(output, 1e-8, None)
    return output / output.sum(axis=1, keepdims=True)


def _binary_probability(model: LogisticRegression | None, prior: float, values: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(values), prior, dtype=float)
    return model.predict_proba(values)[:, 1]


class OrdinalLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Three-class proportional-odds approximation with monotone cumulative probabilities."""

    def __init__(self, C: float = 0.35, max_iter: int = 1200) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogisticClassifier":
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        self.classes_ = np.arange(3)
        self.models_: list[LogisticRegression | None] = []
        self.priors_: list[float] = []
        for threshold in (0, 1):
            binary = (labels > threshold).astype(int)
            prior = float((binary.sum() + 1) / (len(binary) + 2))
            self.priors_.append(prior)
            if np.unique(binary).size < 2:
                self.models_.append(None)
                continue
            model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight="balanced",
            )
            model.fit(values, binary)
            self.models_.append(model)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        greater_zero = _binary_probability(self.models_[0], self.priors_[0], values)
        greater_one = _binary_probability(self.models_[1], self.priors_[1], values)
        greater_one = np.minimum(greater_one, greater_zero)
        return _normalise_probabilities(np.column_stack([
            1 - greater_zero,
            greater_zero - greater_one,
            greater_one,
        ]))


class QuantileReturnClassifier(ClassifierMixin, BaseEstimator):
    """Models continuous return quantiles, then maps them to the existing three classes."""

    def __init__(
        self,
        max_iter: int = 90,
        learning_rate: float = 0.045,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 18,
        random_state: int = 42,
        sideway_limit: float = SIDEWAY_LIMIT,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.sideway_limit = sideway_limit

    def _estimator(self, quantile: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=1.5,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileReturnClassifier":
        values = np.asarray(X, dtype=float)
        returns = np.asarray(y, dtype=float)
        self.classes_ = np.arange(3)
        self.models_ = [self._estimator(quantile) for quantile in (0.15, 0.50, 0.85)]
        for model in self.models_:
            model.fit(values, returns)
        median_residual = returns - self.models_[1].predict(values)
        residual_mad = float(np.nanmedian(np.abs(median_residual - np.nanmedian(median_residual))))
        self.residual_scale_ = max(0.0025, residual_mad / math.log(3))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        quantiles = np.column_stack([model.predict(values) for model in self.models_])
        quantiles.sort(axis=1)
        median = quantiles[:, 1]
        # The 15th-to-85th percentile distance of a logistic distribution is 3.469 scales.
        scale = np.maximum((quantiles[:, 2] - quantiles[:, 0]) / 3.469, self.residual_scale_)
        down = 1 / (1 + np.exp(np.clip((median + self.sideway_limit) / scale, -40, 40)))
        up = 1 / (1 + np.exp(np.clip((self.sideway_limit - median) / scale, -40, 40)))
        sideway = np.maximum(1e-8, 1 - down - up)
        return _normalise_probabilities(np.column_stack([down, sideway, up]))


class HurdleReturnClassifier(ClassifierMixin, BaseEstimator):
    """Separates move magnitude from the conditional direction of a material move."""

    def __init__(self, C: float = 0.4, max_iter: int = 1200, sideway_limit: float = SIDEWAY_LIMIT) -> None:
        self.C = C
        self.max_iter = max_iter
        self.sideway_limit = sideway_limit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HurdleReturnClassifier":
        values = np.asarray(X, dtype=float)
        returns = np.asarray(y, dtype=float)
        self.classes_ = np.arange(3)
        magnitude = (np.abs(returns) > self.sideway_limit).astype(int)
        self.magnitude_prior_ = float((magnitude.sum() + 1) / (len(magnitude) + 2))
        self.magnitude_model_: LogisticRegression | None = None
        if np.unique(magnitude).size == 2:
            self.magnitude_model_ = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight="balanced",
            ).fit(values, magnitude)
        material = magnitude.astype(bool)
        sign = (returns[material] > 0).astype(int)
        self.up_prior_ = float((sign.sum() + 1) / (len(sign) + 2)) if len(sign) else 0.5
        self.sign_model_: LogisticRegression | None = None
        if len(sign) >= 20 and np.unique(sign).size == 2:
            self.sign_model_ = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight="balanced",
            ).fit(values[material], sign)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        move = _binary_probability(self.magnitude_model_, self.magnitude_prior_, values)
        up_given_move = _binary_probability(self.sign_model_, self.up_prior_, values)
        return _normalise_probabilities(np.column_stack([
            move * (1 - up_given_move),
            1 - move,
            move * up_given_move,
        ]))


class ThresholdUtilityClassifier(ClassifierMixin, BaseEstimator):
    """Estimate the score-rule thresholds directly from the continuous return.

    The dashboard grade is not a plain three-class target: a directional call can
    earn a partial score before it reaches +/-3%, while SIDEWAY is exact inside
    +/-1%. Independent cumulative models preserve that information and return
    normalized expected grading utilities for DOWN, SIDEWAY and UP.
    """

    def __init__(self, C: float = 0.30, max_iter: int = 1200) -> None:
        self.C = C
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdUtilityClassifier":
        values = np.asarray(X, dtype=float)
        returns = np.asarray(y, dtype=float)
        self.classes_ = np.arange(3)
        self.thresholds_ = np.asarray([
            DOWN_CORRECT,
            -SIDEWAY_LIMIT,
            DOWN_PARTIAL_MAX,
            UP_PARTIAL_MIN,
            SIDEWAY_LIMIT,
            UP_CORRECT,
        ])
        self.models_: list[LogisticRegression | None] = []
        self.priors_: list[float] = []
        for threshold in self.thresholds_:
            above = (returns > threshold).astype(int)
            prior = float((above.sum() + 1) / (len(above) + 2))
            self.priors_.append(prior)
            if np.unique(above).size < 2:
                self.models_.append(None)
                continue
            self.models_.append(LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
            ).fit(values, above))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        probability_above = np.column_stack([
            _binary_probability(model, prior, values)
            for model, prior in zip(self.models_, self.priors_)
        ])
        # P(r > threshold) must be non-increasing as the threshold rises.
        probability_above = np.minimum.accumulate(probability_above, axis=1)
        down_utility = 0.5 * (
            (1 - probability_above[:, 0]) + (1 - probability_above[:, 2])
        )
        sideway_utility = np.maximum(
            1e-8,
            probability_above[:, 1] - probability_above[:, 4],
        )
        up_utility = 0.5 * (probability_above[:, 3] + probability_above[:, 5])
        return _normalise_probabilities(np.column_stack([
            down_utility,
            sideway_utility,
            up_utility,
        ]))


class HMMRegimeClassifier(ClassifierMixin, BaseEstimator):
    """Maps train-only HMM regimes to classes without Viterbi look-ahead on test rows."""

    def __init__(self, n_components: int = 4, n_iter: int = 35, random_state: int = 42) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HMMRegimeClassifier":
        from hmmlearn.hmm import GaussianHMM

        values = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        component_count = min(self.n_components, max(2, len(values) // 80))
        self.classes_ = np.arange(3)
        self.model_ = GaussianHMM(
            n_components=component_count,
            covariance_type="diag",
            n_iter=self.n_iter,
            tol=1e-3,
            min_covar=1e-4,
            random_state=self.random_state,
        )
        self.model_.fit(values)
        train_state_probabilities = self.model_.predict_proba(values)
        self.state_class_probabilities_ = np.full((component_count, 3), 0.5, dtype=float)
        for class_index in range(3):
            self.state_class_probabilities_[:, class_index] += train_state_probabilities[labels == class_index].sum(axis=0)
        self.state_class_probabilities_ /= self.state_class_probabilities_.sum(axis=1, keepdims=True)
        try:
            self.state_prior_ = np.asarray(self.model_.get_stationary_distribution(), dtype=float)
        except (AttributeError, ValueError, np.linalg.LinAlgError):
            self.state_prior_ = np.asarray(self.model_.startprob_, dtype=float)
        self.state_prior_ = np.clip(self.state_prior_, 1e-8, None)
        self.state_prior_ /= self.state_prior_.sum()
        return self

    def _independent_state_probabilities(self, values: np.ndarray) -> np.ndarray:
        means = np.asarray(self.model_.means_, dtype=float)
        covariances = np.asarray(self.model_.covars_, dtype=float)
        if covariances.ndim == 3:
            covariances = np.diagonal(covariances, axis1=1, axis2=2)
        covariances = np.maximum(covariances, 1e-6)
        difference = values[:, None, :] - means[None, :, :]
        log_emission = -0.5 * np.sum(
            np.log(2 * np.pi * covariances)[None, :, :] + difference**2 / covariances[None, :, :],
            axis=2,
        )
        log_posterior = log_emission + np.log(self.state_prior_)[None, :]
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        state_probabilities = self._independent_state_probabilities(values)
        return _normalise_probabilities(state_probabilities @ self.state_class_probabilities_)


def candidate_model_specs(feature_columns: Sequence[str], random_state: int = 42) -> list[ModelCandidateSpec]:
    columns = tuple(feature_columns)
    linear_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), list(columns))],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(columns))],
        remainder="drop",
    )

    def linear_pipeline(estimator: object) -> Pipeline:
        return Pipeline([("preprocess", clone(linear_preprocessor)), ("model", estimator)])

    def tree_pipeline(estimator: object) -> Pipeline:
        return Pipeline([("preprocess", clone(tree_preprocessor)), ("model", estimator)])

    specs = [
        ModelCandidateSpec(
            "Logistic",
            linear_pipeline(LogisticRegression(C=0.35, max_iter=1500, class_weight="balanced")),
            columns,
            "target",
            "linear",
        ),
        ModelCandidateSpec(
            "Ordinal Logistic",
            linear_pipeline(OrdinalLogisticClassifier(C=0.35)),
            columns,
            "target",
            "ordinal",
        ),
        ModelCandidateSpec(
            "Random Forest",
            tree_pipeline(RandomForestClassifier(
                n_estimators=240,
                max_depth=7,
                min_samples_leaf=10,
                max_features=0.65,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            )),
            columns,
            "target",
            "tree",
        ),
        ModelCandidateSpec(
            "HistGradientBoosting",
            tree_pipeline(HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=1.5,
                min_samples_leaf=18,
                random_state=random_state,
            )),
            columns,
            "target",
            "boosting",
        ),
        ModelCandidateSpec(
            "Quantile Boosting",
            tree_pipeline(QuantileReturnClassifier(random_state=random_state)),
            columns,
            "daily_return",
            "distributional-regression",
        ),
        ModelCandidateSpec(
            "Hurdle Return",
            linear_pipeline(HurdleReturnClassifier()),
            columns,
            "daily_return",
            "hurdle",
        ),
        ModelCandidateSpec(
            "Threshold Utility",
            linear_pipeline(ThresholdUtilityClassifier()),
            columns,
            "daily_return",
            "threshold-utility",
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(ModelCandidateSpec(
            "XGBoost",
            tree_pipeline(XGBClassifier(
                n_estimators=220,
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=8,
                subsample=0.78,
                colsample_bytree=0.72,
                reg_alpha=0.2,
                reg_lambda=2.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=random_state,
            )),
            columns,
            "target",
            "boosting",
        ))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        specs.append(ModelCandidateSpec(
            "LightGBM",
            tree_pipeline(LGBMClassifier(
                n_estimators=220,
                num_leaves=15,
                learning_rate=0.035,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=2.0,
                verbosity=-1,
                n_jobs=-1,
                random_state=random_state,
            )),
            columns,
            "target",
            "boosting",
        ))
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        specs.append(ModelCandidateSpec(
            "CatBoost",
            tree_pipeline(CatBoostClassifier(
                iterations=190,
                depth=5,
                learning_rate=0.04,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )),
            columns,
            "target",
            "boosting",
        ))
    except ImportError:
        pass

    if importlib.util.find_spec("hmmlearn") is not None:
        regime_candidates = (
            "market_return_1", "market_return_3", "volatility_7", "volatility_21",
            "distance_ma20", "rsi14", "atr14_pct", "volatility_regime_z",
            "trend_regime_score", "return_change_point", "volatility_change_point",
            "index_btc_slope_3", "index_me_slope_3", "gap_index", "astro_volatility_z30",
        )
        regime_columns = tuple(column for column in regime_candidates if column in columns)
        if len(regime_columns) >= 4:
            regime_preprocessor = ColumnTransformer(
                [("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]), list(regime_columns))],
                remainder="drop",
            )
            specs.append(ModelCandidateSpec(
                "HMM Regime",
                Pipeline([
                    ("preprocess", regime_preprocessor),
                    ("model", HMMRegimeClassifier(random_state=random_state)),
                ]),
                regime_columns,
                "target",
                "regime",
            ))
    return specs


def model_availability_rows() -> list[dict[str, object]]:
    installed = lambda package: importlib.util.find_spec(package) is not None
    return [
        {"model": "Logistic", "family": "linear", "available": True, "cadence": "daily"},
        {"model": "Ordinal Logistic", "family": "ordinal", "available": True, "cadence": "daily"},
        {"model": "Random Forest", "family": "tree", "available": True, "cadence": "daily"},
        {"model": "HistGradientBoosting", "family": "boosting", "available": True, "cadence": "daily"},
        {"model": "Quantile Boosting", "family": "distributional-regression", "available": True, "cadence": "daily"},
        {"model": "Hurdle Return", "family": "hurdle", "available": True, "cadence": "daily"},
        {"model": "Threshold Utility", "family": "threshold-utility", "available": True, "cadence": "daily"},
        {"model": "XGBoost", "family": "boosting", "available": installed("xgboost"), "cadence": "daily"},
        {"model": "LightGBM", "family": "boosting", "available": installed("lightgbm"), "cadence": "daily"},
        {"model": "CatBoost", "family": "boosting", "available": installed("catboost"), "cadence": "daily"},
        {"model": "HMM Regime", "family": "regime", "available": installed("hmmlearn"), "cadence": "daily"},
    ]


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indices = np.arange(1, len(vector) + 1)
    valid = ordered - cumulative / indices > 0
    threshold_index = np.flatnonzero(valid)[-1]
    threshold = cumulative[threshold_index] / (threshold_index + 1)
    return np.maximum(vector - threshold, 0)


def learn_simplex_weights(
    probability_sets: Sequence[np.ndarray],
    labels: np.ndarray,
    l2_regularization: float = 0.08,
    max_iter: int = 800,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if not probability_sets:
        return np.empty(0), {"method": "nonnegative-simplex-oof", "iterations": 0}
    tensor = np.stack([_normalise_probabilities(item) for item in probability_sets], axis=0)
    target = np.asarray(labels, dtype=int)
    count = len(probability_sets)
    uniform = np.full(count, 1 / count)
    weights = uniform.copy()

    def loss(candidate: np.ndarray) -> float:
        mixture = np.einsum("m,mnc->nc", candidate, tensor)
        selected = np.clip(mixture[np.arange(len(target)), target], 1e-10, 1)
        return float(-np.log(selected).mean() + l2_regularization * np.square(candidate - uniform).sum())

    best_weights = weights.copy()
    best_loss = loss(weights)
    stale_iterations = 0
    iteration = 0
    for iteration in range(1, max_iter + 1):
        mixture = np.einsum("m,mnc->nc", weights, tensor)
        selected = np.clip(mixture[np.arange(len(target)), target], 1e-10, 1)
        selected_by_model = tensor[:, np.arange(len(target)), target]
        gradient = -np.mean(selected_by_model / selected[None, :], axis=1)
        gradient += 2 * l2_regularization * (weights - uniform)
        step = 0.08 / math.sqrt(iteration)
        weights = project_to_simplex(weights - step * gradient)
        current_loss = loss(weights)
        if current_loss < best_loss - 1e-9:
            best_loss = current_loss
            best_weights = weights.copy()
            stale_iterations = 0
        else:
            stale_iterations += 1
        if stale_iterations >= 100:
            break
    return best_weights, {
        "method": "nonnegative-simplex-oof",
        "iterations": iteration,
        "loss": best_loss,
        "uniform_loss": loss(uniform),
        "l2_regularization": l2_regularization,
    }

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_sample_weight

from models import build_model


PUBLIC_FEATURE_BOUNDS = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (10, 100),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}


class PublicBoundsMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_bounds=None, clip=True):
        self.feature_bounds = feature_bounds
        self.clip = clip

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = list(X.columns)

        bounds = self.feature_bounds or PUBLIC_FEATURE_BOUNDS
        missing = [name for name in self.feature_names_in_ if name not in bounds]
        if missing:
            raise ValueError("Missing public bounds for: " + ", ".join(missing))

        lower = []
        upper = []
        for name in self.feature_names_in_:
            lo, hi = bounds[name]
            if hi <= lo:
                raise ValueError(f"Invalid bounds for {name}: {(lo, hi)}")
            lower.append(float(lo))
            upper.append(float(hi))

        self.lower_ = np.asarray(lower)
        self.upper_ = np.asarray(upper)
        self.range_ = self.upper_ - self.lower_
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_", "range_"])

        X = self._as_frame(X, self.feature_names_in_)
        values = X[self.feature_names_in_].to_numpy(dtype=float, copy=True)

        if self.clip:
            values = np.clip(values, self.lower_, self.upper_)

        return (values - self.lower_) / self.range_

    @staticmethod
    def _as_frame(X, columns=None):
        if hasattr(X, "iloc"):
            return X.copy()

        values = np.asarray(X)
        if columns is None:
            columns = [f"feature_{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, columns=list(columns))


class L2RowNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def fit(self, X, y=None):
        if self.max_norm <= 0:
            raise ValueError("max_norm must be positive")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_norm / (norm + 1e-12))
        return X * scale


class GaussianDPLogisticRegression(BaseEstimator, ClassifierMixin):
    """Small logistic regression implementation with Gaussian output noise."""

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        l2_regularization=0.1,
        row_l2_bound=1.0,
        fit_intercept=True,
        class_weight="balanced",
        max_iter=300,
        tol=1e-6,
        random_state=None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.l2_regularization = l2_regularization
        self.row_l2_bound = row_l2_bound
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        self._check_params()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if not np.array_equal(np.unique(y), np.array([0, 1])):
            raise ValueError("GaussianDPLogisticRegression expects labels 0 and 1.")

        weights = self._sample_weights(y)
        X_aug = self._add_intercept(X)
        theta0 = np.zeros(X_aug.shape[1])

        result = minimize(
            fun=lambda theta: self._loss_and_grad(theta, X_aug, y, weights)[0],
            x0=theta0,
            jac=lambda theta: self._loss_and_grad(theta, X_aug, y, weights)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.optimization_warning_ = None if result.success else result.message

        theta = np.asarray(result.x, dtype=float)
        sigma = self._noise_sigma(
            n_samples=len(y),
            max_sample_weight=float(weights.max()),
        )

        rng = np.random.default_rng(self.random_state)
        theta_private = theta + rng.normal(0.0, sigma, size=theta.shape)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.theta_non_private_ = theta
        self.theta_private_ = theta_private
        self.noise_sigma_ = sigma
        self.max_sample_weight_ = float(weights.max())
        self.sample_weight_mean_ = float(weights.mean())

        if self.fit_intercept:
            self.coef_ = theta_private[:-1].reshape(1, -1)
            self.intercept_ = np.array([theta_private[-1]])
        else:
            self.coef_ = theta_private.reshape(1, -1)
            self.intercept_ = np.array([0.0])

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _check_params(self):
        if self.epsilon is None or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta is None or not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if self.l2_regularization <= 0:
            raise ValueError("l2_regularization must be positive")
        if self.row_l2_bound <= 0:
            raise ValueError("row_l2_bound must be positive")

    def _sample_weights(self, y):
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")

        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        if n0 == 0 or n1 == 0:
            raise ValueError("Both classes are needed for balanced weights.")

        w0 = len(y) / (2.0 * n0)
        w1 = len(y) / (2.0 * n1)
        return np.where(y == 1, w1, w0)

    def _loss_and_grad(self, theta, X, y, weights):
        scores = X @ theta
        losses = np.logaddexp(0.0, scores) - y * scores
        loss = np.mean(weights * losses)
        loss += 0.5 * self.l2_regularization * np.dot(theta, theta)

        error = weights * (expit(scores) - y)
        grad = (X.T @ error) / len(y)
        grad += self.l2_regularization * theta

        return loss, grad

    def _noise_sigma(self, n_samples, max_sample_weight):
        if self.fit_intercept:
            row_bound = math.sqrt(self.row_l2_bound ** 2 + 1.0)
        else:
            row_bound = self.row_l2_bound

        sensitivity = (
            2.0 * max_sample_weight * row_bound
            / (n_samples * self.l2_regularization)
        )
        self.sensitivity_ = sensitivity

        return (
            math.sqrt(2.0 * math.log(1.25 / self.delta))
            * sensitivity
            / self.epsilon
        )

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.column_stack([X, np.ones(X.shape[0])])


def build_gaussian_dp_logistic_regression(
    epsilon,
    delta,
    random_state,
    l2_regularization=0.1,
    row_l2_bound=1.0,
    max_iter=300,
):
    return Pipeline([
        ("public_bounds_scaler", PublicBoundsMinMaxScaler()),
        ("row_l2_normalizer", L2RowNormalizer(max_norm=row_l2_bound)),
        ("classifier", GaussianDPLogisticRegression(
            epsilon=float(epsilon),
            delta=float(delta),
            l2_regularization=float(l2_regularization),
            row_l2_bound=float(row_l2_bound),
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])


def privatize_training_data(
    X_train,
    mechanism,
    epsilon,
    random_state,
    split_epsilon_across_features=False,
):
    if mechanism != "ldp_randomized_response":
        return X_train.copy()
    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be positive for randomized response")

    return randomized_response_dataframe(
        X_train,
        epsilon,
        random_state,
        split_epsilon_across_features=split_epsilon_across_features,
    )


def randomized_response_dataframe(
    X,
    epsilon,
    random_state,
    split_epsilon_across_features=False,
):
    rng = np.random.default_rng(random_state)
    X_private = X.copy()

    n_features = len(X_private.columns)
    eps_feature = epsilon / n_features if split_epsilon_across_features else epsilon
    exp_eps = np.exp(eps_feature)

    for column in X_private.columns:
        values = X_private[column].to_numpy(copy=True)
        domain = pd.Series(values).dropna().unique()

        if len(domain) <= 1:
            continue

        keep_prob = exp_eps / (exp_eps + len(domain) - 1)
        replace_idx = np.where(rng.random(len(values)) > keep_prob)[0]

        for idx in replace_idx:
            candidates = domain[domain != values[idx]]
            if len(candidates):
                values[idx] = rng.choice(candidates)

        X_private[column] = values

    return X_private


def add_laplace_parameter_noise(model, epsilon, n_samples, random_state):
    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be positive for Laplace parameter noise")

    rng = np.random.default_rng(random_state)
    classifier = model.named_steps.get("classifier", model)

    # Proposal-style parameter perturbation. We do not claim this as formal DP.
    scale = (1.0 / np.sqrt(n_samples)) / epsilon

    def laplace(shape):
        return rng.laplace(0.0, scale, size=shape)

    if hasattr(classifier, "coef_"):
        classifier.coef_ = classifier.coef_ + laplace(classifier.coef_.shape)
    if hasattr(classifier, "intercept_"):
        classifier.intercept_ = classifier.intercept_ + laplace(classifier.intercept_.shape)
    if hasattr(classifier, "coefs_"):
        classifier.coefs_ = [w + laplace(w.shape) for w in classifier.coefs_]
    if hasattr(classifier, "intercepts_"):
        classifier.intercepts_ = [b + laplace(b.shape) for b in classifier.intercepts_]

    return model


def train_model_with_defense(
    model_name,
    mechanism,
    X_train,
    y_train,
    epsilon,
    delta,
    random_state,
):
    if mechanism == "gaussian":
        if model_name != "logistic_regression":
            raise NotImplementedError(
                "Gaussian DP is implemented only for logistic_regression."
            )
        if delta is None:
            raise ValueError("delta is required for Gaussian DP")

        model = build_gaussian_dp_logistic_regression(
            epsilon=epsilon,
            delta=delta,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model

    X_used = privatize_training_data(
        X_train,
        mechanism,
        epsilon,
        random_state,
    )

    model = build_model(model_name, random_state=random_state)

    if model_name == "mlp":
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )
        model.fit(
            X_used,
            y_train,
            classifier__sample_weight=sample_weight,
        )
    else:
        model.fit(X_used, y_train)

    if mechanism == "laplace":
        model = add_laplace_parameter_noise(
            model,
            epsilon=epsilon,
            n_samples=len(X_used),
            random_state=random_state,
        )

    return model

import numpy as np
import pandas as pd


def privatize_training_data(X_train, mechanism, epsilon, random_state):
    """
    Apply input-level privacy mechanisms before training.

    Only LDP randomized response changes the training data.
    Laplace/Gaussian are parameter-noise mechanisms and are applied after fitting.
    """
    if mechanism != "ldp_randomized_response":
        return X_train.copy()

    return randomized_response_dataframe(X_train, epsilon, random_state)


def randomized_response_dataframe(X, epsilon, random_state):
    """
    Simple feature-wise generalized randomized response.

    For every feature value:
    - keep the true value with probability exp(epsilon) / (exp(epsilon) + k - 1)
    - otherwise replace it with another value from the same column domain

    This keeps all values valid for the dataset columns.
    """
    rng = np.random.default_rng(random_state)
    X_private = X.copy()
    exp_eps = np.exp(epsilon)

    for column in X_private.columns:
        values = X_private[column].to_numpy(copy=True)
        domain = pd.Series(values).dropna().unique()
        k = len(domain)

        if k <= 1:
            continue

        keep_prob = exp_eps / (exp_eps + k - 1)
        replace_mask = rng.random(len(values)) > keep_prob
        replace_indices = np.where(replace_mask)[0]

        for idx in replace_indices:
            old_value = values[idx]
            candidates = domain[domain != old_value]

            if len(candidates) > 0:
                values[idx] = rng.choice(candidates)

        X_private[column] = values

    return X_private


def add_parameter_noise(model, mechanism, epsilon, delta, n_samples, random_state):
    """
    Add noise to fitted model parameters.

    Supported:
    - LogisticRegression: coef_, intercept_
    - MLPClassifier: coefs_, intercepts_

    The sensitivity value is a simple reproducible project choice, not a formal
    proof for the complete scikit-learn training algorithm.
    """
    if mechanism not in ["laplace", "gaussian"]:
        return model

    rng = np.random.default_rng(random_state)
    classifier = model.named_steps.get("classifier", model)

    sensitivity = 1.0 / np.sqrt(n_samples)

    def generate_noise(shape):
        if mechanism == "laplace":
            scale = sensitivity / epsilon
            return rng.laplace(loc=0.0, scale=scale, size=shape)

        sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon
        return rng.normal(loc=0.0, scale=sigma, size=shape)

    # Logistic Regression
    if hasattr(classifier, "coef_"):
        classifier.coef_ = classifier.coef_ + generate_noise(classifier.coef_.shape)

    if hasattr(classifier, "intercept_"):
        classifier.intercept_ = classifier.intercept_ + generate_noise(
            classifier.intercept_.shape
        )

    # MLPClassifier
    if hasattr(classifier, "coefs_"):
        classifier.coefs_ = [
            weights + generate_noise(weights.shape)
            for weights in classifier.coefs_
        ]

    if hasattr(classifier, "intercepts_"):
        classifier.intercepts_ = [
            bias + generate_noise(bias.shape)
            for bias in classifier.intercepts_
        ]

    return model
"""Alternating Conditional Expectations with regression-tree learners."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class _StandardizedTree:
    """Regression tree followed by empirical centering and normalization."""

    def __init__(self, learner, mean, scale):
        self.learner = learner
        self.mean = mean
        self.scale = scale

    def predict(self, inputs):
        inputs = np.asarray(inputs).reshape(len(inputs), -1)
        return (self.learner.predict(inputs) - self.mean) / self.scale


def _fit_tree(inputs, targets, max_depth, min_samples_leaf, seed):
    learner = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    learner.fit(inputs, targets)
    fitted_values = learner.predict(inputs)
    mean = fitted_values.mean()
    scale = fitted_values.std()

    if scale <= 1e-12:
        raise np.linalg.LinAlgError(
            "ACE produced a constant conditional-mean estimate."
        )

    return _StandardizedTree(learner, mean, scale)


def fit_ace(
    x,
    y,
    n_iterations=20,
    max_depth=3,
    min_samples_leaf=20,
    seed=0,
):
    """Fit rank-one ACE and return normalized transformations and correlation."""
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    rng = np.random.default_rng(seed)

    psi_model = _fit_tree(
        y,
        rng.normal(size=len(y)),
        max_depth,
        min_samples_leaf,
        seed,
    )
    correlation_history = []

    for iteration in range(n_iterations):
        phi_model = _fit_tree(
            x,
            psi_model.predict(y),
            max_depth,
            min_samples_leaf,
            seed + 2 * iteration + 1,
        )
        psi_model = _fit_tree(
            y,
            phi_model.predict(x),
            max_depth,
            min_samples_leaf,
            seed + 2 * iteration + 2,
        )
        phi = phi_model.predict(x)
        psi = psi_model.predict(y)
        correlation_history.append(np.mean(phi * psi))

    singular_value = correlation_history[-1]
    history = {"correlation": np.asarray(correlation_history)}
    return phi_model, psi_model, singular_value, history

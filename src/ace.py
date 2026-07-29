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


class _OrthogonalizedTree:
    """Regression tree centered, deflated, and normalized on training data."""

    def __init__(
        self,
        learner,
        previous_models,
        projection,
        mean,
        scale,
    ):
        self.learner = learner
        self.previous_models = previous_models
        self.projection = projection
        self.mean = mean
        self.scale = scale

    def predict(self, inputs):
        inputs = np.asarray(inputs).reshape(len(inputs), -1)
        values = self.learner.predict(inputs) - self.mean
        if self.previous_models:
            previous_values = np.column_stack(
                [model.predict(inputs) for model in self.previous_models]
            )
            values -= previous_values @ self.projection
        return values / self.scale


class _FeatureStack:
    """Stack scalar transformation models into a feature map."""

    def __init__(self, models):
        self.models = models

    def predict(self, inputs):
        return np.column_stack(
            [model.predict(inputs) for model in self.models]
        )


def _fit_tree(
    inputs,
    targets,
    max_depth,
    min_samples_leaf,
    seed,
    previous_models=(),
):
    learner = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    learner.fit(inputs, targets)
    fitted_values = learner.predict(inputs)
    mean = fitted_values.mean()
    centered_values = fitted_values - mean

    if previous_models:
        previous_values = np.column_stack(
            [model.predict(inputs) for model in previous_models]
        )
        gram = previous_values.T @ previous_values / len(inputs)
        covariance = previous_values.T @ centered_values / len(inputs)
        projection = np.linalg.solve(
            gram + 1e-10 * np.eye(len(previous_models)),
            covariance,
        )
        centered_values -= previous_values @ projection
    else:
        projection = np.empty(0)

    scale = np.sqrt(np.mean(centered_values**2))

    if scale <= 1e-12:
        raise np.linalg.LinAlgError(
            "ACE produced a constant conditional-mean estimate."
        )

    if previous_models:
        return _OrthogonalizedTree(
            learner,
            tuple(previous_models),
            projection,
            mean,
            scale,
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


def fit_ace_components(
    x,
    y,
    rank=2,
    n_iterations=20,
    max_depth=3,
    min_samples_leaf=20,
    seed=0,
):
    """Fit multiple ACE components sequentially with empirical deflation."""
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    rng = np.random.default_rng(seed)
    phi_models = []
    psi_models = []
    singular_values = []
    component_histories = []

    for component in range(rank):
        component_seed = seed + component * (2 * n_iterations + 3)
        psi_model = _fit_tree(
            y,
            rng.normal(size=len(y)),
            max_depth,
            min_samples_leaf,
            component_seed,
            psi_models,
        )
        correlation_history = []

        for iteration in range(n_iterations):
            phi_model = _fit_tree(
                x,
                psi_model.predict(y),
                max_depth,
                min_samples_leaf,
                component_seed + 2 * iteration + 1,
                phi_models,
            )
            psi_model = _fit_tree(
                y,
                phi_model.predict(x),
                max_depth,
                min_samples_leaf,
                component_seed + 2 * iteration + 2,
                psi_models,
            )
            phi = phi_model.predict(x)
            psi = psi_model.predict(y)
            correlation_history.append(np.mean(phi * psi))

        phi_models.append(phi_model)
        psi_models.append(psi_model)
        singular_values.append(correlation_history[-1])
        component_histories.append(
            {"correlation": np.asarray(correlation_history)}
        )

    singular_values = np.asarray(singular_values)
    order = np.argsort(singular_values)[::-1]
    phi_model = _FeatureStack([phi_models[index] for index in order])
    psi_model = _FeatureStack([psi_models[index] for index in order])
    history = {
        "components": [component_histories[index] for index in order]
    }
    return phi_model, psi_model, singular_values[order], history

"""Functional Spectral-Newton Method with configurable weak learners."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor


class _LearnerExpansion:
    """Linear expansion of vector-valued regression learners."""

    def __init__(self, rank, terms):
        self.rank = rank
        self.terms = terms

    def predict(self, inputs):
        inputs = np.asarray(inputs).reshape(len(inputs), -1)
        prediction = np.zeros((len(inputs), self.rank))
        for learner, coefficient in self.terms:
            prediction += learner.predict(inputs) @ coefficient
        return prediction

    def update(self, learner, step_size, newton_matrix):
        terms = [
            (base_learner, (1 - step_size) * coefficient)
            for base_learner, coefficient in self.terms
        ]
        terms.append((learner, step_size * newton_matrix))
        return _LearnerExpansion(self.rank, terms)

    def transform(self, matrix):
        return _LearnerExpansion(
            self.rank,
            [
                (learner, coefficient @ matrix)
                for learner, coefficient in self.terms
            ],
        )


def _fit_learner(
    inputs,
    targets,
    learner_type,
    max_depth,
    min_samples_leaf,
    n_knots,
    spline_degree,
    learner_ridge,
    seed,
):
    if learner_type == "tree":
        learner = MultiOutputRegressor(
            DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
            )
        )
    elif learner_type == "linear_spline":
        learner = make_pipeline(
            SplineTransformer(
                n_knots=n_knots,
                degree=spline_degree,
                include_bias=False,
            ),
            Ridge(alpha=learner_ridge),
        )
    else:
        raise ValueError(
            "learner_type must be either 'tree' or 'linear_spline'."
        )
    learner.fit(inputs, targets)
    return learner


def _balance(phi_model, psi_model, phi, psi):
    """Balance the factors without changing their inner product."""
    n_samples = len(phi)
    sigma_phi = phi.T @ phi / n_samples
    sigma_psi = psi.T @ psi / n_samples

    eigenvalues, eigenvectors = np.linalg.eigh(sigma_phi)
    if eigenvalues.min() <= 0:
        raise np.linalg.LinAlgError("Sigma_phi is not positive definite.")

    root_phi = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root_phi = (
        eigenvectors * (1 / np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    middle = root_phi @ sigma_psi @ root_phi
    singular_values_squared, rotation = np.linalg.eigh(
        (middle + middle.T) / 2
    )

    if singular_values_squared.min() <= 0:
        raise np.linalg.LinAlgError("Sigma_psi is not positive definite.")

    order = np.argsort(singular_values_squared)[::-1]
    singular_values_squared = singular_values_squared[order]
    rotation = rotation[:, order]
    transform = (
        np.diag(singular_values_squared**0.25)
        @ rotation.T
        @ inverse_root_phi
    )

    phi_model = phi_model.transform(transform.T)
    psi_model = psi_model.transform(np.linalg.inv(transform))
    return phi_model, psi_model


def empirical_loss(phi, psi):
    """Empirical FSNM loss, up to its constant term."""
    phi = np.asarray(phi).reshape(len(phi), -1)
    psi = np.asarray(psi).reshape(len(psi), -1)
    n_samples = len(phi)
    sigma_phi = phi.T @ phi / n_samples
    sigma_psi = psi.T @ psi / n_samples
    quadratic = np.trace(sigma_phi @ sigma_psi)
    product_of_means = phi.mean(axis=0) @ psi.mean(axis=0)
    paired_product = np.mean(np.sum(phi * psi, axis=1))
    return quadratic + 2 * product_of_means - 2 * paired_product


def fit_fsnm(
    x,
    y,
    rank=2,
    n_iterations=50,
    step_size=0.5,
    max_depth=3,
    min_samples_leaf=20,
    learner_type="tree",
    n_knots=10,
    spline_degree=3,
    learner_ridge=1e-3,
    seed=0,
    validation_data=None,
    patience=None,
    ridge=1e-8,
):
    """Fit FSNM and return singular functions, singular values, and history."""
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    n_samples = len(x)
    identity = np.eye(rank)
    rng = np.random.default_rng(seed)

    phi_learner = _fit_learner(
        x,
        rng.normal(size=(n_samples, rank)),
        learner_type,
        max_depth,
        min_samples_leaf,
        n_knots,
        spline_degree,
        learner_ridge,
        seed,
    )
    psi_learner = _fit_learner(
        y,
        rng.normal(size=(n_samples, rank)),
        learner_type,
        max_depth,
        min_samples_leaf,
        n_knots,
        spline_degree,
        learner_ridge,
        seed + 1,
    )
    phi_model = _LearnerExpansion(rank, [(phi_learner, identity)])
    psi_model = _LearnerExpansion(rank, [(psi_learner, identity)])

    interaction_history = []
    training_loss_history = []
    validation_history = []
    best_loss = np.inf
    best_models = None
    stale_iterations = 0

    if validation_data is not None:
        x_validation, y_validation = validation_data
        x_validation = np.asarray(x_validation).reshape(
            len(x_validation), -1
        )
        y_validation = np.asarray(y_validation).reshape(
            len(y_validation), -1
        )

    for iteration in range(n_iterations):
        phi = phi_model.predict(x)
        sigma_phi = phi.T @ phi / n_samples + ridge * identity
        conditional_phi = _fit_learner(
            y,
            phi - phi.mean(axis=0),
            learner_type,
            max_depth,
            min_samples_leaf,
            n_knots,
            spline_degree,
            learner_ridge,
            seed + 2 * iteration + 2,
        )
        psi_model = psi_model.update(
            conditional_phi,
            step_size,
            np.linalg.inv(sigma_phi),
        )

        psi = psi_model.predict(y)
        sigma_psi = psi.T @ psi / n_samples + ridge * identity
        conditional_psi = _fit_learner(
            x,
            psi - psi.mean(axis=0),
            learner_type,
            max_depth,
            min_samples_leaf,
            n_knots,
            spline_degree,
            learner_ridge,
            seed + 2 * iteration + 3,
        )
        phi_model = phi_model.update(
            conditional_psi,
            step_size,
            np.linalg.inv(sigma_psi),
        )

        phi = phi_model.predict(x)
        psi = psi_model.predict(y)
        phi_model, psi_model = _balance(
            phi_model,
            psi_model,
            phi,
            psi,
        )

        phi = phi_model.predict(x)
        psi = psi_model.predict(y)
        sigma_phi = phi.T @ phi / n_samples
        sigma_psi = psi.T @ psi / n_samples
        interaction_history.append(np.trace(sigma_phi @ sigma_psi))
        training_loss_history.append(empirical_loss(phi, psi))

        if validation_data is not None:
            validation_loss = empirical_loss(
                phi_model.predict(x_validation),
                psi_model.predict(y_validation),
            )
            validation_history.append(validation_loss)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_models = (phi_model, psi_model)
                stale_iterations = 0
            else:
                stale_iterations += 1
                if (
                    patience is not None
                    and stale_iterations >= patience
                ):
                    break

    if best_models is not None:
        phi_model, psi_model = best_models

    phi = phi_model.predict(x)
    sigma_phi = phi.T @ phi / n_samples
    singular_values = np.diag(sigma_phi)
    normalization = np.diag(1 / np.sqrt(singular_values))
    phi_model = phi_model.transform(normalization)
    psi_model = psi_model.transform(normalization)

    validation_history = np.asarray(validation_history)
    history = {
        "interaction": np.asarray(interaction_history),
        "training_loss": np.asarray(training_loss_history),
        "validation_loss": validation_history,
        "best_iteration": (
            int(np.argmin(validation_history)) + 1
            if len(validation_history)
            else len(interaction_history)
        ),
    }
    return phi_model, psi_model, singular_values, history

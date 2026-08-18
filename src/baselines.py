"""Baseline estimators compared against FSNM on the synthetic experiments.

Three methods already discussed in the paper's related work are implemented
here:

* ACE (Alternating Conditional Expectations, Breiman and Friedman 1985),
  extended to rank ``d`` by fitting successive components with empirical
  deflation against previously fitted factors.
* uLSIF (unconstrained Least-Squares Importance Fitting, Kanamori et al.
  2009), which targets the full density ratio pointwise with a Gaussian
  kernel model instead of a low-rank factorization.
* Regularized kernel CCA (Bach and Jordan 2002), solved on a random landmark
  subsample for tractability, with out-of-sample projections through the
  usual centered-kernel expansion.

Where a method produces two factor maps (ACE, kernel CCA), the returned
``phi_model``/``psi_model`` objects expose the same ``predict(inputs) ->
(n, rank)`` interface as :func:`fsnm.fit_fsnm`, so the existing evaluation
code (kernel reconstruction, subspace error, orthogonality error) applies
unchanged. uLSIF does not factorize and instead exposes direct pointwise and
outer-grid kernel evaluators.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# ACE
# ---------------------------------------------------------------------------


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
    """Regression tree centered, deflated against prior components, and
    normalized on training data."""

    def __init__(self, learner, previous_models, projection, mean, scale):
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
    """Stack scalar transformation models into a ``(n, rank)`` feature map."""

    def __init__(self, models):
        self.models = models

    def predict(self, inputs):
        return np.column_stack(
            [model.predict(inputs) for model in self.models]
        )


def _fit_ace_tree(
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
            learner, tuple(previous_models), projection, mean, scale
        )
    return _StandardizedTree(learner, mean, scale)


def fit_ace(
    x,
    y,
    rank=2,
    n_iterations=20,
    max_depth=3,
    min_samples_leaf=20,
    seed=0,
    validation_data=None,
    patience=None,
):
    """Fit rank-``d`` ACE by sequential components with empirical deflation.

    Each component alternates conditional-mean regressions as in classical
    ACE. When ``validation_data`` is given, the iteration within each
    component is selected by validation correlation (mirroring FSNM's own
    validation-based iteration selection), with optional early stopping via
    ``patience``.
    """
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    rng = np.random.default_rng(seed)

    if validation_data is not None:
        x_validation, y_validation = validation_data
        x_validation = np.asarray(x_validation).reshape(
            len(x_validation), -1
        )
        y_validation = np.asarray(y_validation).reshape(
            len(y_validation), -1
        )

    phi_models, psi_models = [], []
    singular_values = []
    component_histories = []

    for component in range(rank):
        component_seed = seed + component * (2 * n_iterations + 3)
        psi_model = _fit_ace_tree(
            y,
            rng.normal(size=len(y)),
            max_depth,
            min_samples_leaf,
            component_seed,
            psi_models,
        )
        train_correlation_history = []
        validation_correlation_history = []
        best_validation_correlation = -np.inf
        best_pair = None
        stale_iterations = 0

        for iteration in range(n_iterations):
            phi_model = _fit_ace_tree(
                x,
                psi_model.predict(y),
                max_depth,
                min_samples_leaf,
                component_seed + 2 * iteration + 1,
                phi_models,
            )
            psi_model = _fit_ace_tree(
                y,
                phi_model.predict(x),
                max_depth,
                min_samples_leaf,
                component_seed + 2 * iteration + 2,
                psi_models,
            )
            train_correlation_history.append(
                float(np.mean(phi_model.predict(x) * psi_model.predict(y)))
            )

            if validation_data is not None:
                validation_correlation = float(
                    np.mean(
                        phi_model.predict(x_validation)
                        * psi_model.predict(y_validation)
                    )
                )
                validation_correlation_history.append(validation_correlation)
                if validation_correlation > best_validation_correlation:
                    best_validation_correlation = validation_correlation
                    best_pair = (phi_model, psi_model)
                    stale_iterations = 0
                else:
                    stale_iterations += 1
                    if patience is not None and stale_iterations >= patience:
                        break

        if validation_data is not None and best_pair is not None:
            phi_model, psi_model = best_pair

        phi_models.append(phi_model)
        psi_models.append(psi_model)
        singular_values.append(
            float(np.mean(phi_model.predict(x) * psi_model.predict(y)))
        )
        component_histories.append(
            {
                "train_correlation": np.asarray(train_correlation_history),
                "validation_correlation": np.asarray(
                    validation_correlation_history
                ),
            }
        )

    singular_values = np.asarray(singular_values)
    order = np.argsort(singular_values)[::-1]
    phi_model = _FeatureStack([phi_models[index] for index in order])
    psi_model = _FeatureStack([psi_models[index] for index in order])
    history = {
        "components": [component_histories[index] for index in order]
    }
    return phi_model, psi_model, singular_values[order], history


# ---------------------------------------------------------------------------
# uLSIF
# ---------------------------------------------------------------------------


class _StandardScaler:
    def __init__(self, values):
        self.mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        self.std = np.where(std > 1e-12, std, 1.0)

    def transform(self, values):
        return (values - self.mean) / self.std


def _pairwise_sq_dists(a, b):
    a2 = np.sum(a**2, axis=1, keepdims=True)
    b2 = np.sum(b**2, axis=1, keepdims=True)
    return np.maximum(a2 + b2.T - 2 * a @ b.T, 0.0)


class _ULSIFModel:
    """Fitted uLSIF density-ratio model with pointwise/grid evaluation."""

    def __init__(self, centers_x, centers_y, x_scaler, y_scaler, bandwidth, coefficients):
        self.centers_x = centers_x
        self.centers_y = centers_y
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.bandwidth = bandwidth
        self.coefficients = coefficients

    def _basis(self, x, y):
        x = self.x_scaler.transform(np.asarray(x).reshape(len(x), -1))
        y = self.y_scaler.transform(np.asarray(y).reshape(len(y), -1))
        sq_dist = _pairwise_sq_dists(x, self.centers_x) + _pairwise_sq_dists(
            y, self.centers_y
        )
        return np.exp(-sq_dist / (2 * self.bandwidth**2))

    def predict_pairs(self, x, y):
        """Evaluate kappa-hat at paired observations (x_i, y_i)."""
        x = np.asarray(x).reshape(len(x), -1)
        y = np.asarray(y).reshape(len(y), -1)
        n = len(x)
        x_scaled = self.x_scaler.transform(x)
        y_scaled = self.y_scaler.transform(y)
        sq_dist = np.sum(
            (x_scaled[:, None, :] - self.centers_x[None, :, :]) ** 2,
            axis=2,
        ) + np.sum(
            (y_scaled[:, None, :] - self.centers_y[None, :, :]) ** 2,
            axis=2,
        )
        basis = np.exp(-sq_dist / (2 * self.bandwidth**2))
        return basis @ self.coefficients

    def predict_grid(self, x_grid, y_grid):
        """Evaluate kappa-hat on the outer product of two 1-D grids."""
        x_grid = np.asarray(x_grid).reshape(-1, 1)
        y_grid = np.asarray(y_grid).reshape(-1, 1)
        x_scaled = self.x_scaler.transform(x_grid)
        y_scaled = self.y_scaler.transform(y_grid)
        sq_dist_x = _pairwise_sq_dists(x_scaled, self.centers_x)
        sq_dist_y = _pairwise_sq_dists(y_scaled, self.centers_y)
        basis_x = np.exp(-sq_dist_x / (2 * self.bandwidth**2))
        basis_y = np.exp(-sq_dist_y / (2 * self.bandwidth**2))
        # sum_l theta_l * basis_x[i, l] * basis_y[j, l] as an outer product
        return basis_x @ (self.coefficients[:, None] * basis_y.T)


def _ulsif_theta(basis_numerator, basis_denominator, ridge):
    h_hat = basis_numerator.mean(axis=0)
    hessian = basis_denominator.T @ basis_denominator / len(basis_denominator)
    theta = np.linalg.solve(
        hessian + ridge * np.eye(hessian.shape[0]), h_hat
    )
    return np.maximum(theta, 0.0)


def _ulsif_objective(theta, basis_numerator, basis_denominator):
    hessian = basis_denominator.T @ basis_denominator / len(basis_denominator)
    h_hat = basis_numerator.mean(axis=0)
    return 0.5 * theta @ hessian @ theta - h_hat @ theta


def fit_ulsif(
    x,
    y,
    validation_data=None,
    n_basis=300,
    bandwidth_grid=(0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5),
    ridge_grid=(1e-4, 1e-3, 1e-2, 1e-1),
    seed=0,
):
    """Fit uLSIF (Kanamori et al. 2009): a direct pointwise estimate of the
    density ratio using a Gaussian kernel model, with basis centers sampled
    from the numerator (joint) training pairs.

    Denominator (product-of-marginals) pairs are formed by independently
    permuting the response coordinate of the training sample. When
    ``validation_data`` is given, ``bandwidth`` and ``ridge`` are selected by
    the (up to constant) unbiased squared-error criterion evaluated on an
    analogous held-out numerator/denominator split, mirroring how FSNM
    selects hyperparameters by validation loss.
    """
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    rng = np.random.default_rng(seed)

    x_scaler = _StandardScaler(x)
    y_scaler = _StandardScaler(y)
    x_scaled = x_scaler.transform(x)
    y_scaled = y_scaler.transform(y)

    n_basis = min(n_basis, len(x))
    center_indices = rng.choice(len(x), size=n_basis, replace=False)
    centers_x = x_scaled[center_indices]
    centers_y = y_scaled[center_indices]

    denominator_y = y_scaled[rng.permutation(len(y))]

    if validation_data is not None:
        x_validation, y_validation = validation_data
        x_validation = x_scaler.transform(
            np.asarray(x_validation).reshape(len(x_validation), -1)
        )
        y_validation = y_scaler.transform(
            np.asarray(y_validation).reshape(len(y_validation), -1)
        )
        validation_denominator_y = y_validation[
            rng.permutation(len(y_validation))
        ]

    def basis_of(x_values, y_values, bandwidth):
        sq_dist = _pairwise_sq_dists(x_values, centers_x) + _pairwise_sq_dists(
            y_values, centers_y
        )
        return np.exp(-sq_dist / (2 * bandwidth**2))

    search_results = []
    for bandwidth in bandwidth_grid:
        basis_numerator = basis_of(x_scaled, y_scaled, bandwidth)
        basis_denominator = basis_of(x_scaled, denominator_y, bandwidth)
        for ridge in ridge_grid:
            theta = _ulsif_theta(basis_numerator, basis_denominator, ridge)
            if validation_data is not None:
                basis_val_numerator = basis_of(
                    x_validation, y_validation, bandwidth
                )
                basis_val_denominator = basis_of(
                    x_validation, validation_denominator_y, bandwidth
                )
                score = _ulsif_objective(
                    theta, basis_val_numerator, basis_val_denominator
                )
            else:
                score = _ulsif_objective(
                    theta, basis_numerator, basis_denominator
                )
            search_results.append((score, bandwidth, ridge, theta))

    best_score, best_bandwidth, best_ridge, best_theta = min(
        search_results, key=lambda result: result[0]
    )
    history = {
        "search_results": [
            {
                "bandwidth": bandwidth,
                "ridge": ridge,
                "objective": float(score),
            }
            for score, bandwidth, ridge, _ in search_results
        ],
        "selected_bandwidth": best_bandwidth,
        "selected_ridge": best_ridge,
        "selected_objective": float(best_score),
    }
    model = _ULSIFModel(
        centers_x, centers_y, x_scaler, y_scaler, best_bandwidth, best_theta
    )
    return model, history


# ---------------------------------------------------------------------------
# Kernel CCA
# ---------------------------------------------------------------------------


class _KernelFeatureMap:
    """Out-of-sample projection through a centered Gaussian-kernel expansion."""

    def __init__(self, landmarks, scaler, bandwidth, row_mean, grand_mean, coefficients):
        self.landmarks = landmarks
        self.scaler = scaler
        self.bandwidth = bandwidth
        self.row_mean = row_mean
        self.grand_mean = grand_mean
        self.coefficients = coefficients

    def predict(self, inputs):
        inputs = self.scaler.transform(
            np.asarray(inputs).reshape(len(inputs), -1)
        )
        sq_dist = _pairwise_sq_dists(inputs, self.landmarks)
        raw = np.exp(-sq_dist / (2 * self.bandwidth**2))
        centered = raw - raw.mean(axis=1, keepdims=True) - self.row_mean[None, :] + self.grand_mean
        return centered @ self.coefficients


def _center_kernel(raw_kernel):
    row_mean = raw_kernel.mean(axis=0)
    grand_mean = raw_kernel.mean()
    centered = raw_kernel - row_mean[None, :] - row_mean[:, None] + grand_mean
    return centered, row_mean, grand_mean


def _fit_kernel_cca_once(
    x_landmarks, y_landmarks, x_scaler, y_scaler, bandwidth_x, bandwidth_y, rank, reg
):
    raw_kx = np.exp(
        -_pairwise_sq_dists(x_landmarks, x_landmarks) / (2 * bandwidth_x**2)
    )
    raw_ky = np.exp(
        -_pairwise_sq_dists(y_landmarks, y_landmarks) / (2 * bandwidth_y**2)
    )
    kx, row_mean_x, grand_mean_x = _center_kernel(raw_kx)
    ky, row_mean_y, grand_mean_y = _center_kernel(raw_ky)
    m = len(x_landmarks)

    zero = np.zeros((m, m))
    a_matrix = np.block([[zero, kx @ ky], [ky @ kx, zero]])
    b_matrix = np.block(
        [
            [kx @ kx + reg * kx, zero],
            [zero, ky @ ky + reg * ky],
        ]
    )
    b_matrix = (b_matrix + b_matrix.T) / 2 + 1e-8 * np.eye(2 * m)

    eigenvalues, eigenvectors = eigh(a_matrix, b_matrix)
    order = np.argsort(eigenvalues)[::-1]

    alphas, betas, canonical_correlations = [], [], []
    for index in order[:rank]:
        vector = eigenvectors[:, index]
        alpha, beta = vector[:m], vector[m:]

        f_landmarks = kx @ alpha
        g_landmarks = ky @ beta
        f_std = np.sqrt(np.mean(f_landmarks**2))
        g_std = np.sqrt(np.mean(g_landmarks**2))
        if f_std <= 1e-12 or g_std <= 1e-12:
            continue
        alpha = alpha / f_std
        beta = beta / g_std
        f_landmarks = kx @ alpha
        g_landmarks = ky @ beta
        correlation = float(np.mean(f_landmarks * g_landmarks))
        if correlation < 0:
            beta = -beta
            correlation = -correlation

        alphas.append(alpha)
        betas.append(beta)
        canonical_correlations.append(correlation)

    phi_model = _KernelFeatureMap(
        x_landmarks,
        x_scaler,
        bandwidth_x,
        row_mean_x,
        grand_mean_x,
        np.column_stack(alphas),
    )
    psi_model = _KernelFeatureMap(
        y_landmarks,
        y_scaler,
        bandwidth_y,
        row_mean_y,
        grand_mean_y,
        np.column_stack(betas),
    )
    return phi_model, psi_model, np.asarray(canonical_correlations)


def fit_kernel_cca(
    x,
    y,
    rank=2,
    n_landmarks=1000,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
    reg_grid=(1e-2, 1e-1),
    validation_data=None,
    seed=0,
):
    """Fit regularized kernel CCA (Bach and Jordan 2002) on a random landmark
    subsample of the training data, with Gaussian kernels on each side and
    out-of-sample projection through the centered-kernel expansion.

    The median pairwise distance on the landmark set sets a reference
    bandwidth per side; ``bandwidth_multipliers`` and ``reg_grid`` are
    searched and, when ``validation_data`` is given, selected by the mean
    validation canonical correlation across the ``rank`` components
    (mirroring FSNM's own validation-based selection, using this method's
    natural objective).
    """
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    rng = np.random.default_rng(seed)

    x_scaler = _StandardScaler(x)
    y_scaler = _StandardScaler(y)
    x_scaled = x_scaler.transform(x)
    y_scaled = y_scaler.transform(y)

    n_landmarks = min(n_landmarks, len(x))
    landmark_indices = rng.choice(len(x), size=n_landmarks, replace=False)
    x_landmarks = x_scaled[landmark_indices]
    y_landmarks = y_scaled[landmark_indices]

    def median_bandwidth(values):
        sq_dist = _pairwise_sq_dists(values, values)
        upper = sq_dist[np.triu_indices(len(values), k=1)]
        return float(np.sqrt(np.median(upper[upper > 0])))

    reference_bandwidth_x = median_bandwidth(x_landmarks)
    reference_bandwidth_y = median_bandwidth(y_landmarks)

    if validation_data is not None:
        x_validation, y_validation = validation_data
        x_validation = np.asarray(x_validation).reshape(
            len(x_validation), -1
        )
        y_validation = np.asarray(y_validation).reshape(
            len(y_validation), -1
        )

    search_results = []
    for multiplier in bandwidth_multipliers:
        bandwidth_x = multiplier * reference_bandwidth_x
        bandwidth_y = multiplier * reference_bandwidth_y
        for reg in reg_grid:
            phi_model, psi_model, correlations = _fit_kernel_cca_once(
                x_landmarks,
                y_landmarks,
                x_scaler,
                y_scaler,
                bandwidth_x,
                bandwidth_y,
                rank,
                reg,
            )
            if len(correlations) < rank:
                continue
            if validation_data is not None:
                score = float(
                    np.mean(
                        phi_model.predict(x_validation)
                        * psi_model.predict(y_validation)
                    )
                )
            else:
                score = float(correlations.mean())
            search_results.append(
                (score, multiplier, reg, phi_model, psi_model, correlations)
            )

    (
        best_score,
        best_multiplier,
        best_reg,
        phi_model,
        psi_model,
        singular_values,
    ) = max(search_results, key=lambda result: result[0])
    history = {
        "search_results": [
            {"multiplier": multiplier, "reg": reg, "score": score}
            for score, multiplier, reg, *_ in search_results
        ],
        "selected_multiplier": best_multiplier,
        "selected_reg": best_reg,
        "selected_score": best_score,
    }
    return phi_model, psi_model, singular_values, history

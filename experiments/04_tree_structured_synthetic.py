"""Synthetic settings tailored to tree weak learners."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fsnm import fit_fsnm


RANK = 3
REGION_SIGMAS = np.array([0.35, 0.25, 0.15])
TABULAR_SIGMAS = np.array([0.30, 0.22, 0.14])
N_TRAIN = 8_000
N_VALIDATION = 3_000
N_EVALUATION = 10_000
N_FEATURES = 20
N_RELEVANT = 3
SEED = 2026


def region_basis(values):
    """Centered orthonormal step functions on four equal-width regions."""
    values = np.asarray(values).reshape(-1)
    regions = np.digitize(values, [-0.5, 0.0, 0.5])
    hadamard_contrasts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    return hadamard_contrasts[regions]


def tabular_basis(inputs):
    """Three orthonormal contrasts involving only the first three columns."""
    inputs = np.asarray(inputs).reshape(len(inputs), -1)
    signs = np.where(inputs[:, :N_RELEVANT] >= 0, 1.0, -1.0)
    return signs


def pointwise_density_ratio(phi, psi, singular_values):
    return 1 + np.sum(phi * singular_values * psi, axis=1)


def sample_joint(size, dimension, x_basis, singular_values, seed):
    """Rejection sample from a known density ratio with uniform marginals."""
    rng = np.random.default_rng(seed)
    upper_bound = 1 + singular_values.sum()
    x_parts = []
    y_parts = []
    n_accepted = 0
    while n_accepted < size:
        batch_size = max(size, size - n_accepted)
        x = rng.uniform(-1, 1, size=(batch_size, dimension))
        y = rng.uniform(-1, 1, size=batch_size)
        density_ratio = pointwise_density_ratio(
            x_basis(x), region_basis(y), singular_values
        )
        accepted = rng.uniform(size=batch_size) < density_ratio / upper_bound
        x_parts.append(x[accepted])
        y_parts.append(y[accepted])
        n_accepted += accepted.sum()
    return np.concatenate(x_parts)[:size], np.concatenate(y_parts)[:size]


def fit_tree_model(
    x_train,
    y_train,
    x_validation,
    y_validation,
    max_depth,
    min_samples_leaf,
):
    return fit_fsnm(
        x_train,
        y_train,
        rank=RANK,
        n_iterations=40,
        step_size=0.15,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        seed=0,
        validation_data=(x_validation, y_validation),
    )


def estimated_pointwise_kernel(model, x, y):
    phi_model, psi_model, singular_values, _ = model
    phi = phi_model.predict(x)
    psi = psi_model.predict(np.asarray(y).reshape(-1, 1))
    return pointwise_density_ratio(phi, psi, singular_values)


def subspace_error(estimated, exact):
    estimated = estimated - estimated.mean(axis=0)
    exact = exact - exact.mean(axis=0)
    estimated_basis = np.linalg.qr(estimated)[0][:, : exact.shape[1]]
    exact_basis = np.linalg.qr(exact)[0][:, : exact.shape[1]]
    estimated_projection = estimated_basis @ estimated_basis.T
    exact_projection = exact_basis @ exact_basis.T
    return np.linalg.norm(
        estimated_projection - exact_projection, ord="fro"
    ) / np.sqrt(2 * exact.shape[1])


def evaluate_model(model, x, y, x_basis, singular_values):
    exact = pointwise_density_ratio(
        x_basis(x), region_basis(y), singular_values
    )
    estimated = estimated_pointwise_kernel(model, x, y)
    phi = model[0].predict(x)
    psi = model[1].predict(y[:, None])
    return {
        "kernel_rmse": float(np.sqrt(np.mean((estimated - exact) ** 2))),
        "spectrum_error": float(np.linalg.norm(model[2] - singular_values)),
        "subspace_error": 0.5
        * (
            subspace_error(phi, x_basis(x))
            + subspace_error(psi, region_basis(y))
        ),
        "exact_kernel": exact,
        "estimated_kernel": estimated,
    }


def run_region_experiment(figure_directory):
    x_train, y_train = sample_joint(
        N_TRAIN, 1, region_basis, REGION_SIGMAS, SEED
    )
    x_validation, y_validation = sample_joint(
        N_VALIDATION, 1, region_basis, REGION_SIGMAS, SEED + 1
    )
    model = fit_tree_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        max_depth=3,
        min_samples_leaf=120,
    )

    grid = np.linspace(-1, 1, 240)
    exact_kernel = 1 + (region_basis(grid) * REGION_SIGMAS) @ region_basis(grid).T
    estimated_kernel = 1 + (
        model[0].predict(grid[:, None]) * model[2]
    ) @ model[1].predict(grid[:, None]).T
    error = estimated_kernel - exact_kernel
    metrics = {
        "kernel_rmse": float(np.sqrt(np.mean(error**2))),
        "spectrum_error": float(np.linalg.norm(model[2] - REGION_SIGMAS)),
        "subspace_error": 0.5
        * (
            subspace_error(model[0].predict(grid[:, None]), region_basis(grid))
            + subspace_error(model[1].predict(grid[:, None]), region_basis(grid))
        ),
    }

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.35), constrained_layout=True)
    kernel_limits = (exact_kernel.min(), exact_kernel.max())
    panels = [
        (exact_kernel, r"Exact $\kappa$", "viridis", kernel_limits),
        (estimated_kernel, r"FSNM estimate $\widehat\kappa$", "viridis", kernel_limits),
        (
            error,
            r"Error $\widehat\kappa-\kappa$",
            "coolwarm",
            (-np.abs(error).max(), np.abs(error).max()),
        ),
    ]
    for axis, (values, title, cmap, limits) in zip(axes, panels):
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=(-1, 1, -1, 1),
            aspect="equal",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            interpolation="nearest",
        )
        axis.set(title=title, xlabel="$x$", ylabel="$y$")
        figure.colorbar(image, ax=axis, shrink=0.78)
    path = figure_directory / "09_discontinuous_regions.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return model, metrics, path


def permutation_sensitivity(model, x, y):
    baseline = estimated_pointwise_kernel(model, x, y)
    rng = np.random.default_rng(SEED + 20)
    sensitivities = np.empty(x.shape[1])
    for feature in range(x.shape[1]):
        permuted_x = x.copy()
        permuted_x[:, feature] = permuted_x[
            rng.permutation(len(x)), feature
        ]
        permuted = estimated_pointwise_kernel(model, permuted_x, y)
        sensitivities[feature] = np.sqrt(np.mean((permuted - baseline) ** 2))
    return sensitivities


def run_tabular_experiment(figure_directory):
    x_train, y_train = sample_joint(
        N_TRAIN, N_FEATURES, tabular_basis, TABULAR_SIGMAS, SEED + 10
    )
    x_validation, y_validation = sample_joint(
        N_VALIDATION,
        N_FEATURES,
        tabular_basis,
        TABULAR_SIGMAS,
        SEED + 11,
    )
    model = fit_tree_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        max_depth=3,
        min_samples_leaf=250,
    )

    rng = np.random.default_rng(SEED + 12)
    x_evaluation = rng.uniform(-1, 1, size=(N_EVALUATION, N_FEATURES))
    y_evaluation = rng.uniform(-1, 1, size=N_EVALUATION)
    metrics = evaluate_model(
        model,
        x_evaluation,
        y_evaluation,
        tabular_basis,
        TABULAR_SIGMAS,
    )
    sensitivities = permutation_sensitivity(model, x_evaluation, y_evaluation)
    metrics["relevant_sensitivity"] = float(sensitivities[:N_RELEVANT].mean())
    metrics["irrelevant_sensitivity"] = float(sensitivities[N_RELEVANT:].mean())

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.45), constrained_layout=True)
    axes[0].hexbin(
        metrics["exact_kernel"],
        metrics["estimated_kernel"],
        gridsize=28,
        mincnt=1,
        cmap="Blues",
    )
    limits = [
        min(metrics["exact_kernel"].min(), metrics["estimated_kernel"].min()),
        max(metrics["exact_kernel"].max(), metrics["estimated_kernel"].max()),
    ]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1.2)
    axes[0].set(
        title="Held-out product pairs",
        xlabel=r"Exact $\kappa(x,y)$",
        ylabel=r"Estimated $\widehat\kappa(x,y)$",
        xlim=limits,
        ylim=limits,
    )

    locations = np.arange(1, RANK + 1)
    width = 0.36
    axes[1].bar(
        locations - width / 2,
        TABULAR_SIGMAS,
        width,
        label="Exact",
        color="0.55",
    )
    axes[1].bar(
        locations + width / 2,
        model[2],
        width,
        label="Estimated",
        color="tab:blue",
    )
    axes[1].set(
        title="Dependence spectrum",
        xlabel="Mode",
        ylabel="Singular value",
        xticks=locations,
    )
    axes[1].legend(frameon=False)

    colors = ["tab:red"] * N_RELEVANT + ["0.65"] * (N_FEATURES - N_RELEVANT)
    axes[2].bar(np.arange(1, N_FEATURES + 1), sensitivities, color=colors)
    axes[2].axvline(N_RELEVANT + 0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set(
        title="Permutation sensitivity",
        xlabel="Input feature",
        ylabel=r"RMS change in $\widehat\kappa$",
        xticks=[1, 4, 8, 12, 16, 20],
        xlim=(0.3, N_FEATURES + 0.7),
    )
    path = figure_directory / "10_tabular_irrelevant_features.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return model, metrics, path


def report(name, model, metrics, path):
    print(name)
    print(f"  selected iteration: {model[3]['best_iteration']}")
    print(f"  singular values: {np.round(model[2], 4)}")
    for metric, value in metrics.items():
        if np.isscalar(value):
            print(f"  {metric.replace('_', ' ')}: {value:.4f}")
    print(f"  figure saved to: {path}")


def main():
    project_directory = Path(__file__).resolve().parents[3]
    figure_directory = project_directory / "tex" / "figures"
    figure_directory.mkdir(exist_ok=True)
    region_model, region_metrics, region_path = run_region_experiment(
        figure_directory
    )
    report("Discontinuous regions", region_model, region_metrics, region_path)
    tabular_model, tabular_metrics, tabular_path = run_tabular_experiment(
        figure_directory
    )
    report("Tabular irrelevant features", tabular_model, tabular_metrics, tabular_path)


if __name__ == "__main__":
    main()

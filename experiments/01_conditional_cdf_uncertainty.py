import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fsnm import fit_fsnm


SIGMAS = np.array([0.18, 0.16, 0.12])
RANK = len(SIGMAS)
N_TRAIN = 10_000
X_EVALUATION = np.array([-0.6, 0.0, 0.6])
CDF_GRID = np.linspace(-1, 1, 301)
SEED = 2026

TREE_PARAMETERS = {
    "n_iterations": 11,
    "step_size": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 300,
}

SPLINE_PARAMETERS = {
    "n_iterations": 11,
    "step_size": 0.1,
    "learner_type": "linear_spline",
    "n_knots": 6,
    "spline_degree": 3,
    "learner_ridge": 1e-2,
}


def basis_matrix(values):
    values = np.asarray(values)
    return np.column_stack(
        [
            np.sqrt(2) * np.sin(np.pi * values),
            np.sqrt(2) * np.cos(np.pi * values),
            np.sqrt(2) * np.sin(2 * np.pi * values),
        ]
    )


def kappa_exact(x_values, y_values):
    return 1 + (
        basis_matrix(x_values) * SIGMAS
    ) @ basis_matrix(y_values).T


def sample_joint(size, seed):
    rng = np.random.default_rng(seed)
    upper_bound = 1 + 2 * SIGMAS.sum()
    x_parts = []
    y_parts = []
    n_accepted = 0

    while n_accepted < size:
        x = rng.uniform(-1, 1, size)
        y = rng.uniform(-1, 1, size)
        density_ratio = 1 + np.sum(
            basis_matrix(x) * SIGMAS * basis_matrix(y), axis=1
        )
        accepted = rng.uniform(size=size) < density_ratio / upper_bound
        x_parts.append(x[accepted])
        y_parts.append(y[accepted])
        n_accepted += accepted.sum()

    return np.concatenate(x_parts)[:size], np.concatenate(y_parts)[:size]


def exact_conditional_cdf():
    density = kappa_exact(X_EVALUATION, CDF_GRID)
    increments = (
        0.25
        * (density[:, :-1] + density[:, 1:])
        * np.diff(CDF_GRID)[None, :]
    )
    return np.column_stack(
        [np.zeros(len(X_EVALUATION)), np.cumsum(increments, axis=1)]
    )


def estimated_conditional_cdf(model, y_marginal):
    phi, psi, singular_values = model
    phi_values = phi.predict(X_EVALUATION[:, None])
    psi_values = psi.predict(y_marginal[:, None])
    density_ratio = 1 + (
        phi_values * singular_values
    ) @ psi_values.T
    density_ratio = np.maximum(density_ratio, 0)

    order = np.argsort(y_marginal)
    sorted_y = y_marginal[order]
    sorted_weights = density_ratio[:, order]
    cumulative_weights = np.cumsum(sorted_weights, axis=1)
    cumulative_weights /= cumulative_weights[:, -1, None]

    locations = np.searchsorted(sorted_y, CDF_GRID, side="right") - 1
    cdf = np.zeros((len(X_EVALUATION), len(CDF_GRID)))
    available = locations >= 0
    cdf[:, available] = cumulative_weights[:, locations[available]]
    return cdf


def fit_model(x_train, y_train, parameters, seed):
    phi, psi, singular_values, _ = fit_fsnm(
        x_train,
        y_train,
        rank=RANK,
        seed=seed,
        **parameters,
    )
    return phi, psi, singular_values


def bootstrap_cdfs(
    x_train,
    y_train,
    parameters,
    n_bootstrap,
    rng,
):
    replicates = np.empty(
        (n_bootstrap, len(X_EVALUATION), len(CDF_GRID))
    )
    for bootstrap in range(n_bootstrap):
        training_indices = rng.integers(0, N_TRAIN, size=N_TRAIN)
        bootstrap_y = y_train[training_indices]
        model = fit_model(
            x_train[training_indices],
            bootstrap_y,
            parameters,
            seed=0,
        )
        replicates[bootstrap] = estimated_conditional_cdf(
            model, bootstrap_y
        )
        if (bootstrap + 1) % 25 == 0 or bootstrap + 1 == n_bootstrap:
            print(f"  completed {bootstrap + 1}/{n_bootstrap}")
    return replicates


def summarize_bands(name, exact, lower, upper):
    coverage = np.mean((lower <= exact) & (exact <= upper), axis=1)
    width = np.mean(upper - lower, axis=1)
    print(name)
    for x_value, x_coverage, x_width in zip(
        X_EVALUATION, coverage, width
    ):
        print(
            f"  x={x_value:+.1f}: grid coverage={x_coverage:.3f}, "
            f"mean width={x_width:.4f}"
        )


def save_cdf_figure(
    estimate,
    exact,
    lower,
    upper,
    color,
    figure_path,
):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for column, x_value in enumerate(X_EVALUATION):
        axis = axes[column]
        axis.fill_between(
            CDF_GRID,
            lower[column],
            upper[column],
            color=color,
            alpha=0.22,
            label="Pointwise 95% band",
        )
        axis.plot(
            CDF_GRID,
            exact[column],
            color="black",
            linewidth=2,
            label="Exact",
        )
        axis.plot(
            CDF_GRID,
            estimate[column],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label="Estimate",
        )
        axis.set(
            xlim=(-1, 1),
            ylim=(-0.02, 1.02),
            title=rf"$x={x_value:.1f}$",
            xlabel="$y$",
        )
    axes[0].set_ylabel("Conditional CDF")
    axes[0].legend(frameon=False, loc="upper left")
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {figure_path}")


def main(n_bootstrap):
    x_train, y_train = sample_joint(N_TRAIN, seed=12)

    exact_cdf = exact_conditional_cdf()
    tree_model = fit_model(
        x_train, y_train, TREE_PARAMETERS, seed=0
    )
    spline_model = fit_model(
        x_train, y_train, SPLINE_PARAMETERS, seed=0
    )
    tree_estimate = estimated_conditional_cdf(
        tree_model, y_train
    )
    spline_estimate = estimated_conditional_cdf(
        spline_model, y_train
    )

    print("Tree bootstrap")
    tree_replicates = bootstrap_cdfs(
        x_train,
        y_train,
        TREE_PARAMETERS,
        n_bootstrap,
        np.random.default_rng(SEED),
    )
    print("Spline bootstrap")
    spline_replicates = bootstrap_cdfs(
        x_train,
        y_train,
        SPLINE_PARAMETERS,
        n_bootstrap,
        np.random.default_rng(SEED),
    )

    tree_lower, tree_upper = np.percentile(
        tree_replicates, [2.5, 97.5], axis=0
    )
    spline_lower, spline_upper = np.percentile(
        spline_replicates, [2.5, 97.5], axis=0
    )
    summarize_bands("Trees", exact_cdf, tree_lower, tree_upper)
    summarize_bands("Linear splines", exact_cdf, spline_lower, spline_upper)

    figure_directory = Path(__file__).resolve().parents[1] / "figures"
    figure_directory.mkdir(exist_ok=True)
    save_cdf_figure(
        tree_estimate,
        exact_cdf,
        tree_lower,
        tree_upper,
        "tab:blue",
        figure_directory / "02_tree_conditional_cdf_uncertainty.png",
    )
    save_cdf_figure(
        spline_estimate,
        exact_cdf,
        spline_lower,
        spline_upper,
        "tab:orange",
        figure_directory / "04_linear_conditional_cdf_uncertainty.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=100)
    arguments = parser.parse_args()
    main(arguments.n_bootstrap)

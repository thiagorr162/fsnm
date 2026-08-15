from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fsnm import empirical_loss, fit_fsnm


RANK = 3
N_TRAIN = 5_000
N_VALIDATION = 2_000
N_TEST = 2_000
N_ITERATIONS = 40
NOISE_SCALE = 0.08
N_PERMUTATIONS = 999
SEED = 0

FSNM_PARAMETERS = {
    "rank": RANK,
    "n_iterations": N_ITERATIONS,
    "step_size": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 300,
    "seed": SEED,
}


def sample_case(size, seed, dependent):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size)
    driver = x if dependent else rng.uniform(-1, 1, size)
    y = driver**2 + NOISE_SCALE * rng.normal(size=size)
    return x, y


def scaled_factors(phi_model, psi_model, singular_values, x, y):
    scale = np.sqrt(np.maximum(singular_values, 0))
    phi = phi_model.predict(x[:, None]) * scale
    psi = psi_model.predict(y[:, None]) * scale
    return phi, psi


def fit_case(dependent, train_seed, validation_seed, test_seed):
    x_train, y_train = sample_case(N_TRAIN, train_seed, dependent)
    x_validation, y_validation = sample_case(
        N_VALIDATION, validation_seed, dependent
    )
    x_test, y_test = sample_case(N_TEST, test_seed, dependent)
    phi, psi, singular_values, history = fit_fsnm(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        **FSNM_PARAMETERS,
    )
    phi_test, psi_test = scaled_factors(
        phi, psi, singular_values, x_test, y_test
    )
    test_loss = float(empirical_loss(phi_test, psi_test))
    observed_score = -test_loss
    permutation_rng = np.random.default_rng(test_seed + 10_000)
    null_scores = np.empty(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        permuted_psi = psi_test[permutation_rng.permutation(N_TEST)]
        null_scores[permutation] = -empirical_loss(
            phi_test, permuted_psi
        )
    permutation_p_value = (
        1 + np.count_nonzero(null_scores >= observed_score)
    ) / (N_PERMUTATIONS + 1)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": y_test,
        "phi": phi,
        "psi": psi,
        "singular_values": singular_values,
        "history": history,
        "test_loss": test_loss,
        "dependence_score": observed_score,
        "permutation_p_value": permutation_p_value,
        "pearson": float(np.corrcoef(x_test, y_test)[0, 1]),
    }


def centered_kernel(result, x_grid, y_grid):
    phi = result["phi"].predict(x_grid[:, None])
    psi = result["psi"].predict(y_grid[:, None])
    return (phi * result["singular_values"]) @ psi.T


independent = fit_case(
    False, train_seed=21, validation_seed=22, test_seed=23
)
nonlinear = fit_case(True, train_seed=31, validation_seed=32, test_seed=33)

x_grid = np.linspace(-1, 1, 180)
y_grid = np.linspace(-0.2, 1.2, 180)
independent_kernel = centered_kernel(independent, x_grid, y_grid)
nonlinear_kernel = centered_kernel(nonlinear, x_grid, y_grid)

for name, result, kernel in [
    ("Independent", independent, independent_kernel),
    ("Nonlinear", nonlinear, nonlinear_kernel),
]:
    print(name)
    print(f"  Pearson correlation: {result['pearson']:.4f}")
    print(f"  Test loss: {result['test_loss']:.4f}")
    print(f"  Permutation p-value: {result['permutation_p_value']:.3f}")
    print(f"  Selected iteration: {result['history']['best_iteration']}")
    print(
        "  Singular values: "
        f"{np.round(result['singular_values'], 4)}"
    )
    print(f"  Grid RMS of kappa - 1: {np.sqrt(np.mean(kernel**2)):.4f}")

figure_directory = Path(__file__).resolve().parents[1] / "figures"
figure_directory.mkdir(exist_ok=True)

kernel_limit = max(
    np.abs(independent_kernel).max(), np.abs(nonlinear_kernel).max()
)
spectrum_limit = 1.08 * max(
    independent["singular_values"].max(),
    nonlinear["singular_values"].max(),
)

independent_figure, independent_axes = plt.subplots(
    1, 2, figsize=(8.5, 3.5), constrained_layout=True
)
image = independent_axes[0].imshow(
    independent_kernel.T,
    origin="lower",
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    aspect="auto",
    cmap="coolwarm",
    vmin=-kernel_limit,
    vmax=kernel_limit,
)
independent_axes[0].set(
    title=r"Estimated $\widehat\kappa-1$", xlabel="$x$", ylabel="$y$"
)
independent_figure.colorbar(image, ax=independent_axes[0], shrink=0.85)
independent_axes[1].bar(
    np.arange(1, RANK + 1), independent["singular_values"], color="tab:blue"
)
independent_axes[1].set(
    title=(
        "Estimated singular values\n"
        f"permutation $p={independent['permutation_p_value']:.3f}$"
    ),
    xlabel="Index",
    ylabel="Singular value",
    xticks=np.arange(1, RANK + 1),
    ylim=(0, spectrum_limit),
)
independent_path = figure_directory / "04_independent_case.png"
independent_figure.savefig(independent_path, dpi=200, bbox_inches="tight")

nonlinear_figure, nonlinear_axes = plt.subplots(
    1, 3, figsize=(12, 3.5), constrained_layout=True
)
nonlinear_axes[0].scatter(
    nonlinear["x_test"],
    nonlinear["y_test"],
    s=7,
    alpha=0.18,
    color="tab:blue",
    edgecolors="none",
)
nonlinear_axes[0].set(
    title=f"Pearson correlation: {nonlinear['pearson']:.3f}",
    xlabel="$x$",
    ylabel="$y$",
    xlim=(-1, 1),
    ylim=(y_grid.min(), y_grid.max()),
)
image = nonlinear_axes[1].imshow(
    nonlinear_kernel.T,
    origin="lower",
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    aspect="auto",
    cmap="coolwarm",
    vmin=-kernel_limit,
    vmax=kernel_limit,
)
nonlinear_axes[1].set(
    title=r"Estimated $\widehat\kappa-1$", xlabel="$x$", ylabel="$y$"
)
nonlinear_figure.colorbar(image, ax=nonlinear_axes[1], shrink=0.85)
nonlinear_axes[2].bar(
    np.arange(1, RANK + 1), nonlinear["singular_values"], color="tab:blue"
)
nonlinear_axes[2].set(
    title=(
        "Estimated singular values\n"
        f"permutation $p={nonlinear['permutation_p_value']:.3f}$"
    ),
    xlabel="Index",
    ylabel="Singular value",
    xticks=np.arange(1, RANK + 1),
    ylim=(0, spectrum_limit),
)
nonlinear_path = figure_directory / "05_nonlinear_dependence.png"
nonlinear_figure.savefig(nonlinear_path, dpi=200, bbox_inches="tight")

print(f"Figure saved to: {independent_path}")
print(f"Figure saved to: {nonlinear_path}")

"""Compare FSNM against ACE, uLSIF, and kernel CCA on approximation error.

This script reuses the exact data-generating processes and selected FSNM
hyperparameters from ``00_rank3_conditional_queries.py`` and
``04_tree_structured_synthetic.py``, and reports only approximation-quality
metrics (kernel RMSE and, where the baseline exposes a factorization,
spectrum/subspace error) for FSNM and the three baselines side by side.
No figures or paper text are touched by this script.
"""

import time

import numpy as np

from baselines import fit_ace, fit_kernel_cca, fit_ulsif
from fsnm import fit_fsnm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def subspace_error(estimated, exact):
    estimated = estimated - estimated.mean(axis=0)
    exact = exact - exact.mean(axis=0)
    estimated_basis = np.linalg.qr(estimated)[0][:, : exact.shape[1]]
    exact_basis = np.linalg.qr(exact)[0][:, : exact.shape[1]]
    difference = (
        estimated_basis @ estimated_basis.T - exact_basis @ exact_basis.T
    )
    return np.linalg.norm(difference, ord="fro") / np.sqrt(2 * exact.shape[1])


def orthogonality_error(values):
    centered = values - values.mean(axis=0)
    gram = centered.T @ centered / len(centered)
    return np.linalg.norm(
        gram - np.eye(gram.shape[0]), ord="fro"
    ) / np.sqrt(gram.shape[0])


def factorized_kernel(phi_model, psi_model, singular_values, x, y):
    phi = phi_model.predict(x)
    psi = psi_model.predict(y)
    return 1 + (phi * singular_values) @ psi.T


def print_table(rows, columns):
    header = f"{'method':<12}" + "".join(f"{c:>18}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, values in rows:
        cells = "".join(
            f"{values[c]:>18.4f}" if values.get(c) is not None else f"{'--':>18}"
            for c in columns
        )
        print(f"{name:<12}{cells}")


# ---------------------------------------------------------------------------
# Part A: rank-three experiment (Sec. 5.1)
# ---------------------------------------------------------------------------


def basis_matrix_rank3(values, rank=3):
    values = np.asarray(values)
    basis = np.column_stack(
        [
            np.sqrt(2) * np.sin(np.pi * values),
            np.sqrt(2) * np.cos(np.pi * values),
            np.sqrt(2) * np.sin(2 * np.pi * values),
        ]
    )
    return basis[:, :rank]


SIGMAS_RANK3 = np.array([0.18, 0.16, 0.12])


def kappa_exact_rank3(x_values, y_values):
    return (
        1
        + (basis_matrix_rank3(x_values) * SIGMAS_RANK3)
        @ basis_matrix_rank3(y_values).T
    )


def sample_joint_rank3(size, seed):
    rng = np.random.default_rng(seed)
    upper_bound = 1 + 2 * SIGMAS_RANK3.sum()
    x_parts, y_parts, n_accepted = [], [], 0
    while n_accepted < size:
        x = rng.uniform(-1, 1, size)
        y = rng.uniform(-1, 1, size)
        density_ratio = 1 + np.sum(
            basis_matrix_rank3(x) * SIGMAS_RANK3 * basis_matrix_rank3(y),
            axis=1,
        )
        accepted = rng.uniform(size=size) < density_ratio / upper_bound
        x_parts.append(x[accepted])
        y_parts.append(y[accepted])
        n_accepted += accepted.sum()
    return np.concatenate(x_parts)[:size], np.concatenate(y_parts)[:size]


def run_rank3_comparison():
    print("\n" + "=" * 70)
    print("Rank-three experiment (Sec. 5.1)")
    print("=" * 70)

    rank = 3
    x_train, y_train = sample_joint_rank3(10_000, seed=12)
    x_validation, y_validation = sample_joint_rank3(4_000, seed=99)
    grid = np.linspace(-1, 1, 160)
    exact_basis = basis_matrix_rank3(grid)
    kappa_true = kappa_exact_rank3(grid, grid)

    rows = []

    # FSNM, selected hyperparameters from 00_rank3_conditional_queries.py
    t0 = time.time()
    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, seed=0,
        n_iterations=11, step_size=0.1, max_depth=3, min_samples_leaf=300,
    )
    phi_grid, psi_grid = phi.predict(grid[:, None]), psi.predict(grid[:, None])
    kappa_hat = 1 + (phi_grid * values) @ psi_grid.T
    rows.append((
        "FSNM",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_hat - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values - SIGMAS_RANK3),
            "subspace_error": 0.5 * (
                subspace_error(phi_grid, exact_basis)
                + subspace_error(psi_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))

    # ACE, same tree budget as the FSNM selected configuration
    t0 = time.time()
    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=300, seed=0,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_grid = phi_a.predict(grid[:, None])
    psi_a_grid = psi_a.predict(grid[:, None])
    kappa_a = 1 + (phi_a_grid * values_a) @ psi_a_grid.T
    rows.append((
        "ACE",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_a - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values_a - SIGMAS_RANK3),
            "subspace_error": 0.5 * (
                subspace_error(phi_a_grid, exact_basis)
                + subspace_error(psi_a_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))

    # uLSIF: no factorization, only kernel RMSE
    t0 = time.time()
    model_u, history_u = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation), seed=0,
    )
    kappa_u = model_u.predict_grid(grid, grid)
    rows.append((
        "uLSIF",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_u - kappa_true) ** 2)),
            "spectrum_error": None,
            "subspace_error": None,
            "seconds": time.time() - t0,
        },
    ))
    print(f"  uLSIF selected bandwidth={history_u['selected_bandwidth']}, ridge={history_u['selected_ridge']}")

    # Kernel CCA
    t0 = time.time()
    phi_k, psi_k, values_k, history_k = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=0,
    )
    phi_k_grid = phi_k.predict(grid[:, None])
    psi_k_grid = psi_k.predict(grid[:, None])
    kappa_k = 1 + (phi_k_grid * values_k) @ psi_k_grid.T
    rows.append((
        "Kernel CCA",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_k - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values_k - SIGMAS_RANK3),
            "subspace_error": 0.5 * (
                subspace_error(phi_k_grid, exact_basis)
                + subspace_error(psi_k_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))
    print(f"  Kernel CCA selected multiplier={history_k['selected_multiplier']}, reg={history_k['selected_reg']}")

    print(f"\nTrue singular values: {SIGMAS_RANK3}")
    print_table(rows, ["kernel_rmse", "spectrum_error", "subspace_error", "seconds"])


# ---------------------------------------------------------------------------
# Part B: tree-structured settings (Sec. 5.2)
# ---------------------------------------------------------------------------

REGION_SIGMAS = np.array([0.35, 0.25, 0.15])
TABULAR_SIGMAS = np.array([0.30, 0.22, 0.14])
N_RELEVANT = 3


def region_basis(values):
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
    inputs = np.asarray(inputs).reshape(len(inputs), -1)
    signs = np.where(inputs[:, :N_RELEVANT] >= 0, 1.0, -1.0)
    return signs


def pointwise_density_ratio(phi, psi, singular_values):
    return 1 + np.sum(phi * singular_values * psi, axis=1)


def sample_joint_tree(size, dimension, x_basis, singular_values, seed):
    rng = np.random.default_rng(seed)
    upper_bound = 1 + singular_values.sum()
    x_parts, y_parts, n_accepted = [], [], 0
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


def run_region_comparison():
    print("\n" + "=" * 70)
    print("Discontinuous regional structure (Sec. 5.2.1)")
    print("=" * 70)

    rank = 3
    x_train, y_train = sample_joint_tree(
        8_000, 1, region_basis, REGION_SIGMAS, 2026
    )
    x_validation, y_validation = sample_joint_tree(
        3_000, 1, region_basis, REGION_SIGMAS, 2027
    )
    grid = np.linspace(-1, 1, 240)
    exact_basis = region_basis(grid)
    kappa_true = 1 + (exact_basis * REGION_SIGMAS) @ exact_basis.T

    rows = []

    t0 = time.time()
    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, n_iterations=40, step_size=0.15,
        max_depth=3, min_samples_leaf=120, seed=0,
        validation_data=(x_validation, y_validation),
    )
    phi_grid, psi_grid = phi.predict(grid[:, None]), psi.predict(grid[:, None])
    kappa_hat = 1 + (phi_grid * values) @ psi_grid.T
    rows.append((
        "FSNM",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_hat - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values - REGION_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_grid, exact_basis)
                + subspace_error(psi_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))

    t0 = time.time()
    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=120, seed=0,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_grid = phi_a.predict(grid[:, None])
    psi_a_grid = psi_a.predict(grid[:, None])
    kappa_a = 1 + (phi_a_grid * values_a) @ psi_a_grid.T
    rows.append((
        "ACE",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_a - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values_a - REGION_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_a_grid, exact_basis)
                + subspace_error(psi_a_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))

    t0 = time.time()
    model_u, history_u = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation), seed=0,
    )
    kappa_u = model_u.predict_grid(grid, grid)
    rows.append((
        "uLSIF",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_u - kappa_true) ** 2)),
            "spectrum_error": None,
            "subspace_error": None,
            "seconds": time.time() - t0,
        },
    ))
    print(f"  uLSIF selected bandwidth={history_u['selected_bandwidth']}, ridge={history_u['selected_ridge']}")

    t0 = time.time()
    phi_k, psi_k, values_k, history_k = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=0,
    )
    phi_k_grid = phi_k.predict(grid[:, None])
    psi_k_grid = psi_k.predict(grid[:, None])
    kappa_k = 1 + (phi_k_grid * values_k) @ psi_k_grid.T
    rows.append((
        "Kernel CCA",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_k - kappa_true) ** 2)),
            "spectrum_error": np.linalg.norm(values_k - REGION_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_k_grid, exact_basis)
                + subspace_error(psi_k_grid, exact_basis)
            ),
            "seconds": time.time() - t0,
        },
    ))
    print(f"  Kernel CCA selected multiplier={history_k['selected_multiplier']}, reg={history_k['selected_reg']}")

    print(f"\nTrue singular values: {REGION_SIGMAS}")
    print_table(rows, ["kernel_rmse", "spectrum_error", "subspace_error", "seconds"])


def run_tabular_comparison():
    print("\n" + "=" * 70)
    print("Tabular input with irrelevant features (Sec. 5.2.2)")
    print("=" * 70)

    rank = 3
    n_features = 20
    x_train, y_train = sample_joint_tree(
        8_000, n_features, tabular_basis, TABULAR_SIGMAS, 2036
    )
    x_validation, y_validation = sample_joint_tree(
        3_000, n_features, tabular_basis, TABULAR_SIGMAS, 2037
    )
    rng = np.random.default_rng(2038)
    x_eval = rng.uniform(-1, 1, size=(10_000, n_features))
    y_eval = rng.uniform(-1, 1, size=10_000)
    exact_eval = pointwise_density_ratio(
        tabular_basis(x_eval), region_basis(y_eval), TABULAR_SIGMAS
    )
    exact_basis_eval = tabular_basis(x_eval)
    exact_region_eval = region_basis(y_eval)

    rows = []

    t0 = time.time()
    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, n_iterations=40, step_size=0.15,
        max_depth=3, min_samples_leaf=250, seed=0,
        validation_data=(x_validation, y_validation),
    )
    phi_eval = phi.predict(x_eval)
    psi_eval = psi.predict(y_eval[:, None])
    kappa_hat = pointwise_density_ratio(phi_eval, psi_eval, values)
    rows.append((
        "FSNM",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_hat - exact_eval) ** 2)),
            "spectrum_error": np.linalg.norm(values - TABULAR_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_eval, exact_basis_eval)
                + subspace_error(psi_eval, exact_region_eval)
            ),
            "seconds": time.time() - t0,
        },
    ))

    t0 = time.time()
    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=250, seed=0,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_eval = phi_a.predict(x_eval)
    psi_a_eval = psi_a.predict(y_eval[:, None])
    kappa_a = pointwise_density_ratio(phi_a_eval, psi_a_eval, values_a)
    rows.append((
        "ACE",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_a - exact_eval) ** 2)),
            "spectrum_error": np.linalg.norm(values_a - TABULAR_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_a_eval, exact_basis_eval)
                + subspace_error(psi_a_eval, exact_region_eval)
            ),
            "seconds": time.time() - t0,
        },
    ))

    t0 = time.time()
    model_u, history_u = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation), seed=0,
    )
    kappa_u = model_u.predict_pairs(x_eval, y_eval)
    rows.append((
        "uLSIF",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_u - exact_eval) ** 2)),
            "spectrum_error": None,
            "subspace_error": None,
            "seconds": time.time() - t0,
        },
    ))
    print(f"  uLSIF selected bandwidth={history_u['selected_bandwidth']}, ridge={history_u['selected_ridge']}")

    t0 = time.time()
    phi_k, psi_k, values_k, history_k = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=0,
    )
    phi_k_eval = phi_k.predict(x_eval)
    psi_k_eval = psi_k.predict(y_eval[:, None])
    kappa_k = pointwise_density_ratio(phi_k_eval, psi_k_eval, values_k)
    rows.append((
        "Kernel CCA",
        {
            "kernel_rmse": np.sqrt(np.mean((kappa_k - exact_eval) ** 2)),
            "spectrum_error": np.linalg.norm(values_k - TABULAR_SIGMAS),
            "subspace_error": 0.5 * (
                subspace_error(phi_k_eval, exact_basis_eval)
                + subspace_error(psi_k_eval, exact_region_eval)
            ),
            "seconds": time.time() - t0,
        },
    ))
    print(f"  Kernel CCA selected multiplier={history_k['selected_multiplier']}, reg={history_k['selected_reg']}")

    print(f"\nTrue singular values: {TABULAR_SIGMAS}")
    print_table(rows, ["kernel_rmse", "spectrum_error", "subspace_error", "seconds"])


if __name__ == "__main__":
    run_rank3_comparison()
    run_region_comparison()
    run_tabular_comparison()

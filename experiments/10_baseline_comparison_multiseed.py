"""Multi-seed robustness check for the baseline comparison of 07.

Repeats the rank-three, discontinuous-region, and tabular baseline
comparisons (FSNM vs. ACE, uLSIF, kernel CCA) over N independent data draws,
holding the already-selected hyperparameters fixed and varying only the
data-generation and fitting seed. Replicate 0 reproduces the exact
single-run numbers already reported in the paper (Section 5.1), so this
script also serves as a consistency check on 07_baseline_comparison.py.
"""

import json
from pathlib import Path

import numpy as np

from baselines import fit_ace, fit_kernel_cca, fit_ulsif
from fsnm import fit_fsnm


N_REPLICATES = 10


def subspace_error(estimated, exact):
    estimated = estimated - estimated.mean(axis=0)
    exact = exact - exact.mean(axis=0)
    estimated_basis = np.linalg.qr(estimated)[0][:, : exact.shape[1]]
    exact_basis = np.linalg.qr(exact)[0][:, : exact.shape[1]]
    difference = (
        estimated_basis @ estimated_basis.T - exact_basis @ exact_basis.T
    )
    return np.linalg.norm(difference, ord="fro") / np.sqrt(2 * exact.shape[1])


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


# ---------------------------------------------------------------------------
# Part A: rank-three
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


def run_rank3_replicate(offset):
    rank = 3
    x_train, y_train = sample_joint_rank3(10_000, seed=12 + offset)
    x_validation, y_validation = sample_joint_rank3(4_000, seed=99 + offset)
    grid = np.linspace(-1, 1, 160)
    exact_basis = basis_matrix_rank3(grid)
    kappa_true = kappa_exact_rank3(grid, grid)

    metrics = {}

    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, seed=offset,
        n_iterations=11, step_size=0.1, max_depth=3, min_samples_leaf=300,
    )
    phi_grid, psi_grid = phi.predict(grid[:, None]), psi.predict(grid[:, None])
    kappa_hat = 1 + (phi_grid * values) @ psi_grid.T
    metrics["FSNM"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_hat - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values - SIGMAS_RANK3)),
        "subspace_error": 0.5 * (
            subspace_error(phi_grid, exact_basis)
            + subspace_error(psi_grid, exact_basis)
        ),
    }

    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=300, seed=offset,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_grid = phi_a.predict(grid[:, None])
    psi_a_grid = psi_a.predict(grid[:, None])
    kappa_a = 1 + (phi_a_grid * values_a) @ psi_a_grid.T
    metrics["ACE"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_a - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_a - SIGMAS_RANK3)),
        "subspace_error": 0.5 * (
            subspace_error(phi_a_grid, exact_basis)
            + subspace_error(psi_a_grid, exact_basis)
        ),
    }

    model_u, _ = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation),
        seed=offset,
    )
    kappa_u = model_u.predict_grid(grid, grid)
    metrics["uLSIF"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_u - kappa_true) ** 2))),
    }

    phi_k, psi_k, values_k, _ = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=offset,
    )
    phi_k_grid = phi_k.predict(grid[:, None])
    psi_k_grid = psi_k.predict(grid[:, None])
    kappa_k = 1 + (phi_k_grid * values_k) @ psi_k_grid.T
    metrics["Kernel CCA"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_k - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_k - SIGMAS_RANK3)),
        "subspace_error": 0.5 * (
            subspace_error(phi_k_grid, exact_basis)
            + subspace_error(psi_k_grid, exact_basis)
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# Part B: tree-structured settings
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


def run_region_replicate(offset):
    rank = 3
    x_train, y_train = sample_joint_tree(
        8_000, 1, region_basis, REGION_SIGMAS, 2026 + offset
    )
    x_validation, y_validation = sample_joint_tree(
        3_000, 1, region_basis, REGION_SIGMAS, 2027 + offset
    )
    grid = np.linspace(-1, 1, 240)
    exact_basis = region_basis(grid)
    kappa_true = 1 + (exact_basis * REGION_SIGMAS) @ exact_basis.T

    metrics = {}

    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, n_iterations=40, step_size=0.15,
        max_depth=3, min_samples_leaf=120, seed=offset,
        validation_data=(x_validation, y_validation),
    )
    phi_grid, psi_grid = phi.predict(grid[:, None]), psi.predict(grid[:, None])
    kappa_hat = 1 + (phi_grid * values) @ psi_grid.T
    metrics["FSNM"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_hat - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values - REGION_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_grid, exact_basis)
            + subspace_error(psi_grid, exact_basis)
        ),
    }

    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=120, seed=offset,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_grid = phi_a.predict(grid[:, None])
    psi_a_grid = psi_a.predict(grid[:, None])
    kappa_a = 1 + (phi_a_grid * values_a) @ psi_a_grid.T
    metrics["ACE"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_a - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_a - REGION_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_a_grid, exact_basis)
            + subspace_error(psi_a_grid, exact_basis)
        ),
    }

    model_u, _ = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation),
        seed=offset,
    )
    kappa_u = model_u.predict_grid(grid, grid)
    metrics["uLSIF"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_u - kappa_true) ** 2))),
    }

    phi_k, psi_k, values_k, _ = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=offset,
    )
    phi_k_grid = phi_k.predict(grid[:, None])
    psi_k_grid = psi_k.predict(grid[:, None])
    kappa_k = 1 + (phi_k_grid * values_k) @ psi_k_grid.T
    metrics["Kernel CCA"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_k - kappa_true) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_k - REGION_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_k_grid, exact_basis)
            + subspace_error(psi_k_grid, exact_basis)
        ),
    }
    return metrics


def run_tabular_replicate(offset):
    rank = 3
    n_features = 20
    x_train, y_train = sample_joint_tree(
        8_000, n_features, tabular_basis, TABULAR_SIGMAS, 2036 + offset
    )
    x_validation, y_validation = sample_joint_tree(
        3_000, n_features, tabular_basis, TABULAR_SIGMAS, 2037 + offset
    )
    rng = np.random.default_rng(2038 + offset)
    x_eval = rng.uniform(-1, 1, size=(10_000, n_features))
    y_eval = rng.uniform(-1, 1, size=10_000)
    exact_eval = pointwise_density_ratio(
        tabular_basis(x_eval), region_basis(y_eval), TABULAR_SIGMAS
    )
    exact_basis_eval = tabular_basis(x_eval)
    exact_region_eval = region_basis(y_eval)

    metrics = {}

    phi, psi, values, _ = fit_fsnm(
        x_train, y_train, rank=rank, n_iterations=40, step_size=0.15,
        max_depth=3, min_samples_leaf=250, seed=offset,
        validation_data=(x_validation, y_validation),
    )
    phi_eval = phi.predict(x_eval)
    psi_eval = psi.predict(y_eval[:, None])
    kappa_hat = pointwise_density_ratio(phi_eval, psi_eval, values)
    metrics["FSNM"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_hat - exact_eval) ** 2))),
        "spectrum_error": float(np.linalg.norm(values - TABULAR_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_eval, exact_basis_eval)
            + subspace_error(psi_eval, exact_region_eval)
        ),
    }

    phi_a, psi_a, values_a, _ = fit_ace(
        x_train, y_train, rank=rank, n_iterations=40,
        max_depth=3, min_samples_leaf=250, seed=offset,
        validation_data=(x_validation, y_validation), patience=8,
    )
    phi_a_eval = phi_a.predict(x_eval)
    psi_a_eval = psi_a.predict(y_eval[:, None])
    kappa_a = pointwise_density_ratio(phi_a_eval, psi_a_eval, values_a)
    metrics["ACE"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_a - exact_eval) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_a - TABULAR_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_a_eval, exact_basis_eval)
            + subspace_error(psi_a_eval, exact_region_eval)
        ),
    }

    model_u, _ = fit_ulsif(
        x_train, y_train, validation_data=(x_validation, y_validation),
        seed=offset,
    )
    kappa_u = model_u.predict_pairs(x_eval, y_eval)
    metrics["uLSIF"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_u - exact_eval) ** 2))),
    }

    phi_k, psi_k, values_k, _ = fit_kernel_cca(
        x_train, y_train, rank=rank, n_landmarks=800,
        validation_data=(x_validation, y_validation), seed=offset,
    )
    phi_k_eval = phi_k.predict(x_eval)
    psi_k_eval = psi_k.predict(y_eval[:, None])
    kappa_k = pointwise_density_ratio(phi_k_eval, psi_k_eval, values_k)
    metrics["Kernel CCA"] = {
        "kernel_rmse": float(np.sqrt(np.mean((kappa_k - exact_eval) ** 2))),
        "spectrum_error": float(np.linalg.norm(values_k - TABULAR_SIGMAS)),
        "subspace_error": 0.5 * (
            subspace_error(phi_k_eval, exact_basis_eval)
            + subspace_error(psi_k_eval, exact_region_eval)
        ),
    }
    return metrics


def aggregate(all_replicates):
    methods = all_replicates[0].keys()
    metric_names = set()
    for method in methods:
        metric_names |= set(all_replicates[0][method].keys())
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in metric_names:
            if metric in all_replicates[0][method]:
                values = [rep[method][metric] for rep in all_replicates]
                summary[method][metric] = summarize(values)
    return summary


def main():
    package_directory = Path(__file__).resolve().parents[1]
    results = {}

    for scenario_name, run_replicate in [
        ("rank_three", run_rank3_replicate),
        ("discontinuous_regions", run_region_replicate),
        ("tabular_irrelevant_features", run_tabular_replicate),
    ]:
        print(f"=== {scenario_name} ===", flush=True)
        replicates = []
        for offset in range(N_REPLICATES):
            print(f"  replicate {offset}...", flush=True)
            replicates.append(run_replicate(offset))
        summary = aggregate(replicates)
        results[scenario_name] = {
            "n_replicates": N_REPLICATES,
            "per_replicate": replicates,
            "summary": summary,
        }
        for method, metrics in summary.items():
            line = ", ".join(
                f"{name}={vals['mean']:.4f}+/-{vals['std']:.4f}"
                for name, vals in metrics.items()
            )
            print(f"  {method}: {line}")

    artifact_directory = package_directory / "artifacts"
    artifact_directory.mkdir(exist_ok=True)
    results_path = artifact_directory / "baseline_comparison_multiseed.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()

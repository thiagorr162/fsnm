"""Ablation: does balancing every iteration matter, or only at the end?

Algorithm 1 (as stated in the paper) balances only after the final
iteration. The practical implementation instead balances after every
iteration, as a numerical heuristic. This script quantifies the gap between
the two on the rank-three synthetic experiment, using the same selected
hyperparameters (T=11, step size 0.1, max depth 3, min leaf 300) as
Section 5.1.1, so the "every iteration" row reproduces the numbers already
reported there.
"""

import json
from pathlib import Path

import numpy as np

from fsnm import _LearnerExpansion, _balance, _fit_learner, empirical_loss


def fit_fsnm_custom_balance(
    x,
    y,
    rank,
    n_iterations,
    step_size,
    max_depth,
    min_samples_leaf,
    seed,
    balance_every_iteration,
    validation_data=None,
    ridge=1e-8,
):
    """Same recursion as fsnm.fit_fsnm, but balancing is applied either
    after every iteration or only once, after the final iteration."""
    x = np.asarray(x).reshape(len(x), -1)
    y = np.asarray(y).reshape(len(y), -1)
    n_samples = len(x)
    identity = np.eye(rank)
    rng = np.random.default_rng(seed)

    phi_learner = _fit_learner(
        x, rng.normal(size=(n_samples, rank)), "tree",
        max_depth, min_samples_leaf, 10, 3, 1e-3, seed,
    )
    psi_learner = _fit_learner(
        y, rng.normal(size=(n_samples, rank)), "tree",
        max_depth, min_samples_leaf, 10, 3, 1e-3, seed + 1,
    )
    phi_model = _LearnerExpansion(rank, [(phi_learner, identity)])
    psi_model = _LearnerExpansion(rank, [(psi_learner, identity)])

    training_loss_history = []
    validation_history = []
    max_sigma_cond = 0.0

    if validation_data is not None:
        x_val, y_val = validation_data
        x_val = np.asarray(x_val).reshape(len(x_val), -1)
        y_val = np.asarray(y_val).reshape(len(y_val), -1)

    for iteration in range(n_iterations):
        phi = phi_model.predict(x)
        sigma_phi = phi.T @ phi / n_samples + ridge * identity
        max_sigma_cond = max(max_sigma_cond, float(np.linalg.cond(sigma_phi)))
        conditional_phi = _fit_learner(
            y, phi - phi.mean(axis=0), "tree",
            max_depth, min_samples_leaf, 10, 3, 1e-3, seed + 2 * iteration + 2,
        )
        psi_model = psi_model.update(
            conditional_phi, step_size, np.linalg.inv(sigma_phi)
        )

        psi = psi_model.predict(y)
        sigma_psi = psi.T @ psi / n_samples + ridge * identity
        max_sigma_cond = max(max_sigma_cond, float(np.linalg.cond(sigma_psi)))
        conditional_psi = _fit_learner(
            x, psi - psi.mean(axis=0), "tree",
            max_depth, min_samples_leaf, 10, 3, 1e-3, seed + 2 * iteration + 3,
        )
        phi_model = phi_model.update(
            conditional_psi, step_size, np.linalg.inv(sigma_psi)
        )

        if balance_every_iteration:
            phi = phi_model.predict(x)
            psi = psi_model.predict(y)
            phi_model, psi_model = _balance(phi_model, psi_model, phi, psi)

        phi = phi_model.predict(x)
        psi = psi_model.predict(y)
        training_loss_history.append(float(empirical_loss(phi, psi)))
        if validation_data is not None:
            validation_history.append(float(empirical_loss(
                phi_model.predict(x_val), psi_model.predict(y_val)
            )))

    # Algorithm 1's mandatory final balance, which constructs the returned
    # spectral representation regardless of the schedule above.
    phi = phi_model.predict(x)
    psi = psi_model.predict(y)
    phi_model, psi_model = _balance(phi_model, psi_model, phi, psi)
    phi = phi_model.predict(x)
    sigma_phi = phi.T @ phi / n_samples
    singular_values = np.diag(sigma_phi)
    normalization = np.diag(1 / np.sqrt(singular_values))
    phi_model = phi_model.transform(normalization)
    psi_model = psi_model.transform(normalization)

    return phi_model, psi_model, singular_values, {
        "training_loss": np.asarray(training_loss_history),
        "validation_loss": np.asarray(validation_history),
        "max_sigma_cond": max_sigma_cond,
    }


SIGMAS = np.array([0.18, 0.16, 0.12])
RANK = len(SIGMAS)


def basis_matrix(values, rank=3):
    values = np.asarray(values)
    basis = np.column_stack([
        np.sqrt(2) * np.sin(np.pi * values),
        np.sqrt(2) * np.cos(np.pi * values),
        np.sqrt(2) * np.sin(2 * np.pi * values),
    ])
    return basis[:, :rank]


def kappa_exact(x_values, y_values):
    return 1 + (basis_matrix(x_values) * SIGMAS) @ basis_matrix(y_values).T


def sample_joint(size, seed):
    rng = np.random.default_rng(seed)
    upper_bound = 1 + 2 * SIGMAS.sum()
    x_parts, y_parts, n_accepted = [], [], 0
    while n_accepted < size:
        x = rng.uniform(-1, 1, size)
        y = rng.uniform(-1, 1, size)
        density_ratio = 1 + np.sum(basis_matrix(x) * SIGMAS * basis_matrix(y), axis=1)
        accepted = rng.uniform(size=size) < density_ratio / upper_bound
        x_parts.append(x[accepted])
        y_parts.append(y[accepted])
        n_accepted += accepted.sum()
    return np.concatenate(x_parts)[:size], np.concatenate(y_parts)[:size]


def subspace_error(estimated, exact):
    estimated = estimated - estimated.mean(axis=0)
    exact = exact - exact.mean(axis=0)
    estimated_basis = np.linalg.qr(estimated)[0][:, : exact.shape[1]]
    exact_basis = np.linalg.qr(exact)[0][:, : exact.shape[1]]
    diff = estimated_basis @ estimated_basis.T - exact_basis @ exact_basis.T
    return float(np.linalg.norm(diff, ord="fro") / np.sqrt(2 * exact.shape[1]))


def orthogonality_error(values):
    centered = values - values.mean(axis=0)
    gram = centered.T @ centered / len(centered)
    return float(
        np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro")
        / np.sqrt(gram.shape[0])
    )


def main():
    package_directory = Path(__file__).resolve().parents[1]

    x_train, y_train = sample_joint(10_000, seed=12)
    x_validation, y_validation = sample_joint(4_000, seed=99)
    grid = np.linspace(-1, 1, 160)
    exact_basis = basis_matrix(grid)
    kappa_true = kappa_exact(grid, grid)

    fsnm_params = dict(
        rank=RANK, n_iterations=11, step_size=0.1, max_depth=3,
        min_samples_leaf=300, seed=0,
    )

    results = {}
    for name, balance_every_iteration in [
        ("every_iteration", True),
        ("final_only", False),
    ]:
        phi_model, psi_model, values, history = fit_fsnm_custom_balance(
            x_train, y_train, balance_every_iteration=balance_every_iteration,
            validation_data=(x_validation, y_validation), **fsnm_params,
        )
        phi_grid = phi_model.predict(grid[:, None])
        psi_grid = psi_model.predict(grid[:, None])
        kappa_hat = 1 + (phi_grid * values) @ psi_grid.T
        results[name] = {
            "kernel_rmse": float(np.sqrt(np.mean((kappa_hat - kappa_true) ** 2))),
            "spectrum_error": float(np.linalg.norm(values - SIGMAS)),
            "subspace_error": 0.5 * (
                subspace_error(phi_grid, exact_basis)
                + subspace_error(psi_grid, exact_basis)
            ),
            "orthogonality_error": 0.5 * (
                orthogonality_error(phi_grid) + orthogonality_error(psi_grid)
            ),
            "validation_loss": float(history["validation_loss"][-1]),
            "max_sigma_condition_number": history["max_sigma_cond"],
            "singular_values": values.tolist(),
        }
        print(f"{name}: {results[name]}")

    artifact_directory = package_directory / "artifacts"
    artifact_directory.mkdir(exist_ok=True)
    results_path = artifact_directory / "balancing_ablation.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()

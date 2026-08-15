"""Rank-three kernel estimation and conditional-query experiments."""


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from fsnm import empirical_loss, fit_fsnm


def basis_matrix(values, rank=3):
    values = np.asarray(values)
    basis = np.column_stack(
        [
            np.sqrt(2) * np.sin(np.pi * values),
            np.sqrt(2) * np.cos(np.pi * values),
            np.sqrt(2) * np.sin(2 * np.pi * values),
        ]
    )
    return basis[:, :rank]


def kappa_exact(x_values, y_values):
    return 1 + (basis_matrix(x_values) * SIGMAS) @ basis_matrix(y_values).T


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


def scaled_factors(phi_model, psi_model, singular_values, x, y):
    scale = np.sqrt(np.maximum(singular_values, 0))
    return phi_model.predict(x[:, None]) * scale, psi_model.predict(y[:, None]) * scale


def subspace_error(estimated, exact):
    estimated = estimated - estimated.mean(axis=0)
    exact = exact - exact.mean(axis=0)
    estimated_basis = np.linalg.qr(estimated)[0][:, : exact.shape[1]]
    exact_basis = np.linalg.qr(exact)[0][:, : exact.shape[1]]
    difference = estimated_basis @ estimated_basis.T - exact_basis @ exact_basis.T
    return np.linalg.norm(difference, ord="fro") / np.sqrt(2 * exact.shape[1])


def orthogonality_error(values):
    centered = values - values.mean(axis=0)
    gram = centered.T @ centered / len(centered)
    return np.linalg.norm(gram - np.eye(gram.shape[0]), ord="fro") / np.sqrt(gram.shape[0])


SIGMAS = np.array([0.18, 0.16, 0.12])
RANK = len(SIGMAS)
N_TRAIN = 10_000
N_VALIDATION = 4_000
LOSS_CURVE_ITERATIONS = 40
SEED = 0

x_train, y_train = sample_joint(N_TRAIN, seed=12)
x_validation, y_validation = sample_joint(N_VALIDATION, seed=99)


TUNE_HYPERPARAMETERS = False
FSNM_MAX_ITERATIONS = 40

DEFAULT_FSNM_PARAMETERS = {
    "n_iterations": 11,
    "step_size": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 300,
}
FSNM_PARAMETER_GRID = {
    "step_size": [0.05, 0.1, 0.2],
    "max_depth": [3, None],
    "min_samples_leaf": [100, 300],
}

if TUNE_HYPERPARAMETERS:
    tuning_results = []
    for candidate in ParameterGrid(FSNM_PARAMETER_GRID):
        _, _, _, candidate_history = fit_fsnm(
            x_train, y_train, rank=RANK, seed=SEED,
            n_iterations=FSNM_MAX_ITERATIONS,
            validation_data=(x_validation, y_validation),
            **candidate,
        )
        best_iteration = candidate_history["best_iteration"]
        best_loss = float(candidate_history["validation_loss"].min())
        selected_candidate = {**candidate, "n_iterations": best_iteration}
        tuning_results.append((best_loss, selected_candidate))
    best_validation_loss, fsnm_parameters = min(tuning_results, key=lambda result: result[0])
    print(f"Best tuning loss: {best_validation_loss:.4f}")
else:
    fsnm_parameters = DEFAULT_FSNM_PARAMETERS.copy()

print(f"Hyperparameter tuning: {TUNE_HYPERPARAMETERS}")
print(f"FSNM hyperparameters: {fsnm_parameters}")


tree_fit_parameters = {
    **fsnm_parameters, "n_iterations": LOSS_CURVE_ITERATIONS
}
phi_fsnm, psi_fsnm, values_fsnm, _ = fit_fsnm(
    x_train, y_train, rank=RANK, seed=SEED,
    **fsnm_parameters,
)
_, _, _, history_fsnm = fit_fsnm(
    x_train, y_train, rank=RANK, seed=SEED,
    validation_data=(x_validation, y_validation),
    **tree_fit_parameters,
)
phi_validation, psi_validation = scaled_factors(
    phi_fsnm, psi_fsnm, values_fsnm, x_validation, y_validation
)
validation_loss = float(empirical_loss(phi_validation, psi_validation))

grid = np.linspace(-1, 1, 160)
exact_basis = basis_matrix(grid)
kappa_true = kappa_exact(grid, grid)
phi_fsnm_grid = phi_fsnm.predict(grid[:, None])
psi_fsnm_grid = psi_fsnm.predict(grid[:, None])
kappa_fsnm = 1 + (phi_fsnm_grid * values_fsnm) @ psi_fsnm_grid.T

metrics = {
    "RMSE": np.sqrt(np.mean((kappa_fsnm - kappa_true) ** 2)),
    "spectrum error": np.linalg.norm(values_fsnm - SIGMAS),
    "subspace error": 0.5 * (
        subspace_error(phi_fsnm_grid, exact_basis)
        + subspace_error(psi_fsnm_grid, exact_basis)
    ),
    "orthogonality error": 0.5 * (
        orthogonality_error(phi_fsnm_grid)
        + orthogonality_error(psi_fsnm_grid)
    ),
}

print(f"True singular values: {SIGMAS}")
print(f"FSNM singular values: {np.round(values_fsnm, 4)}")
print(f"FSNM validation loss: {validation_loss:.4f}")
print(f"Selected iteration: {fsnm_parameters['n_iterations']}")
print("\nMetric                 FSNM")
for metric, value in metrics.items():
    print(f"{metric:20s} {value:9.4f}")


value_limits = (
    min(kappa_true.min(), kappa_fsnm.min()),
    max(kappa_true.max(), kappa_fsnm.max()),
)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
for axis, (values, title) in zip(
    axes[:2],
    [
        (kappa_true, r"True $\kappa$"),
        (kappa_fsnm, r"FSNM $\widehat\kappa$"),
    ],
):
    image = axis.imshow(
        values.T,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap="coolwarm",
        vmin=value_limits[0],
        vmax=value_limits[1],
    )
    axis.set(title=title, xlabel="$x$", ylabel="$y$")
    fig.colorbar(image, ax=axis, shrink=0.82)

error = kappa_fsnm - kappa_true
error_limit = np.max(np.abs(error))
image = axes[2].imshow(
    error.T,
    origin="lower",
    extent=[-1, 1, -1, 1],
    cmap="coolwarm",
    vmin=-error_limit,
    vmax=error_limit,
)
axes[2].set(title=r"$\widehat\kappa-\kappa$", xlabel="$x$", ylabel="$y$")
fig.colorbar(image, ax=axes[2], shrink=0.82)

project_directory = Path.cwd().parent if Path.cwd().name == "experiments" else Path.cwd()
figure_directory = project_directory / "figures"
figure_directory.mkdir(exist_ok=True)
figure_path = figure_directory / "00_rank3_close_spectrum.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {figure_path}")

iterations = np.arange(1, len(history_fsnm["training_loss"]) + 1)
loss_figure, loss_axis = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
loss_axis.plot(
    iterations, history_fsnm["training_loss"],
    color="tab:blue", linewidth=2, label="Training",
)
loss_axis.plot(
    iterations, history_fsnm["validation_loss"],
    color="tab:orange", linewidth=2, label="Validation",
)
loss_axis.axvline(
    fsnm_parameters["n_iterations"], color="black",
    linestyle=":", linewidth=1.5, label="Selected iteration",
)
loss_axis.set(xlabel="Iteration", ylabel="Empirical loss")
loss_axis.grid(alpha=0.25)
loss_axis.margins(y=0.12)
loss_axis.legend(frameon=False)
loss_path = figure_directory / "00_rank3_training_loss.png"
loss_figure.savefig(loss_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {loss_path}")

integration_grid = np.linspace(-1, 1, 4_001)
kappa_true_queries = kappa_exact(grid, integration_grid)
psi_training = psi_fsnm.predict(y_train[:, None])
kappa_fsnm_queries = 1 + (phi_fsnm_grid * values_fsnm) @ psi_training.T

def exact_conditional(query_values):
    return 0.5 * np.trapezoid(
        kappa_true_queries * query_values[None, :],
        integration_grid,
        axis=1,
    )


def estimated_conditional(query_values):
    return np.mean(kappa_fsnm_queries * query_values[None, :], axis=1)


exact_mean = exact_conditional(integration_grid)
estimated_mean = estimated_conditional(y_train)
exact_second_moment = exact_conditional(integration_grid**2)
estimated_second_moment = estimated_conditional(y_train**2)
exact_variance = exact_second_moment - exact_mean**2
estimated_variance = estimated_second_moment - estimated_mean**2
exact_tail = exact_conditional((integration_grid > 0.5).astype(float))
estimated_tail = estimated_conditional((y_train > 0.5).astype(float))

query_results = [
    (r"$\mathbb{E}[Y\mid X=x]$", exact_mean, estimated_mean),
    (r"$\mathrm{Var}(Y\mid X=x)$", exact_variance, estimated_variance),
    (r"$\mathbb{P}(Y>0.5\mid X=x)$", exact_tail, estimated_tail),
]
query_errors = {
    title: np.sqrt(np.mean((estimated - exact) ** 2))
    for title, exact, estimated in query_results
}

query_figure, query_axes = plt.subplots(
    1, 3, figsize=(12, 3.2), constrained_layout=True
)
for axis, (title, exact, estimated) in zip(query_axes, query_results):
    axis.plot(grid, exact, color="black", linewidth=2, label="Exact")
    axis.plot(grid, estimated, color="tab:red", linewidth=2, linestyle="--", label="FSNM")
    axis.set(title=title, xlabel="$x$")
query_axes[0].set_ylabel("Conditional functional")
query_axes[0].legend(frameon=False)
query_figure_path = figure_directory / "01_conditional_queries.png"
query_figure.savefig(query_figure_path, dpi=200, bbox_inches="tight")
for title, error in query_errors.items():
    print(f"{title} RMSE: {error:.4f}")
print(f"Figure saved to: {query_figure_path}")
query_figure


TUNE_LINEAR_HYPERPARAMETERS = False
LINEAR_MAX_ITERATIONS = 40

DEFAULT_LINEAR_PARAMETERS = {
    "n_iterations": 11,
    "step_size": 0.1,
    "learner_type": "linear_spline",
    "n_knots": 6,
    "spline_degree": 3,
    "learner_ridge": 1e-2,
}
LINEAR_PARAMETER_GRID = {
    "step_size": [0.1, 0.2, 0.5],
    "learner_type": ["linear_spline"],
    "n_knots": [6, 10, 14],
    "spline_degree": [3],
    "learner_ridge": [1e-4, 1e-2],
}

if TUNE_LINEAR_HYPERPARAMETERS:
    linear_tuning_results = []
    for candidate in ParameterGrid(LINEAR_PARAMETER_GRID):
        _, _, _, candidate_history = fit_fsnm(
            x_train, y_train, rank=RANK, seed=SEED,
            n_iterations=LINEAR_MAX_ITERATIONS,
            validation_data=(x_validation, y_validation),
            **candidate,
        )
        best_iteration = candidate_history["best_iteration"]
        best_loss = float(candidate_history["validation_loss"].min())
        selected_candidate = {**candidate, "n_iterations": best_iteration}
        linear_tuning_results.append((best_loss, selected_candidate))
    best_linear_tuning_loss, linear_parameters = min(
        linear_tuning_results, key=lambda result: result[0]
    )
    print(f"Best linear tuning loss: {best_linear_tuning_loss:.4f}")
else:
    linear_parameters = DEFAULT_LINEAR_PARAMETERS.copy()

print(f"Linear hyperparameter tuning: {TUNE_LINEAR_HYPERPARAMETERS}")
linear_fit_parameters = {
    **linear_parameters, "n_iterations": LOSS_CURVE_ITERATIONS
}
phi_linear, psi_linear, values_linear, _ = fit_fsnm(
    x_train, y_train, rank=RANK, seed=SEED,
    **linear_parameters,
)
_, _, _, history_linear = fit_fsnm(
    x_train, y_train, rank=RANK, seed=SEED,
    validation_data=(x_validation, y_validation),
    **linear_fit_parameters,
)
phi_linear_validation, psi_linear_validation = scaled_factors(
    phi_linear, psi_linear, values_linear, x_validation, y_validation
)
linear_validation_loss = float(
    empirical_loss(phi_linear_validation, psi_linear_validation)
)
phi_linear_grid = phi_linear.predict(grid[:, None])
psi_linear_grid = psi_linear.predict(grid[:, None])
kappa_linear = 1 + (phi_linear_grid * values_linear) @ psi_linear_grid.T
linear_metrics = {
    "RMSE": np.sqrt(np.mean((kappa_linear - kappa_true) ** 2)),
    "spectrum error": np.linalg.norm(values_linear - SIGMAS),
    "subspace error": 0.5 * (
        subspace_error(phi_linear_grid, exact_basis)
        + subspace_error(psi_linear_grid, exact_basis)
    ),
    "orthogonality error": 0.5 * (
        orthogonality_error(phi_linear_grid)
        + orthogonality_error(psi_linear_grid)
    ),
}
print(f"Linear-spline parameters: {linear_parameters}")
print(f"Estimated singular values: {np.round(values_linear, 4)}")
print(f"Validation loss: {linear_validation_loss:.4f}")
print(f"Selected iteration: {linear_parameters['n_iterations']}")
for metric, value in linear_metrics.items():
    print(f"{metric:<24}{value:.4f}")

linear_figure, linear_axes = plt.subplots(
    1, 3, figsize=(11.5, 3.6), constrained_layout=True
)
for axis, (values, title) in zip(
    linear_axes[:2],
    [(kappa_true, r"True $\kappa$"), (kappa_linear, r"FSNM $\widehat\kappa$")],
):
    image = axis.imshow(
        values.T, origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm",
        vmin=value_limits[0], vmax=value_limits[1],
    )
    axis.set(title=title, xlabel="$x$", ylabel="$y$")
    linear_figure.colorbar(image, ax=axis, shrink=0.82)
linear_error = kappa_linear - kappa_true
linear_error_limit = np.max(np.abs(linear_error))
image = linear_axes[2].imshow(
    linear_error.T, origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm",
    vmin=-linear_error_limit, vmax=linear_error_limit,
)
linear_axes[2].set(title=r"$\widehat\kappa-\kappa$", xlabel="$x$", ylabel="$y$")
linear_figure.colorbar(image, ax=linear_axes[2], shrink=0.82)
linear_figure_path = figure_directory / "02_linear_rank3.png"
linear_figure.savefig(linear_figure_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {linear_figure_path}")

linear_iterations = np.arange(1, len(history_linear["training_loss"]) + 1)
linear_loss_figure, linear_loss_axis = plt.subplots(
    figsize=(5.5, 3.6), constrained_layout=True
)
linear_loss_axis.plot(
    linear_iterations, history_linear["training_loss"],
    color="tab:blue", linewidth=2, label="Training",
)
linear_loss_axis.plot(
    linear_iterations, history_linear["validation_loss"],
    color="tab:orange", linewidth=2, label="Validation",
)
linear_loss_axis.axvline(
    linear_parameters["n_iterations"], color="black",
    linestyle=":", linewidth=1.5, label="Selected iteration",
)
linear_loss_axis.set(xlabel="Iteration", ylabel="Empirical loss")
linear_loss_axis.grid(alpha=0.25)
linear_loss_axis.margins(y=0.12)
linear_loss_axis.legend(frameon=False)
linear_loss_path = figure_directory / "02_linear_training_loss.png"
linear_loss_figure.savefig(linear_loss_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {linear_loss_path}")

psi_linear_training = psi_linear.predict(y_train[:, None])
kappa_linear_queries = (
    1 + (phi_linear_grid * values_linear) @ psi_linear_training.T
)
def linear_conditional(query_values):
    return np.mean(kappa_linear_queries * query_values[None, :], axis=1)

linear_mean = linear_conditional(y_train)
linear_second_moment = linear_conditional(y_train**2)
linear_variance = linear_second_moment - linear_mean**2
linear_tail = linear_conditional((y_train > 0.5).astype(float))
linear_query_results = [
    (r"$\mathbb{E}[Y\mid X=x]$", exact_mean, linear_mean),
    (r"$\mathrm{Var}(Y\mid X=x)$", exact_variance, linear_variance),
    (r"$\mathbb{P}(Y>0.5\mid X=x)$", exact_tail, linear_tail),
]
linear_query_figure, linear_query_axes = plt.subplots(
    1, 3, figsize=(12, 3.2), constrained_layout=True
)
for axis, (title, exact, estimated) in zip(
    linear_query_axes, linear_query_results
):
    axis.plot(grid, exact, color="black", linewidth=2, label="Exact")
    axis.plot(
        grid, estimated, color="tab:red", linewidth=2,
        linestyle="--", label="FSNM",
    )
    axis.set(title=title, xlabel="$x$")
linear_query_axes[0].set_ylabel("Conditional functional")
linear_query_axes[0].legend(frameon=False)
linear_query_path = figure_directory / "03_linear_conditional_queries.png"
linear_query_figure.savefig(linear_query_path, dpi=200, bbox_inches="tight")
for title, exact, estimated in linear_query_results:
    error = np.sqrt(np.mean((estimated - exact) ** 2))
    print(f"{title} RMSE: {error:.4f}")
print(f"Figure saved to: {linear_query_path}")
linear_query_figure

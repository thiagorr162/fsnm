import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsnm import empirical_loss, fit_fsnm


OXIDES = ["sio2", "na2o", "b2o3"]
OXIDE_LABELS = [r"SiO$_2$", r"Na$_2$O", r"B$_2$O$_3$"]
TARGET = "refractive_index"
RANK = 5
N_DISPLAY_MODES = 3
SEED = 2026
N_PERMUTATIONS = 999

DEFAULT_PARAMETERS = {
    "rank": RANK,
    "n_iterations": 40,
    "step_size": 0.1,
    "max_depth": 4,
    "min_samples_leaf": 100,
    "seed": 0,
}


def load_data(path):
    data = pd.read_parquet(path)
    composition_columns = [column for column in data if column != TARGET]
    plausible_response = data[TARGET].between(1, 4.5)
    data = data.loc[plausible_response].copy()
    composition = data[composition_columns].to_numpy(float, copy=True)
    composition /= composition.sum(axis=1, keepdims=True)
    response = data[TARGET].to_numpy(float)

    other_oxides = [column for column in composition_columns if column not in OXIDES]
    exact_ternary = (
        data[other_oxides].fillna(0).abs().sum(axis=1) < 1e-10
    ) & (data[OXIDES].fillna(0) > 0).all(axis=1)
    ternary_composition = data.loc[exact_ternary, OXIDES].to_numpy(float, copy=True)
    ternary_composition /= ternary_composition.sum(axis=1, keepdims=True)
    return (
        composition,
        response,
        exact_ternary.to_numpy(),
        ternary_composition,
        composition_columns,
    )


def split_data(composition, response, exact_ternary):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(response))
    n_train = int(0.70 * len(order))
    n_validation = int(0.15 * len(order))
    cutoffs = np.cumsum([n_train, n_validation])
    train, validation, test = np.split(order, cutoffs)
    return {
        "train": (composition[train], response[train], exact_ternary[train]),
        "validation": (
            composition[validation],
            response[validation],
            exact_ternary[validation],
        ),
        "test": (composition[test], response[test], exact_ternary[test]),
    }


def scaled_factors(model, x, y):
    phi_model, psi_model, singular_values, _ = model
    scale = np.sqrt(np.maximum(singular_values, 0))
    return phi_model.predict(x) * scale, psi_model.predict(y[:, None]) * scale


def validation_loss(model, x, y):
    phi, psi = scaled_factors(model, x, y)
    return float(empirical_loss(phi, psi))


def select_parameters(parts):
    x_train, y_train, _ = parts["train"]
    x_validation, y_validation, _ = parts["validation"]
    candidates = product(
        [3, 5],
        [0.05, 0.1],
        [3, 4],
        [100, 300],
    )
    results = []
    for rank, step_size, max_depth, min_samples_leaf in candidates:
        parameters = {
            "rank": rank,
            "n_iterations": 50,
            "step_size": step_size,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "seed": 0,
        }
        try:
            model = fit_fsnm(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                **parameters,
            )
            loss = validation_loss(model, x_validation, y_validation)
            parameters["n_iterations"] = model[3]["best_iteration"]
            results.append((loss, parameters))
            print(f"validation loss={loss:.5f}: {parameters}")
        except np.linalg.LinAlgError:
            print(f"singular fit skipped: {parameters}")
    if not results:
        raise RuntimeError("Every validation fit was singular.")
    best_loss, best_parameters = min(results, key=lambda result: result[0])
    print(f"selected validation loss={best_loss:.5f}: {best_parameters}")
    return best_parameters


def fit_model(parts, parameters):
    x_train, y_train, _ = parts["train"]
    x_validation, y_validation, _ = parts["validation"]
    return fit_fsnm(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        **parameters,
    )


def kernel(model, x, y):
    phi_model, psi_model, singular_values, _ = model
    phi = phi_model.predict(np.asarray(x).reshape(len(x), -1))
    psi = psi_model.predict(np.asarray(y).reshape(len(y), -1))
    return 1 + (phi * singular_values) @ psi.T


def permutation_test(model, x_test, y_test):
    phi, psi = scaled_factors(model, x_test, y_test)
    observed = -float(empirical_loss(phi, psi))
    rng = np.random.default_rng(SEED + 1)
    null_scores = np.empty(N_PERMUTATIONS)
    for index in range(N_PERMUTATIONS):
        null_scores[index] = -empirical_loss(
            phi, psi[rng.permutation(len(psi))]
        )
    p_value = (1 + np.count_nonzero(null_scores >= observed)) / (
        N_PERMUTATIONS + 1
    )
    return observed, p_value


def orient_modes(model, x, y):
    phi_model, psi_model, singular_values, _ = model
    phi = phi_model.predict(x)
    psi = psi_model.predict(y[:, None])
    signs = np.ones(len(singular_values))
    for mode in range(len(singular_values)):
        correlation = np.corrcoef(psi[:, mode], y)[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            signs[mode] = -1
    return phi * signs, psi * signs, signs


def barycentric_to_cartesian(composition):
    composition = np.asarray(composition)
    x_coordinate = composition[:, 2] + 0.5 * composition[:, 0]
    y_coordinate = np.sqrt(3) * composition[:, 0] / 2
    return np.column_stack([x_coordinate, y_coordinate])


def format_simplex(axis):
    height = np.sqrt(3) / 2
    axis.plot([0, 1, 0.5, 0], [0, 0, height, 0], color="black", linewidth=0.9)
    axis.text(0.5, height + 0.035, OXIDE_LABELS[0], ha="center", va="bottom")
    axis.text(-0.025, -0.025, OXIDE_LABELS[1], ha="right", va="top")
    axis.text(1.025, -0.025, OXIDE_LABELS[2], ha="left", va="top")
    axis.set(xlim=(-0.08, 1.08), ylim=(-0.07, height + 0.08), aspect="equal")
    axis.axis("off")


def add_simplex_scatter(axis, composition, values, title, cmap, vmin, vmax):
    coordinates = barycentric_to_cartesian(composition)
    image = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        s=25,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.25,
        edgecolors="white",
    )
    format_simplex(axis)
    axis.set_title(title, pad=8)
    return image


def save_query_figure(
    model,
    ternary_inputs,
    ternary_composition,
    ternary_response,
    figure_path,
):
    response_levels = np.quantile(ternary_response, [0.1, 0.5, 0.9])
    kernel_slices = [
        kernel(model, ternary_inputs, np.array([level]))[:, 0] - 1
        for level in response_levels
    ]

    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.25), constrained_layout=True)
    observed_image = add_simplex_scatter(
        axes[0],
        ternary_composition,
        ternary_response,
        "Observed RI",
        "viridis",
        ternary_response.min(),
        ternary_response.max(),
    )
    figure.colorbar(
        observed_image,
        ax=axes[0],
        shrink=0.72,
        label="Refractive index",
    )
    for axis, level, values in zip(
        axes[1:], response_levels, kernel_slices
    ):
        limit = np.quantile(np.abs(values), 0.98)
        image = add_simplex_scatter(
            axis,
            ternary_composition,
            values,
            rf"$\widehat\kappa_0(x,{level:.3f})$",
            "coolwarm",
            -limit,
            limit,
        )
        figure.colorbar(image, ax=axis, shrink=0.72)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_spectral_figure(
    model,
    ternary_inputs,
    ternary_composition,
    orientation_inputs,
    orientation_response,
    figure_path,
):
    _, _, singular_values, _ = model
    _, _, signs = orient_modes(
        model, orientation_inputs, orientation_response
    )
    phi = model[0].predict(ternary_inputs) * signs
    energy = singular_values**2
    shares = energy / energy.sum()
    y_grid = np.linspace(
        np.quantile(orientation_response, 0.01),
        np.quantile(orientation_response, 0.99),
        250,
    )
    psi_grid = model[1].predict(y_grid[:, None]) * signs

    figure = plt.figure(figsize=(13.2, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        N_DISPLAY_MODES + 1,
        width_ratios=[0.8] + [1] * N_DISPLAY_MODES,
    )
    spectrum_axis = figure.add_subplot(grid[:, 0])
    spectrum_axis.bar(np.arange(1, RANK + 1), singular_values, color="tab:blue")
    for index, (value, share) in enumerate(zip(singular_values, shares), start=1):
        spectrum_axis.text(
            index,
            value,
            f"{100 * share:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    spectrum_axis.set(
        title="Dependence spectrum",
        xlabel="Mode",
        ylabel="Singular value",
        xticks=np.arange(1, RANK + 1),
        ylim=(0, 1.18 * singular_values.max()),
    )

    scaled_phi = phi * np.sqrt(singular_values)
    scaled_psi = psi_grid * np.sqrt(singular_values)
    for mode in range(N_DISPLAY_MODES):
        simplex_axis = figure.add_subplot(grid[0, mode + 1])
        limit = np.abs(scaled_phi[:, mode]).max()
        image = add_simplex_scatter(
            simplex_axis,
            ternary_composition,
            scaled_phi[:, mode],
            rf"Mode {mode + 1}: $\widehat\sigma={singular_values[mode]:.3f}$",
            "coolwarm",
            -limit,
            limit,
        )
        figure.colorbar(image, ax=simplex_axis, shrink=0.67)
        response_axis = figure.add_subplot(grid[1, mode + 1])
        response_axis.axhline(0, color="0.65", linewidth=0.8)
        response_axis.plot(y_grid, scaled_psi[:, mode], color="tab:blue", linewidth=1.8)
        response_axis.set(
            xlabel="Refractive index",
            ylabel=(
                rf"$\sqrt{{\widehat\sigma_{{{mode + 1}}}}}\,"
                rf"\widehat\psi_{{{mode + 1}}}(y)$"
            ),
        )
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_path_figure(
    model,
    ternary_composition,
    ternary_response,
    composition_columns,
    figure_path,
):
    sodium = 0.15
    silica = np.linspace(0.36, 0.80, 180)
    path_composition = np.column_stack(
        [silica, np.full_like(silica, sodium), 1 - silica - sodium]
    )
    path = np.zeros((len(path_composition), len(composition_columns)))
    for oxide, values in zip(OXIDES, path_composition.T):
        path[:, composition_columns.index(oxide)] = values
    y_grid = np.linspace(
        np.quantile(ternary_response, 0.01),
        np.quantile(ternary_response, 0.99),
        220,
    )
    centered_density_ratio = kernel(model, path, y_grid) - 1

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    coordinates = barycentric_to_cartesian(ternary_composition)
    path_coordinates = barycentric_to_cartesian(path_composition)
    axes[0].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=15,
        color="0.65",
        alpha=0.55,
    )
    axes[0].plot(
        path_coordinates[:, 0],
        path_coordinates[:, 1],
        color="tab:red",
        linewidth=2.2,
    )
    format_simplex(axes[0])
    axes[0].set_title(r"Path with Na$_2$O fixed at 15%", pad=8)

    limit = np.quantile(np.abs(centered_density_ratio), 0.99)
    image = axes[1].imshow(
        centered_density_ratio.T,
        origin="lower",
        aspect="auto",
        extent=[100 * silica.min(), 100 * silica.max(), y_grid.min(), y_grid.max()],
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    nearby = (
        (np.abs(ternary_composition[:, 1] - sodium) <= 0.025)
        & (ternary_composition[:, 0] >= silica.min())
        & (ternary_composition[:, 0] <= silica.max())
    )
    axes[1].scatter(
        100 * ternary_composition[nearby, 0],
        ternary_response[nearby],
        s=18,
        facecolors="none",
        edgecolors="white",
        linewidths=0.7,
        alpha=0.75,
    )
    axes[1].set(
        title="Learned kernel along the path",
        xlabel=r"SiO$_2$ (mol%)",
        ylabel="Refractive index",
        xlim=(100 * silica.min(), 100 * silica.max()),
    )
    figure.colorbar(
        image,
        ax=axes[1],
        shrink=0.8,
        label=r"$\widehat\kappa_0(x,y)$",
    )
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def report_results(
    model,
    parts,
    ternary_inputs,
    ternary_composition,
    ternary_response,
):
    x_test, y_test, ternary_test = parts["test"]
    score, p_value = permutation_test(model, x_test, y_test)
    singular_values = model[2]
    energy = singular_values**2
    shares = energy / energy.sum()
    effective_rank = energy.sum() ** 2 / np.sum(energy**2)
    _, _, signs = orient_modes(model, x_test, y_test)
    ternary_phi = model[0].predict(ternary_inputs) * signs
    psi = model[1].predict(y_test[:, None]) * signs

    print(f"full observations: {sum(len(values[1]) for values in parts.values())}")
    print(f"exact ternary observations: {len(ternary_response)}")
    print(
        "split sizes: "
        + ", ".join(
            f"{name}={len(values[1])}" for name, values in parts.items()
        )
    )
    print(f"selected iteration: {model[3]['best_iteration']}")
    print(f"singular values: {np.round(singular_values, 4)}")
    print(f"spectral energy shares: {np.round(shares, 4)}")
    print(f"captured chi-square energy: {energy.sum():.4f}")
    print(f"effective spectral rank: {effective_rank:.3f}")
    print(f"ternary observations in test split: {ternary_test.sum()}")
    print(f"held-out dependence score: {score:.4f}")
    print(f"permutation p-value: {p_value:.3f}")
    for mode in range(len(singular_values)):
        oxide_correlations = [
            np.corrcoef(
                ternary_phi[:, mode], ternary_composition[:, oxide]
            )[0, 1]
            for oxide in range(3)
        ]
        response_correlation = np.corrcoef(psi[:, mode], y_test)[0, 1]
        print(
            f"mode {mode + 1} correlations: "
            + ", ".join(
                f"{oxide}={value:+.3f}"
                for oxide, value in zip(OXIDES, oxide_correlations)
            )
            + f", RI={response_correlation:+.3f}"
        )


def main(validate):
    project_directory = Path(__file__).resolve().parents[1]
    (
        composition,
        response,
        exact_ternary,
        ternary_composition,
        composition_columns,
    ) = load_data(project_directory / "data" / "refractive_index.parquet")
    parts = split_data(composition, response, exact_ternary)
    parameters = select_parameters(parts) if validate else DEFAULT_PARAMETERS
    model = fit_model(parts, parameters)
    ternary_inputs = composition[exact_ternary]
    ternary_response = response[exact_ternary]
    report_results(
        model,
        parts,
        ternary_inputs,
        ternary_composition,
        ternary_response,
    )

    figure_directory = project_directory / "figures"
    figure_directory.mkdir(exist_ok=True)
    x_test, y_test, _ = parts["test"]
    paths = {
        "queries": figure_directory / "06_glass_simplex_queries.png",
        "spectrum": figure_directory / "07_glass_spectral_modes.png",
        "path": figure_directory / "08_glass_conditional_path.png",
    }
    save_query_figure(
        model,
        ternary_inputs,
        ternary_composition,
        ternary_response,
        paths["queries"],
    )
    save_spectral_figure(
        model,
        ternary_inputs,
        ternary_composition,
        x_test,
        y_test,
        paths["spectrum"],
    )
    save_path_figure(
        model,
        ternary_composition,
        ternary_response,
        composition_columns,
        paths["path"],
    )
    for path in paths.values():
        print(f"figure saved to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Select the tree hyperparameters on the validation split.",
    )
    main(parser.parse_args().validate)

"""Evaluate identity-functional RI prediction on a held-out test set."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsnm import fit_fsnm


TARGET = "refractive_index"
SPLIT_SEED = 2026
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_data(path):
    data = pd.read_parquet(path)
    composition_columns = [column for column in data if column != TARGET]
    data = data.loc[data[TARGET].between(1, 4.5)].copy()
    composition = data[composition_columns].to_numpy(float, copy=True)
    composition /= composition.sum(axis=1, keepdims=True)
    response = data[TARGET].to_numpy(float)
    return composition, response, composition_columns


def split_train_validation_test(composition, response):
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(response))
    n_validation = int(np.ceil(VALIDATION_FRACTION * len(order)))
    n_test = int(np.ceil(TEST_FRACTION * len(order)))
    validation_indices = order[:n_validation]
    test_indices = order[n_validation : n_validation + n_test]
    train_indices = order[n_validation + n_test :]
    return (
        composition[train_indices],
        response[train_indices],
        composition[validation_indices],
        response[validation_indices],
        composition[test_indices],
        response[test_indices],
    )


def load_selected_parameters(path):
    metadata = json.loads(path.read_text())
    return {
        "rank": metadata["selected_rank"],
        "n_iterations": metadata["selected_iteration"],
        "step_size": metadata["tree_parameters"]["step_size"],
        "max_depth": metadata["selected_max_depth"],
        "min_samples_leaf": metadata["selected_min_samples_leaf"],
        "seed": metadata["tree_parameters"]["seed"],
    }


def identity_prediction(model, x, marginal_response):
    phi_model, psi_model, singular_values, _ = model
    phi = phi_model.predict(x)
    psi_reference = psi_model.predict(marginal_response[:, None])
    scaled_phi = phi * singular_values
    numerator = marginal_response.mean() + scaled_phi @ np.mean(
        marginal_response[:, None] * psi_reference,
        axis=0,
    )
    estimated_mass = 1 + scaled_phi @ psi_reference.mean(axis=0)
    return numerator, estimated_mass


def regression_metrics(observed, predicted):
    errors = predicted - observed
    mse = float(np.mean(errors**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "r_squared": float(
            1 - np.sum(errors**2) / np.sum((observed - observed.mean()) ** 2)
        ),
        "bias": float(errors.mean()),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def calibration_bins(observed, predicted, n_bins=10):
    order = np.argsort(predicted)
    bins = np.array_split(order, n_bins)
    predicted_means = np.array([predicted[index].mean() for index in bins])
    observed_means = np.array([observed[index].mean() for index in bins])
    observed_errors = np.array(
        [observed[index].std(ddof=1) / np.sqrt(len(index)) for index in bins]
    )
    return predicted_means, observed_means, observed_errors


def save_figure(observed, predicted, fit_mean, metrics, figure_path):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.1),
        constrained_layout=True,
    )

    limits = (
        min(observed.min(), predicted.min()),
        max(observed.max(), predicted.max()),
    )
    density = axes[0].hexbin(
        observed,
        predicted,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1.2)
    axes[0].axhline(
        fit_mean,
        color="tab:red",
        linestyle=":",
        linewidth=1.3,
        label="Mean baseline",
    )
    axes[0].set(
        title="Held-out RI prediction",
        xlabel="Observed refractive index",
        ylabel="Predicted refractive index",
        xlim=limits,
        ylim=limits,
    )
    axes[0].legend(frameon=False)
    figure.colorbar(density, ax=axes[0], label="log count")

    predicted_means, observed_means, standard_errors = calibration_bins(
        observed,
        predicted,
    )
    calibration_minimum = min(
        predicted_means.min(),
        np.min(observed_means - 1.96 * standard_errors),
    )
    calibration_maximum = max(
        predicted_means.max(),
        np.max(observed_means + 1.96 * standard_errors),
    )
    calibration_padding = 0.08 * (calibration_maximum - calibration_minimum)
    calibration_limits = (
        calibration_minimum - calibration_padding,
        calibration_maximum + calibration_padding,
    )
    axes[1].errorbar(
        predicted_means,
        observed_means,
        yerr=1.96 * standard_errors,
        fmt="o-",
        color="tab:blue",
        capsize=3,
        label="Prediction deciles",
    )
    axes[1].plot(
        calibration_limits,
        calibration_limits,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].set(
        title=(
            f"MSE={metrics['mse']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"$R^2$={metrics['r_squared']:.3f}"
        ),
        xlabel="Mean predicted RI",
        ylabel="Mean observed RI",
        xlim=calibration_limits,
        ylim=calibration_limits,
    )
    axes[1].legend(frameon=False)

    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    package_directory = Path(__file__).resolve().parents[1]
    project_directory = Path(__file__).resolve().parents[3]
    composition, response, composition_columns = load_data(
        package_directory / "data" / "refractive_index.parquet"
    )
    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = split_train_validation_test(composition, response)
    x_fit = np.concatenate([x_train, x_validation], axis=0)
    y_fit = np.concatenate([y_train, y_validation], axis=0)
    parameters = load_selected_parameters(
        package_directory / "artifacts" / "glass_refractive_index_fsnm.json"
    )

    print(
        f"refitting selected configuration on {len(y_fit)} "
        "training-plus-validation observations; "
        f"holding out {len(y_test)} observations",
        flush=True,
    )
    model = fit_fsnm(x_fit, y_fit, **parameters)
    predicted, estimated_mass = identity_prediction(model, x_test, y_fit)
    baseline = np.full_like(y_test, y_fit.mean())
    metrics = regression_metrics(y_test, predicted)
    baseline_metrics = regression_metrics(y_test, baseline)

    figure_path = (
        project_directory / "tex" / "figures" / "12_glass_identity_prediction.png"
    )
    save_figure(y_test, predicted, y_fit.mean(), metrics, figure_path)

    artifact_directory = package_directory / "artifacts"
    artifact_directory.mkdir(exist_ok=True)
    model_path = artifact_directory / "glass_refractive_index_holdout_fsnm.joblib"
    results_path = artifact_directory / "glass_identity_prediction.json"
    joblib.dump(model, model_path, compress=3)
    results = {
        "target": TARGET,
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "train_observations": len(y_train),
        "validation_observations": len(y_validation),
        "fit_observations": len(y_fit),
        "test_observations": len(y_test),
        "composition_columns": composition_columns,
        "parameters": parameters,
        "identity_prediction_metrics": metrics,
        "mean_baseline_metrics": baseline_metrics,
        "fit_response_mean": float(y_fit.mean()),
        "estimated_kernel_mass": {
            "mean": float(estimated_mass.mean()),
            "standard_deviation": float(estimated_mass.std()),
            "minimum": float(estimated_mass.min()),
            "maximum": float(estimated_mass.max()),
        },
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"identity prediction metrics: {metrics}")
    print(f"mean baseline metrics: {baseline_metrics}")
    print(
        "estimated kernel mass: "
        f"mean={estimated_mass.mean():.5f}, "
        f"sd={estimated_mass.std():.5f}, "
        f"range=({estimated_mass.min():.5f}, {estimated_mass.max():.5f})"
    )
    print(f"figure saved to: {figure_path}")
    print(f"model saved to: {model_path}")
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()

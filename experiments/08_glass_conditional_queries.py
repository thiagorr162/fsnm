"""Conditional-distribution queries from the fitted glass RI kernel.

Reuses the holdout-refit kernel from ``06_glass_identity_prediction.py``
(no refitting here) to answer two functionals that a scalar point
predictor cannot provide without extra machinery: conditional prediction
intervals for refractive index given a composition, and the screening
probability that a candidate composition's RI exceeds a target threshold.
Both reuse the same fitted factor maps and the same fit-sample responses
for marginal integration as the identity-functional experiment.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "refractive_index"
SPLIT_SEED = 2026
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
THRESHOLD = 1.8
BATCH_SIZE = 500
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def load_data(path):
    data = pd.read_parquet(path)
    composition_columns = [column for column in data if column != TARGET]
    data = data.loc[data[TARGET].between(1, 4.5)].copy()
    composition = data[composition_columns].to_numpy(float, copy=True)
    composition /= composition.sum(axis=1, keepdims=True)
    response = data[TARGET].to_numpy(float)
    return composition, response


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


def conditional_summaries(
    phi_model,
    psi_model,
    singular_values,
    x_batch,
    y_marginal,
    y_sorted,
    order,
    threshold,
    quantile_levels,
):
    phi = phi_model.predict(x_batch)
    psi_all = psi_model.predict(y_marginal[:, None])
    weights = 1 + (phi * singular_values) @ psi_all.T
    weights = np.maximum(weights, 0.0)

    weights_sorted = weights[:, order]
    cum_weights = np.cumsum(weights_sorted, axis=1)
    total_weights = cum_weights[:, -1:]
    normalized_cum = cum_weights / total_weights

    quantiles = np.empty((len(x_batch), len(quantile_levels)))
    for j, level in enumerate(quantile_levels):
        idx = (normalized_cum >= level).argmax(axis=1)
        quantiles[:, j] = y_sorted[idx]

    tail_weight = np.sum(weights * (y_marginal[None, :] > threshold), axis=1)
    exceedance_probability = tail_weight / weights.sum(axis=1)

    return quantiles, exceedance_probability


def reliability_bins(observed_binary, predicted_probability, n_bins=10):
    order = np.argsort(predicted_probability)
    bins = np.array_split(order, n_bins)
    predicted_means = np.array(
        [predicted_probability[index].mean() for index in bins]
    )
    observed_means = np.array(
        [observed_binary[index].mean() for index in bins]
    )
    observed_errors = np.array(
        [
            np.sqrt(observed_means[i] * (1 - observed_means[i]) / len(bins[i]))
            for i in range(n_bins)
        ]
    )
    return predicted_means, observed_means, observed_errors


def save_figure(
    y_test,
    q10,
    q25,
    q50,
    q75,
    q90,
    observed_exceedance,
    exceedance_probability,
    figure_path,
    n_interval_points=150,
    seed=0,
):
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.2), constrained_layout=True
    )

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(y_test), size=n_interval_points, replace=False)
    sample = sample[np.argsort(q50[sample])]
    x_axis = np.arange(len(sample))

    axes[0].fill_between(
        x_axis, q10[sample], q90[sample],
        color="tab:blue", alpha=0.18, label="80% interval",
    )
    axes[0].fill_between(
        x_axis, q25[sample], q75[sample],
        color="tab:blue", alpha=0.35, label="50% interval",
    )
    axes[0].plot(
        x_axis, q50[sample], color="tab:blue", linewidth=1.3,
        label="Conditional median",
    )
    axes[0].scatter(
        x_axis, y_test[sample], color="black", s=9, zorder=5,
        label="Observed RI",
    )
    axes[0].set(
        title="Conditional prediction intervals (random test subset)",
        xlabel="Test observation (sorted by median)",
        ylabel="Refractive index",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    predicted_means, observed_means, observed_errors = reliability_bins(
        observed_exceedance, exceedance_probability
    )
    axes[1].errorbar(
        predicted_means, observed_means, yerr=1.96 * observed_errors,
        fmt="o-", color="tab:red", capsize=3,
    )
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2)
    axes[1].set(
        title=f"Screening probability reliability ($\\mathrm{{RI}}>{THRESHOLD}$)",
        xlabel="Predicted exceedance probability",
        ylabel="Observed exceedance frequency",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )

    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    package_directory = Path(__file__).resolve().parents[1]
    project_directory = Path(__file__).resolve().parents[3]

    composition, response = load_data(
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

    model_path = (
        package_directory
        / "artifacts"
        / "glass_refractive_index_holdout_fsnm.joblib"
    )
    phi_model, psi_model, singular_values, _ = joblib.load(model_path)

    order = np.argsort(y_fit)
    y_sorted = y_fit[order]

    n_test = len(y_test)
    quantile_results = np.empty((n_test, len(QUANTILE_LEVELS)))
    exceedance_results = np.empty(n_test)

    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)
        quantiles, exceedance = conditional_summaries(
            phi_model,
            psi_model,
            singular_values,
            x_test[start:end],
            y_fit,
            y_sorted,
            order,
            THRESHOLD,
            QUANTILE_LEVELS,
        )
        quantile_results[start:end] = quantiles
        exceedance_results[start:end] = exceedance
        print(f"processed {end}/{n_test}", flush=True)

    q10, q25, q50, q75, q90 = quantile_results.T
    coverage_50 = float(np.mean((y_test >= q25) & (y_test <= q75)))
    coverage_80 = float(np.mean((y_test >= q10) & (y_test <= q90)))
    width_50 = float(np.mean(q75 - q25))
    width_80 = float(np.mean(q90 - q10))

    observed_exceedance = (y_test > THRESHOLD).astype(float)
    brier_score = float(np.mean((exceedance_results - observed_exceedance) ** 2))
    climatological_rate = float(observed_exceedance.mean())
    climatological_brier = float(
        np.mean((climatological_rate - observed_exceedance) ** 2)
    )

    figure_path = (
        project_directory / "tex" / "figures" / "13_glass_conditional_queries.png"
    )
    save_figure(
        y_test, q10, q25, q50, q75, q90,
        observed_exceedance, exceedance_results, figure_path,
    )

    results = {
        "target": TARGET,
        "threshold": THRESHOLD,
        "quantile_levels": QUANTILE_LEVELS,
        "test_observations": n_test,
        "coverage_50": coverage_50,
        "coverage_80": coverage_80,
        "mean_interval_width_50": width_50,
        "mean_interval_width_80": width_80,
        "exceedance_brier_score": brier_score,
        "climatological_exceedance_rate": climatological_rate,
        "climatological_brier_score": climatological_brier,
    }
    artifact_directory = package_directory / "artifacts"
    results_path = artifact_directory / "glass_conditional_queries.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"50% interval coverage: {coverage_50:.4f} (nominal 0.50)")
    print(f"80% interval coverage: {coverage_80:.4f} (nominal 0.80)")
    print(f"mean 50% interval width: {width_50:.4f}")
    print(f"mean 80% interval width: {width_80:.4f}")
    print(f"exceedance Brier score: {brier_score:.4f}")
    print(
        "climatological (base-rate) Brier score: "
        f"{climatological_brier:.4f} at rate {climatological_rate:.4f}"
    )
    print(f"figure saved to: {figure_path}")
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()

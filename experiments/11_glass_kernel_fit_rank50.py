"""Fit the glass composition--RI kernel at rank=50, holding the other
selected hyperparameters (max depth 10, min leaf 100, step size 0.1) fixed
at their validated values from 05_glass_kernel_fit.py. Saves a checkpoint
to the same directory as the original grid search, following the same
fit-and-checkpoint pattern, so the work survives even if this script is
interrupted; reruns reuse the checkpoint instead of refitting. Also saves
the kernel-recovery figure and prints the spectral summary used in the
paper text, both computed directly from the training-partition fit (no
separate refit).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fsnm import fit_fsnm


TARGET = "refractive_index"
SEED = 2026
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
RANK = 50
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 100
MAX_ITERATIONS = 100
PATIENCE = 8
TREE_PARAMETERS = {"step_size": 0.1, "seed": 0}


def load_data(path):
    data = pd.read_parquet(path)
    composition_columns = [column for column in data if column != TARGET]
    data = data.loc[data[TARGET].between(1, 4.5)].copy()
    composition = data[composition_columns].to_numpy(float, copy=True)
    composition /= composition.sum(axis=1, keepdims=True)
    response = data[TARGET].to_numpy(float)
    return composition, response


def split_train_validation_test(composition, response):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(response))
    n_validation = int(np.ceil(VALIDATION_FRACTION * len(order)))
    n_test = int(np.ceil(TEST_FRACTION * len(order)))
    validation_indices = order[:n_validation]
    test_indices = order[n_validation : n_validation + n_test]
    train_indices = order[n_validation + n_test :]
    return (
        composition[train_indices], response[train_indices],
        composition[validation_indices], response[validation_indices],
        composition[test_indices], response[test_indices],
    )


def fit_or_load(x_train, y_train, x_validation, y_validation, checkpoint_path):
    if checkpoint_path.exists():
        print(f"loading existing checkpoint: {checkpoint_path}", flush=True)
        return joblib.load(checkpoint_path)

    print(
        f"fitting rank={RANK}, depth={MAX_DEPTH}, leaf={MIN_SAMPLES_LEAF} "
        f"on {len(y_train)} training / {len(y_validation)} validation obs",
        flush=True,
    )
    model = fit_fsnm(
        x_train, y_train,
        rank=RANK,
        n_iterations=MAX_ITERATIONS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        patience=PATIENCE,
        validation_data=(x_validation, y_validation),
        verbose=True,
        **TREE_PARAMETERS,
    )
    history = model[3]
    best_iteration = history["best_iteration"]
    best_validation_loss = float(history["validation_loss"][best_iteration - 1])
    print(
        f"finished rank={RANK}, depth={MAX_DEPTH}, "
        f"min_leaf={MIN_SAMPLES_LEAF}: iteration={best_iteration}, "
        f"validation loss={best_validation_loss:.6f}",
        flush=True,
    )

    candidate = {
        "rank": RANK,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "best_iteration": best_iteration,
        "best_validation_loss": best_validation_loss,
        "model": model,
    }
    temporary_path = checkpoint_path.with_suffix(".joblib.tmp")
    joblib.dump(candidate, temporary_path, compress=3)
    temporary_path.replace(checkpoint_path)
    print(f"checkpoint saved to: {checkpoint_path}")
    return candidate


def spectral_summary(singular_values):
    energy = singular_values**2
    shares = energy / energy.sum()
    effective_rank = energy.sum() ** 2 / np.sum(energy**2)
    return energy, shares, float(effective_rank)


def save_figure(selected, figure_path):
    history = selected["model"][3]
    singular_values = selected["model"][2]
    _, shares, _ = spectral_summary(singular_values)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), constrained_layout=True)

    iterations = np.arange(1, len(history["training_loss"]) + 1)
    axes[0].plot(
        iterations, history["training_loss"],
        color="tab:blue", linewidth=1.8, label="Training",
    )
    axes[0].plot(
        iterations, history["validation_loss"],
        color="tab:orange", linewidth=1.8, label="Validation",
    )
    axes[0].axvline(
        selected["best_iteration"], color="black", linestyle="--",
        linewidth=1.2, label=f"Selected: {selected['best_iteration']}",
    )
    axes[0].set(
        title=f"Selection: rank {selected['rank']}, depth {selected['max_depth']}",
        xlabel="Iteration",
        ylabel="Empirical FSNM loss",
        xlim=(1, len(iterations)),
    )
    axes[0].legend(frameon=False)

    locations = np.arange(1, len(singular_values) + 1)
    axes[1].bar(locations, singular_values, color="tab:blue", label="Singular value")
    axes[1].set(
        title="Training-partition dependence spectrum",
        xlabel="Mode",
        ylabel="Singular value",
        xlim=(0.25, len(singular_values) + 0.75),
        ylim=(0, 1.08 * singular_values.max()),
    )
    energy_axis = axes[1].twinx()
    energy_axis.plot(
        locations, np.cumsum(shares), color="tab:orange",
        marker="o", markersize=2.2, linewidth=1.2, label="Cumulative energy",
    )
    energy_axis.set(ylabel="Cumulative energy", ylim=(0, 1.04))
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = energy_axis.get_legend_handles_labels()
    axes[1].legend(
        handles_left + handles_right, labels_left + labels_right,
        frameon=False, loc="center right",
    )

    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    package_directory = Path(__file__).resolve().parents[1]
    composition, response = load_data(
        package_directory / "data" / "refractive_index.parquet"
    )
    x_train, y_train, x_validation, y_validation, x_test, y_test = (
        split_train_validation_test(composition, response)
    )

    checkpoint_directory = package_directory / "artifacts" / "glass_kernel_search"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / (
        f"rank_{RANK}_depth_{MAX_DEPTH}_leaf_{MIN_SAMPLES_LEAF}.joblib"
    )
    selected = fit_or_load(x_train, y_train, x_validation, y_validation, checkpoint_path)

    singular_values = selected["model"][2]
    energy, shares, effective_rank = spectral_summary(singular_values)
    cumulative = np.cumsum(shares)
    modes_to_90_percent = int(np.searchsorted(cumulative, 0.90) + 1)

    figure_path = package_directory / "figures" / "11_glass_kernel_fit.png"
    save_figure(selected, figure_path)

    summary = {
        "rank": selected["rank"],
        "max_depth": selected["max_depth"],
        "min_samples_leaf": selected["min_samples_leaf"],
        "best_iteration": selected["best_iteration"],
        "best_validation_loss": selected["best_validation_loss"],
        "singular_value_min": float(singular_values.min()),
        "singular_value_max": float(singular_values.max()),
        "captured_chi_square_energy": float(energy.sum()),
        "effective_spectral_rank": effective_rank,
        "first_mode_share": float(shares[0]),
        "first_ten_share": float(cumulative[9]),
        "modes_to_90_percent": modes_to_90_percent,
    }
    print(json.dumps(summary, indent=2))
    print(f"figure saved to: {figure_path}")


if __name__ == "__main__":
    main()

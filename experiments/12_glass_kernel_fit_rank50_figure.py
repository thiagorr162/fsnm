"""Build the composition--RI kernel-fit figure and spectral summary at
rank=50, from the checkpoint produced by 11_glass_kernel_fit_rank50.py.
No refitting is performed here: the singular values and loss trajectory
are those of the training-partition fit itself (validation is used only
for early stopping, as in the original rank/depth grid search).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


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
    project_directory = Path(__file__).resolve().parents[3]
    checkpoint_path = (
        package_directory / "artifacts" / "glass_kernel_search"
        / "rank_50_depth_10_leaf_100.joblib"
    )
    selected = joblib.load(checkpoint_path)
    singular_values = selected["model"][2]
    energy, shares, effective_rank = spectral_summary(singular_values)

    figure_path = project_directory / "tex" / "figures" / "11_glass_kernel_fit.png"
    save_figure(selected, figure_path)

    cumulative = np.cumsum(shares)
    modes_to_90 = int(np.searchsorted(cumulative, 0.90) + 1)

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
        "modes_to_90_percent": modes_to_90,
    }
    print(json.dumps(summary, indent=2))
    print(f"figure saved to: {figure_path}")


if __name__ == "__main__":
    main()

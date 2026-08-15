"""Fit the real glass composition--refractive-index kernel."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fsnm import fit_fsnm


TARGET = "refractive_index"
SEED = 2026
VALIDATION_FRACTION = 0.15
RANK_CANDIDATES = (5, 10, 20)
DEPTH_CANDIDATES = (2, 5, 10)
LEAF_SIZE_CANDIDATES = (50, 100, 200)
MAX_ITERATIONS = 100
PATIENCE = 8
N_JOBS = 4
TREE_PARAMETERS = {
    "step_size": 0.1,
    "seed": 0,
}


def load_data(path):
    data = pd.read_parquet(path)
    composition_columns = [column for column in data if column != TARGET]
    plausible_response = data[TARGET].between(1, 4.5)
    data = data.loc[plausible_response].copy()
    composition = data[composition_columns].to_numpy(float, copy=True)
    row_sums = composition.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every retained composition must have a positive sum.")
    composition /= row_sums
    response = data[TARGET].to_numpy(float)
    return composition, response, composition_columns, int((~plausible_response).sum())


def split_train_validation(composition, response):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(response))
    n_validation = int(np.ceil(VALIDATION_FRACTION * len(order)))
    validation_indices = order[:n_validation]
    train_indices = order[n_validation:]
    return (
        composition[train_indices],
        response[train_indices],
        composition[validation_indices],
        response[validation_indices],
    )


def fit_candidate(
    rank,
    max_depth,
    min_samples_leaf,
    x_train,
    y_train,
    x_validation,
    y_validation,
    checkpoint_directory,
):
    model = fit_fsnm(
        x_train,
        y_train,
        rank=rank,
        n_iterations=MAX_ITERATIONS,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        patience=PATIENCE,
        validation_data=(x_validation, y_validation),
        **TREE_PARAMETERS,
    )
    history = model[3]
    best_iteration = history["best_iteration"]
    best_validation_loss = float(history["validation_loss"][best_iteration - 1])
    print(
        f"finished rank={rank}, depth={max_depth}, "
        f"min_leaf={min_samples_leaf}: iteration={best_iteration}, "
        f"validation loss={best_validation_loss:.6f}",
        flush=True,
    )
    candidate = {
        "rank": rank,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "best_iteration": best_iteration,
        "best_validation_loss": best_validation_loss,
        "model": model,
    }
    checkpoint_path = checkpoint_directory / (
        f"rank_{rank}_depth_{max_depth}_leaf_{min_samples_leaf}.joblib"
    )
    temporary_path = checkpoint_path.with_suffix(".joblib.tmp")
    joblib.dump(candidate, temporary_path, compress=3)
    temporary_path.replace(checkpoint_path)
    return candidate


def select_model(
    x_train,
    y_train,
    x_validation,
    y_validation,
    checkpoint_directory,
    fit_pending=True,
):
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    configurations = [
        (rank, max_depth, min_samples_leaf)
        for rank in RANK_CANDIDATES
        for max_depth in DEPTH_CANDIDATES
        for min_samples_leaf in LEAF_SIZE_CANDIDATES
    ]
    candidates = []
    pending = []
    for rank, max_depth, min_samples_leaf in configurations:
        checkpoint_path = checkpoint_directory / (
            f"rank_{rank}_depth_{max_depth}_leaf_{min_samples_leaf}.joblib"
        )
        if checkpoint_path.exists():
            candidates.append(joblib.load(checkpoint_path))
        else:
            pending.append((rank, max_depth, min_samples_leaf))

    if fit_pending:
        print(
            f"loaded {len(candidates)} checkpoints; fitting {len(pending)} "
            "remaining validation trajectories "
            f"with {N_JOBS} parallel workers",
            flush=True,
        )
    else:
        print(
            f"selecting among {len(candidates)} completed checkpoints; "
            f"leaving {len(pending)} configurations pending",
            flush=True,
        )
    if fit_pending and pending:
        candidates.extend(
            Parallel(n_jobs=N_JOBS, verbose=10)(
                delayed(fit_candidate)(
                    rank,
                    max_depth,
                    min_samples_leaf,
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    checkpoint_directory,
                )
                for rank, max_depth, min_samples_leaf in pending
            )
        )
    return min(candidates, key=lambda candidate: candidate["best_validation_loss"]), candidates


def fit_full_model(composition, response, selected):
    print(
        "refitting on all observations with "
        f"rank={selected['rank']} and "
        f"depth={selected['max_depth']}, "
        f"min_leaf={selected['min_samples_leaf']}, and "
        f"iterations={selected['best_iteration']}",
        flush=True,
    )
    return fit_fsnm(
        composition,
        response,
        rank=selected["rank"],
        n_iterations=selected["best_iteration"],
        max_depth=selected["max_depth"],
        min_samples_leaf=selected["min_samples_leaf"],
        **TREE_PARAMETERS,
    )


def spectral_summary(singular_values):
    energy = singular_values**2
    shares = energy / energy.sum()
    effective_rank = energy.sum() ** 2 / np.sum(energy**2)
    return energy, shares, float(effective_rank)


def save_figure(selected, full_model, figure_path):
    history = selected["model"][3]
    singular_values = full_model[2]
    _, shares, _ = spectral_summary(singular_values)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), constrained_layout=True)

    iterations = np.arange(1, len(history["training_loss"]) + 1)
    axes[0].plot(
        iterations,
        history["training_loss"],
        color="tab:blue",
        linewidth=1.8,
        label="Training",
    )
    axes[0].plot(
        iterations,
        history["validation_loss"],
        color="tab:orange",
        linewidth=1.8,
        label="Validation",
    )
    axes[0].axvline(
        selected["best_iteration"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Selected: {selected['best_iteration']}",
    )
    axes[0].set(
        title=(
            f"Selection: rank {selected['rank']}, depth "
            f"{selected['max_depth']}"
        ),
        xlabel="Iteration",
        ylabel="Empirical FSNM loss",
        xlim=(1, len(iterations)),
    )
    axes[0].legend(frameon=False)

    locations = np.arange(1, len(singular_values) + 1)
    axes[1].bar(locations, singular_values, color="tab:blue")
    offset = 0.025 * singular_values.max()
    for location, value, share in zip(locations, singular_values, shares):
        axes[1].text(
            location,
            value + offset,
            f"{100 * share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set(
        title="Full-data dependence spectrum",
        xlabel="Mode",
        ylabel="Singular value",
        xticks=locations,
        ylim=(0, 1.16 * singular_values.max()),
    )

    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_artifacts(full_model, selected, candidates, composition_columns, directory):
    directory.mkdir(exist_ok=True)
    model_path = directory / "glass_refractive_index_fsnm.joblib"
    metadata_path = directory / "glass_refractive_index_fsnm.json"
    joblib.dump(full_model, model_path, compress=3)
    metadata = {
        "target": TARGET,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "composition_columns": composition_columns,
        "selected_rank": selected["rank"],
        "selected_max_depth": selected["max_depth"],
        "selected_min_samples_leaf": selected["min_samples_leaf"],
        "selected_iteration": selected["best_iteration"],
        "selected_validation_loss": selected["best_validation_loss"],
        "maximum_iterations": MAX_ITERATIONS,
        "patience": PATIENCE,
        "parallel_workers": N_JOBS,
        "evaluated_configurations": len(candidates),
        "total_configurations": (
            len(RANK_CANDIDATES)
            * len(DEPTH_CANDIDATES)
            * len(LEAF_SIZE_CANDIDATES)
        ),
        "tree_parameters": TREE_PARAMETERS,
        "rank_candidates": [
            {
                "rank": candidate["rank"],
                "max_depth": candidate["max_depth"],
                "min_samples_leaf": candidate["min_samples_leaf"],
                "best_iteration": candidate["best_iteration"],
                "best_validation_loss": candidate["best_validation_loss"],
            }
            for candidate in candidates
        ],
        "singular_values": full_model[2].tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, metadata_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-completed",
        action="store_true",
        help="Select only among existing checkpoints without fitting pending cases.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    package_directory = Path(__file__).resolve().parents[1]
    project_directory = Path(__file__).resolve().parents[3]
    composition, response, composition_columns, n_discarded = load_data(
        package_directory / "data" / "refractive_index.parquet"
    )
    x_train, y_train, x_validation, y_validation = split_train_validation(
        composition, response
    )
    selected, candidates = select_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        package_directory / "artifacts" / "glass_kernel_search",
        fit_pending=not arguments.use_completed,
    )
    full_model = fit_full_model(composition, response, selected)
    energy, shares, effective_rank = spectral_summary(full_model[2])

    figure_path = project_directory / "tex" / "figures" / "11_glass_kernel_fit.png"
    save_figure(selected, full_model, figure_path)
    model_path, metadata_path = save_artifacts(
        full_model,
        selected,
        candidates,
        composition_columns,
        package_directory / "artifacts",
    )

    print(f"retained observations: {len(response)}")
    print(f"discarded response outliers: {n_discarded}")
    print(f"training observations: {len(y_train)}")
    print(f"validation observations: {len(y_validation)}")
    for candidate in candidates:
        print(
            f"rank {candidate['rank']}, depth={candidate['max_depth']}, "
            f"min_leaf={candidate['min_samples_leaf']}: "
            f"iteration={candidate['best_iteration']}, "
            f"validation loss={candidate['best_validation_loss']:.6f}"
        )
    print(f"selected rank: {selected['rank']}")
    print(f"selected max depth: {selected['max_depth']}")
    print(f"selected min samples leaf: {selected['min_samples_leaf']}")
    print(f"selected iteration: {selected['best_iteration']}")
    print(f"full-data singular values: {np.round(full_model[2], 5)}")
    print(f"spectral energy shares: {np.round(shares, 5)}")
    print(f"captured chi-square energy: {energy.sum():.5f}")
    print(f"effective spectral rank: {effective_rank:.4f}")
    print(f"figure saved to: {figure_path}")
    print(f"model saved to: {model_path}")
    print(f"metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

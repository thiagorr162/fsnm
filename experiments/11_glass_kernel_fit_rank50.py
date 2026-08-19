"""Fit the glass composition--RI kernel at rank=50, holding the other
selected hyperparameters (max depth 10, min leaf 100, step size 0.1) fixed
at their validated values from 05_glass_kernel_fit.py. Saves a checkpoint
to the same directory as the original grid search, following the same
fit-and-checkpoint pattern, so the work survives even if this script is
interrupted.
"""

from pathlib import Path

import joblib
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


if __name__ == "__main__":
    main()

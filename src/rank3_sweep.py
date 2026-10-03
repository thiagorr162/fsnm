"""One W&B sweep trial, or held-out evaluation of its validation winner."""

import argparse
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"  # One BLAS thread per agent.

import numpy as np  # noqa: E402
import wandb  # noqa: E402
from fsnm import empirical_loss, fit_fsnm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = np.array([0.18, 0.16, 0.12])
DEFAULTS = dict(learner_type="tree", step_size=0.1, max_depth=3,
                min_samples_leaf=300, n_knots=10, spline_degree=3,
                learner_ridge=1e-3, rank=3, seed=0, n_iterations=200,
                patience=30, n_train=10000, n_validation=4000,
                train_seed=12, validation_seed=100)


def basis_matrix(values):
    return np.sqrt(2) * np.column_stack([
        np.sin(np.pi * values), np.cos(np.pi * values), np.sin(2 * np.pi * values),
    ])


def sample_joint(size, seed):
    """Preserve the notebook's rejection sampler and exact training draw."""
    rng = np.random.default_rng(seed)
    x_parts, y_parts, count = [], [], 0
    while count < size:
        x, y = rng.uniform(-1, 1, size), rng.uniform(-1, 1, size)
        ratio = 1 + np.sum(basis_matrix(x) * SIGMAS * basis_matrix(y), axis=1)
        accepted = rng.uniform(size=size) < ratio / (1 + 2 * SIGMAS.sum())
        x_parts.append(x[accepted])
        y_parts.append(y[accepted])
        count += accepted.sum()
    return np.concatenate(x_parts)[:size], np.concatenate(y_parts)[:size]


def fit_parameters(config):
    keys = ["rank", "seed", "learner_type", "step_size"]
    keys += (["max_depth", "min_samples_leaf"] if config["learner_type"] == "tree"
             else ["n_knots", "spline_degree", "learner_ridge"])
    parameters = {key: config[key] for key in keys}
    if "min_samples_leaf" in parameters:
        parameters["min_samples_leaf"] = int(np.rint(parameters["min_samples_leaf"]))
    return parameters


def fit(config, iterations=None):
    return fit_fsnm(
        *sample_joint(config["n_train"], config["train_seed"]),
        **fit_parameters(config), n_iterations=iterations or config["n_iterations"],
        patience=config["patience"],
        validation_data=sample_joint(config["n_validation"], config["validation_seed"]),
    )


def train(smoke=False):
    defaults = {**DEFAULTS, **(dict(n_train=1000, n_validation=400, n_iterations=5, patience=2)
                              if smoke else {})}
    with wandb.init(config=defaults, job_type="sweep", save_code=True) as run:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                _, _, values, history = fit(run.config)
                losses = history["validation_loss"]
                if not all(np.isfinite(a).all() for a in (values, losses, history["training_loss"])):
                    raise FloatingPointError("Non-finite spectrum or loss history")
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            run.summary.update({"status": "failed", "error": str(error)})
            run.finish(exit_code=1)
            return
        for iteration, (training, validation) in enumerate(zip(history["training_loss"], losses), 1):
            run.log({"iteration": iteration, "training_loss": float(training),
                     "validation_loss": float(validation)}, step=iteration)
        run.summary.update({
            "status": "ok", "best_validation_loss": float(losses.min()),
            "best_iteration": history["best_iteration"], "singular_values": values.tolist(),
            "fit_parameters": fit_parameters(run.config),
        })


def evaluate(model, joint, pairs):
    phi, psi, values, _ = model
    x_joint, y_joint = joint
    loss = empirical_loss(phi.predict(x_joint[:, None]) * np.sqrt(values),
                          psi.predict(y_joint[:, None]) * np.sqrt(values))
    x, y = pairs
    estimate = 1 + np.sum(phi.predict(x[:, None]) * values * psi.predict(y[:, None]), axis=1)
    exact = 1 + np.sum(basis_matrix(x) * SIGMAS * basis_matrix(y), axis=1)
    return dict(empirical_loss=float(loss), kernel_rmse=float(np.sqrt(np.mean((estimate-exact)**2))))


def comparison_figure(baseline, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(-1, 1, 160)
    exact = 1 + (basis_matrix(grid) * SIGMAS) @ basis_matrix(grid).T
    estimates = [1 + (m[0].predict(grid[:, None]) * m[2]) @ m[1].predict(grid[:, None]).T
                 for m in (baseline, winner)]
    limits = (min(a.min() for a in [exact, *estimates]), max(a.max() for a in [exact, *estimates]))
    error_limit = max(float(np.abs(a-exact).max()) for a in estimates) or np.finfo(float).eps
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, (name, estimate) in enumerate(zip(("Original", "Sweep winner"), estimates)):
        for column, (matrix, title) in enumerate(zip((exact, estimate, estimate-exact),
                                                    ("Exact kernel", f"{name} estimate", f"{name} error"))):
            image = axes[row, column].imshow(
                matrix.T, origin="lower", extent=[-1, 1, -1, 1],
                cmap="coolwarm" if column == 2 else "viridis",
                vmin=-error_limit if column == 2 else limits[0],
                vmax=error_limit if column == 2 else limits[1],
            )
            axes[row, column].set(title=title, xlabel="x", ylabel="y")
            figure.colorbar(image, ax=axes[row, column], shrink=0.8)
    return figure


def evaluate_best(sweep_path, output):
    runs = list(wandb.Api().sweep(sweep_path).runs)
    if any(r.state in ("running", "pending") for r in runs):
        raise ValueError("Wait for active sweep trials to finish before evaluating the test set")
    candidates = [r for r in runs if r.state == "finished" and r.summary.get("status") == "ok"
                  and np.isfinite(r.summary.get("best_validation_loss", np.nan))]
    if not candidates:
        raise ValueError("Sweep has no successful completed trials")
    best = min(candidates, key=lambda r: (r.summary["best_validation_loss"], r.id))
    config = dict(best.config)
    winner = fit(config, int(best.summary["best_iteration"]))
    if not np.isclose(winner[3]["validation_loss"].min(), best.summary["best_validation_loss"],
                      rtol=1e-7, atol=1e-9):
        raise RuntimeError("Winner reconstruction differs from W&B; check the code and environment")
    baseline = fit_fsnm(*sample_joint(config["n_train"], config["train_seed"]), rank=3, seed=0,
                        n_iterations=11, step_size=0.1, max_depth=3, min_samples_leaf=300)
    test = sample_joint(4000, 101)  # Created only after validation selection.
    rng = np.random.default_rng(102)
    pairs = (rng.uniform(-1, 1, 20000), rng.uniform(-1, 1, 20000))
    original, selected = evaluate(baseline, test, pairs), evaluate(winner, test, pairs)
    summary = dict(selected_run=best.url, selected_parameters=fit_parameters(config),
                   best_iteration=int(best.summary["best_iteration"]),
                   best_validation_loss=float(best.summary["best_validation_loss"]),
                   baseline_test=original, winner_test=selected,
                   kernel_rmse_reduction_percent=100*(1-selected["kernel_rmse"]/original["kernel_rmse"]))
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    figure = comparison_figure(baseline, winner)
    entity, project, _ = sweep_path.split("/")
    with wandb.init(entity=entity, project=project, job_type="evaluation", config={
        **config, "selected_run": best.url, "test_seed": 101, "kernel_pairs_seed": 102,
        "n_test": 4000, "n_eval_pairs": 20000,
    }) as run:
        run.summary.update(summary)
        run.log({"kernel_comparison": wandb.Image(figure)})
        summary["evaluation_run"] = run.url
        figure.savefig(output / "kernel_comparison.png", dpi=200)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib.pyplot as plt
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", metavar="ENTITY/PROJECT/SWEEP_ID")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rank3_sweep")
    parser.add_argument("--smoke", action="store_true", help="Run one small trial (supports W&B offline mode)")
    args = parser.parse_args()
    if args.evaluate:
        output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        evaluate_best(args.evaluate, output)
    else:
        train(args.smoke)

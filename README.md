# FSNM — Functional Spectral-Newton Method

Implementation and reproducible experiments for the Functional
Spectral-Newton Method (FSNM). The estimator lives in `src/fsnm.py`, and the
ACE, uLSIF, and kernel-CCA baselines live in `src/baselines.py`.

## Setup

From this directory (`code/fsnm`), install the locked environment with:

```bash
uv sync --dev
```

The development dependencies include the minimal Jupyter stack needed to
execute the notebooks. Data are stored in `data/`, generated figures in
`figures/`, and numerical summaries or model checkpoints in `artifacts/`.

The notebooks never write into the TeX repository. After checking a generated
figure, copy it manually from `code/fsnm/figures/` to `tex/figures/`.

## Paper notebooks

The six notebooks follow the experimental sections of the paper. Each one is
an executable narrative containing its setup, reported metrics, and all
figures for that block.

1. `notebooks/01_rank3_kernel_recovery.ipynb`
   fits the representative rank-three synthetic model and generates
   `00_rank3_close_spectrum.png` and `00_rank3_training_loss.png`.
2. `notebooks/02_tree_structured_kernel_recovery.ipynb`
   runs the discontinuous-region and irrelevant-feature experiments and
   generates `09_discontinuous_regions.png` and
   `10_tabular_irrelevant_features.png`.
3. `notebooks/03_baseline_comparison.ipynb`
   refits FSNM, ACE, uLSIF, and kernel CCA on ten independent draws for each
   of the three recovery settings. It writes
   `artifacts/baseline_comparison_multiseed.json`, the source for all three
   baseline tables.
4. `notebooks/04_conditional_queries.ipynb`
   uses one rank-three fit for the conditional mean, variance, exceedance
   probability, and conditional CDF. It generates
   `01_conditional_queries.png` and
   `02_tree_conditional_cdf_uncertainty.png`.
5. `notebooks/05_dependence_detection.ipynb`
   runs the independent and nonlinear zero-correlation examples and generates
   `04_independent_case.png` and `05_nonlinear_dependence.png`.
6. `notebooks/06_glass_application.ipynb`
   runs the complete restricted-family glass experiment and generates
   `11_glass_kernel_fit.png`, `12_glass_identity_prediction.png`, and
   `13_glass_conditional_queries.png`.

The notebooks can be executed interactively or headlessly with `nbclient`.
For example:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("notebooks/01_rank3_kernel_recovery.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook,
    timeout=7200,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.write(notebook, path)
```

## Conditional queries

Every downstream functional uses the same direct empirical operator

```python
estimate = np.mean(kappa_hat * g(y_marginal), axis=1)
```

without clipping or normalizing individual kernel weights. For a conditional
CDF, `g_t(y) = 1{y <= t}` is evaluated across the complete threshold grid.
Because a finite-sample signed curve need not itself satisfy all CDF
constraints, the complete curve is then projected by isotonic regression onto
the monotone `[0, 1]`-valued CDF class. This projection is applied after the
operator query and therefore does not redefine the learned kernel or the
individual weights.

The glass intervals additionally recalibrate quantile levels using validation
probability-integral-transform values. The screening probability uses the
direct query `g(y) = 1{y > 1.8}` and clips only the final scalar to `[0, 1]`.

## Restricted-family glass setup

The glass notebook retains only records whose nonzero oxide support is
contained in

```text
TiO2, Nb2O5, Ta2O5, La2O3, SiO2,
B2O3, P2O5, K2O, Na2O, Li2O
```

and requires every other oxide fraction to be zero to tolerance `1e-10`.
After restricting RI to `[1, 4.5]`, 3,013 records remain. The fixed seed-2026
split contains 2,109 training, 452 validation, and 452 test observations.

The validation grid uses rank `{5, 10, 20}`, maximum depth `{3, 5}`, minimum
leaf size `{25, 50, 100}`, step size `0.1`, at most 80 iterations, and
patience 10. Both validation FSNM loss and validation CRPS select rank 20,
depth 5, leaf size 25, and iteration 39.

The fresh full execution reports:

- identity-query RMSE `0.0711`, test R² `0.811`, and an `81.1%` MSE reduction
  relative to the training-mean baseline;
- calibrated 50%, 70%, and 90% interval coverage of `54.6%`, `71.5%`, and
  `89.4%`, with widths `0.043`, `0.069`, and `0.126` RI units;
- direct screening Brier score `0.0375`, versus `0.1156` for the
  training-rate climatological baseline.

## Continuous rank-three W&B sweep

Create a native W&B random sweep from the repository root:

```bash
.venv/bin/wandb login
.venv/bin/wandb sweep --project fsnm sweeps/rank3.yaml
```

Copy the `ENTITY/fsnm/SWEEP_ID` printed by W&B, then start four agents:

```bash
SWEEP_ID=ENTITY/fsnm/SWEEP_ID
for worker in 1 2 3 4; do
  .venv/bin/wandb agent --count 250 "$SWEEP_ID" &
done
wait
```

The sweep is capped at 1,000 trials. Restart agents against the same sweep
ID to continue assigning new trials after an interruption; completed runs
stay in W&B, while interrupted trials are not resumed midway through fitting.
W&B generates random configurations; their assignment order is not controlled
by a local seed. Model and data seeds remain fixed, and each run records its
configuration and source code so that individual fits can be reproduced.

The search uses equally likely trees and cubic splines, log-uniform step
sizes 0.01–0.5, tree depths 2–6, log-uniform leaf sizes 30–1000 rounded to
integers, spline knots 4–40, and log-uniform spline ridge 1e-6–10. Only the
parameters relevant to the chosen learner are passed to the estimator. Rank
is three, model seed zero, and each agent uses one BLAS thread. Trials use
the original 10,000-point training draw (seed 12), a fresh 4,000-point
validation draw (seed 100), at most 200 iterations, and patience 30.

W&B records training/validation curves, `best_validation_loss`, the selected
iteration, spectrum, and numerical failures. Sort successful runs by
`best_validation_loss` ascending to compare them. The estimator is unchanged.

After all agents finish, select the validation winner and evaluate it once
against the original 11-iteration model:

```bash
.venv/bin/python src/rank3_sweep.py --evaluate "$SWEEP_ID"
```

Evaluation uses 4,000 untouched joint observations (seed 101) and 20,000
independent uniform kernel-evaluation pairs (seed 102). Trials never see these
test sets. The command creates a separate W&B evaluation run containing test
metrics and the kernel comparison image. It also saves `summary.json` and
`kernel_comparison.png` to `artifacts/rank3_sweep/` for the final notebook cells.
Relative `--output-dir` paths resolve from the repository root. Test improvement
is measured, not guaranteed. The previous local manifest/log/leaderboard format
is replaced by W&B tracking; existing local outputs are left on disk.

One small trial can be checked without logging in or uploading data:

```bash
WANDB_MODE=offline .venv/bin/python src/rank3_sweep.py --smoke
```

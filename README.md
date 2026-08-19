# FSNM — Functional Spectral-Newton Method

Implementation and experiments for the Functional Spectral-Newton Method
(FSNM), an alternating functional block-Newton procedure for learning the
leading singular structure of a conditional expectation operator with
regression-tree weak learners.

## Setup

Run everything with `uv` from this directory (`code/fsnm`):

```bash
uv run python experiments/<script>.py
```

Source lives in `src/` (`fsnm.py` for the method, `baselines.py` for
ACE/uLSIF/kernel CCA). Experiment scripts live in `experiments/`, figures
are written to `figures/`, and cached model checkpoints go to `artifacts/`.

## Experiments

Each experiment script is self-contained and prints the numbers reported
in the paper. Details below cover data splits, hyperparameter grids, and
selection criteria; consult the corresponding script for exact code.

### Rank-three synthetic experiment (`00_rank3_conditional_queries.py`, `01_conditional_cdf_uncertainty.py`)

Independent training and validation samples of 10,000 and 4,000 pairs from
a known rank-3 density-ratio kernel, fitting a rank-`d=3` model.
Hyperparameters are selected by validation loss over step size
`{0.05, 0.1, 0.2}`, maximum tree depth `{3, unrestricted}`, and minimum
leaf size `{100, 300}`, each trajectory run for 40 iterations. The
selected configuration uses `T=11`, step size `0.1`, maximum depth `3`,
minimum leaf size `300`. Conditional queries reuse the training responses
as the empirical marginal.

**Conditional CDF bootstrap.** With hyperparameters fixed, 100 nonparametric
bootstrap samples (resampling the 10,000 training pairs, refitting, and
evaluating the conditional CDF on a grid of 301 values) give pointwise
2.5%–97.5% percentile bands; density-ratio weights are truncated at zero
and normalized within each `x`.

**Linear spline weak learners** (not currently reported in the paper). The
same experiment repeated with ridge regression on a cubic B-spline basis
for every scalar regression: step size from `{0.1, 0.2, 0.5}`, number of
knots from `{6, 10, 14}`, ridge penalty from `{1e-4, 1e-2}`, iteration
among the first 40. Validation selects 6 knots, ridge penalty `1e-2`, step
size `0.1`, `T=11`.

### Tree-structured synthetic settings (`04_tree_structured_synthetic.py`)

Two constructions — a discontinuous piecewise-constant kernel and a
20-dimensional tabular kernel with 17 irrelevant coordinates — both using
rejection sampling, 8,000 training and 3,000 validation pairs, rank
`d=3`, step size `0.15`, up to 40 iterations selected by validation. The
figures and spectrum in the main text show a single representative trial;
baseline comparison tables average over 10 independent trials (see
below). The discontinuous experiment uses max depth `3`, min leaf `120`,
errors on a 240×240 grid. The tabular experiment uses max depth `3`, min
leaf `250`, evaluated on 10,000 fresh pairs from `P_X ⊗ P_Y`; the
permutation diagnostic permutes each input column once on these same
pairs, after training.

### Baseline methods (`src/baselines.py`, `07_baseline_comparison.py`, `10_baseline_comparison_multiseed.py`)

For the rank-three and tree-structured synthetic experiments, ACE, uLSIF,
and regularized kernel CCA are additionally fit on the same training and
validation samples used for FSNM, with rank (or number of components)
fixed to `d=3`.

- **ACE**: rank-one ACE extended to rank `d` by fitting successive
  components with regression trees (same depth/leaf as the selected FSNM
  configuration), each deflated against previous components. Iteration
  within each component selected by validation correlation, up to 40
  iterations, early stopping after 8 without improvement.
- **uLSIF**: Gaussian-kernel model with 300 basis functions centered at a
  random subsample of the numerator (joint) training pairs; denominator
  pairs independently permute the training response. Bandwidth from
  `{0.15, 0.25, 0.35, 0.5, 0.75, 1, 1.5}` (standardized units), ridge
  penalty from `{1e-4, 1e-3, 1e-2, 1e-1}`, selected by validation
  unbiased risk estimate; final coefficients truncated to be nonnegative.
- **Kernel CCA**: fit on a random landmark subsample of 800 training
  observations, Gaussian kernels on standardized inputs/responses.
  Bandwidth multiplier from `{0.5, 1, 2}` of the median pairwise landmark
  distance, ridge regularization from `{1e-2, 1e-1}`, selected by mean
  validation canonical correlation.

**Multi-seed protocol.** Each baseline comparison table refits all four
methods on 10 independent draws (replicate `r` uses seed `r`, `r=0..9`;
replicate 0 reproduces the single trial reported elsewhere), with every
hyperparameter above held fixed. We report the mean and a 95% confidence
interval, `x̄ ± t(0.975, 9) · s/√10 ≈ x̄ ± 2.262 · s/√10`, and bold an
entry when its interval overlaps that of the lowest-mean method in its
column.

### Dependence-detection experiment (`02_dependence_detection.py`)

Independent and nonlinear-dependence cases both use 10,000 training,
4,000 validation, 4,000 test observations, rank `d=3`, step size `0.1`,
max tree depth `3`, min leaf size `300`; the independent and nonlinear
cases use iterations 1 and 13 respectively. The permutation test uses the
negative empirical test loss as the dependence score, compared with 999
scores from permuting the rows of the fitted `Y`-factor matrix (one added
to both the exceedance count and denominator).

### Glass composition–refractive-index kernel fit (`05_glass_kernel_fit.py`, `11_glass_kernel_fit_rank50.py`)

The input table has 73 oxide columns and one refractive-index (RI)
response. After retaining RI in `[1, 4.5]` and renormalizing compositions,
58,496 records remain. A fixed IID split (seed 2026, fractions
0.70/0.15/0.15) gives 40,946 / 8,775 / 8,775 training/validation/test
observations. All weak learners use step size `0.1`, minimum leaf size
100, seed 0; validation selects rank 50, depth 10, iteration 47
(validation loss −9.3442), each trajectory capped at 100 iterations with
patience-8 early stopping. `11_glass_kernel_fit_rank50.py` caches this fit
as a checkpoint (`artifacts/glass_kernel_search/`) and reuses it on rerun
instead of refitting.

This single fit, trained on the 40,946 training observations, is used
throughout — for the kernel-recovery figure, and for the prediction and
conditional-query experiments below, which reuse the training responses
as the marginal Monte Carlo sample and evaluate on the held-out test
partition (no refitting on combined train+validation data).

### RI prediction with the identity functional (`06_glass_identity_prediction.py`, `13_glass_identity_prediction_rank50.py`)

Uses the fitted kernel as a regression model via the identity functional
`f(y) = y`, approximating the marginal expectation by Monte Carlo over the
training responses. Reports MSE/RMSE/MAE/R² against a constant
(training-mean) baseline on the test partition.

### Conditional intervals and screening probabilities (`08_glass_conditional_queries.py`, `14_glass_conditional_queries_rank50.py`)

Reuses the same fitted kernel (no refitting) to compute, for each test
composition: a conditional prediction interval for RI (inverting the
self-normalized conditional CDF), and a screening probability
`P(RI > τ | X=x)` for a target threshold `τ = 1.8`. Reports empirical
coverage and mean width of the central intervals against their nominal
levels, and the Brier score of the screening probability against a
climatological (base-rate) baseline.

### Balancing ablation (`09_balancing_ablation.py`)

Compares `balance_every_iteration=True` (the practical heuristic) against
balancing only after the final iteration (matching the theorem's
finite-sample guarantee), on the rank-three synthetic experiment.

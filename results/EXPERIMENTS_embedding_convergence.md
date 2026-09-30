# Embedding Distribution Convergence (empirical check)

Log of [scripts/embedding_convergence.py](../scripts/embedding_convergence.py).

**Question.** The paper conjectures $W_1(\mu_n, \mu_m; \bar d) \to 0$ as $n, m \to \infty$ for the
distribution $\mu_n$ of width-$n$ INR weights. If a duplication-equivariant metanetwork is Lipschitz
w.r.t. $\bar d$, its *embedding* distributions across widths must converge too. This run tests
whether that is empirically plausible.

## Run configuration

| | |
|---|---|
| Date | 2026-07-11 (01:33–02:07) |
| Model family | plain GMN (`symmetry: permutation`), config `configs/mnist_cls/gmn_sizegen_mup.yml` |
| INR data | muP ReLU MNIST INRs, widths 16/24/32/48/64/80/96, `test` split of `mnist_sizegen_splits.json` (2000 INRs per width) |
| Embedding | input to `readout.global_mlp`, captured by forward hook (dup-equiv: 640-d, baseline: 128-d) |
| Metrics | Sliced Wasserstein (1000 random projections), Gaussian MMD (median-heuristic bandwidth, 500-sample subsets) |
| Conditions | `random-dupequiv` / `random-baseline` (10 seeds each), `trained-dupequiv` / `trained-baseline` (1 checkpoint each) |
| Trained checkpoints | `sizegen-v3` MNIST GMN-muP runs `sizegen-v3-gmn-mup-deq_ikmnmpj4`, `sizegen-v3-gmn-mup-base_to0udreq` |
| Artifacts | plots → `outputs/embedding_convergence/gmn_mup/plots/`; embeddings + distance CSVs → `$ANYDIM_DATA_ROOT/embedding_convergence/gmn_mup/` |

"dup-equiv" = fan-in rescaled edges (`FanInLabeledINRDataset`) + `aggregator: mean` + `LayerWiseMeanReadout`.
"baseline" = raw weights + `add` aggregation + `PermScaleInvariantReadout`.

## Results — SWD to the largest width (w96)

Mean ± std over 10 seeds for the random conditions; single value for trained.

| condition | w16 | w24 | w32 | w48 | w64 | w80 | monotone |
|---|---|---|---|---|---|---|---|
| `random-dupequiv` | 0.0349 ±0.0053 | 0.0169 ±0.0027 | 0.0105 ±0.0017 | 0.0042 ±0.0007 | 0.0023 ±0.0004 | 0.0012 ±0.0002 | 10/10 seeds |
| `trained-dupequiv` | 0.0613 | 0.0470 | 0.0375 | 0.0270 | 0.0181 | 0.0111 | yes |
| `random-baseline` | 36.65 ±6.22 | 32.98 ±5.59 | 29.31 ±4.97 | 21.98 ±3.73 | 14.65 ±2.48 | 7.33 ±1.24 | 10/10 seeds |
| `trained-baseline` | 7.933 | 7.049 | 6.153 | 4.458 | 2.943 | 1.466 | yes |

## Results — MMD to w96

| condition | w16 | w24 | w32 | w48 | w64 | w80 |
|---|---|---|---|---|---|---|
| `random-dupequiv` | 0.489 ±0.005 | 0.351 ±0.006 | 0.282 ±0.006 | 0.157 ±0.006 | 0.094 ±0.005 | 0.060 ±0.003 |
| `trained-dupequiv` | 0.166 | 0.135 | 0.126 | 0.093 | 0.042 | 0.041 |
| `random-baseline` | 0.897 ±0.005 | 0.898 ±0.005 | 0.899 ±0.006 | 0.903 ±0.008 | 0.907 ±0.011 | 0.908 ±0.016 |
| `trained-baseline` | 0.913 | 0.874 | 0.850 | 0.776 | 0.620 | 0.362 |

## Results — scale-normalized SWD to w96

Mean embedding norm $\|e\|$ by width, w16→w96: `random-baseline` 120 / 179 / 237 / 354 / 471 / 589 / 706; `random-dupequiv` flat at 9.94 across all widths.
SWD normalized by mean embedding norm:

| condition | w16 | w32 | w64 | w80 |
|---|---|---|---|---|
| `random-dupequiv` seed0 | 0.0031 | 0.0009 | 0.0002 | 0.0001 |
| `random-baseline` seed0 | 0.1032 | 0.0723 | 0.0289 | 0.0132 |
| `trained-dupequiv` | 0.0104 | 0.0067 | 0.0033 | 0.0020 |
| `trained-baseline` | 0.1001 | 0.0752 | 0.0304 | 0.0138 |

## Per-class and visual artifacts (not numerically summarized here)

- `perclass_swd_summary.pdf` — per-digit SWD, `random-dupequiv` seed 0.
- `pointcloud_2d_{dupequiv,baseline}_{grid,seed0}.pdf` — 2D-output random projections, 9 seeds.
- `pca_trained-{dupequiv,baseline}.pdf` — PCA of trained embeddings colored by width.
- `heatmap_swd_*_mean.pdf`, `convergence_{swd,mmd}_overlay.pdf`.

## Known issues / follow-ups

- **SP (`gmn_relu`) run incomplete.** Only `trained-dupequiv` w16/w24/w32 embeddings exist, in
  `outputs/embedding_convergence/gmn_relu/embeddings/`. SP-ReLU is the contrast case where
  convergence is expected to be weaker than muP.
- **`scalegmn` was never run.** Only `--model-type gmn`. ScaleGMN's scale-equivariant messages may
  behave differently.
- No fixed-norm / whitened variant of SWD is computed in-script; the normalized numbers above were
  computed post-hoc from the cached `.npz` embeddings.
- Trained-checkpoint paths are not recorded in any artifact. Consider dumping the resolved args to
  a JSON alongside the distance CSVs.

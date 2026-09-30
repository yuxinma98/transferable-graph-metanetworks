# Experiments — CIFAR-10 CNN Accuracy Prediction

## Contents
- [Task and data](#task-and-data)
- [Input-network zoo quality](#input-network-zoo-quality) — muP16's norms are flat, SP's grow $2.9\times$
- [Experiment 1 — Reproduce accuracy prediction at the zoo's native width](#experiment-1--reproduce-accuracy-prediction-at-the-zoos-native-width-w16) (fixed width w16)
- [Experiment 2 — Size generalization on wider equivalent CNNs](#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check) (sanity check, 2 widening families $\times$ 9 conditions)
- [Experiment 3 — Size generalization on our multi-width zoo](#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512) (train w16, test w16–w512), all 10 conditions in both directions, plus the conv matrix-product ScaleGMN
  - [Plain GMN](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional)
  - [ScaleGMN](#scalegmn-results-symmetry-scale-forward-and-bidirectional)
  - [Conv matrix-product GMN](#conv-matrix-product-gmn-results-message_fn_type-matrix_product_conv-forward-and-bidirectional)
  - [Conv matrix-product ScaleGMN](#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward) 
  - [$\tau$ within hyperparameter strata](#tau-within-hyperparameter-strata) (diagnostic 4)
  - [Mandatory diagnostics](#mandatory-diagnostics) (what must accompany the tables above)
- [Experiment status](#experiment-status)

---

## Task and Data

Input networks: the zoo CNN
`[1] → conv3×3/s2 → [c] → conv3×3/s2 → [c] → conv3×3/s2 → [c] → GAP → dense → [10]`, ReLU
(4970 parameters at `c = 16`). Implementation `src/data/zoo_cnn.py`; graph construction
`src/data/cnn_zoo_dataset.py`. The width axis is the **channel count** `c`; `node2type` has 14
values (1 + 3 + 10) at every width, so one model instance and one set of positional encodings
transfer across widths unchanged.

Metanetwork head: `sigmoid(net(batch))` against the CNN's test accuracy, MSE loss.

Two data arms are used from Experiment 3 on:

- **SP** — the sampled hyperparameters instantiated as-is at every width
  (`$ANYDIM_DATA_ROOT/cnn_zoo/w{N}_sp/`).
- **muP16** — muP with base width 16 (= the train width), so the normalized spectral norms are flat
  across widths and the OOD covariate shift is isolated from a norm shift
  (`$ANYDIM_DATA_ROOT/cnn_zoo/w{N}_mup16/`). Verified by `scripts/check_mup_cnn.py`.

Because base width = train width, **w16 muP16 ≡ w16 SP exactly** and the two arms differ only at the
OOD widths — the same design as MNIST Experiment 3's `base_width=24`.

### Roles

Each CNN in the zoo is one hyperparameter draw (base learning rate, init multiplier, weight decay,
dropout, data fraction, epoch count) trained at one width. Within each arm the draws are organized
into four **roles**, defined by the seed they are drawn from
([generate_cnn_zoo.py:89](../scripts/generate_cnn_zoo.py#L89)):

| role | n | widths | seed | purpose |
|---|---|---|---|---|
| `train` | 8000 | w16 only | 1000 | fits the metanetwork |
| `val` | 1000 | w16 only | 2000 | model selection (val τ at the train width) |
| `test` | 1000 per width | all | `3000 + width` | every reported metric |
| `paired` | 200 per width | all | 9000 | one cross-width diagnostic (`τ_HP`) |

The two that matter for reading the tables differ in whether the seed depends on the width:

- **`test`** — seed `3000 + width`, so each width samples *different* hyperparameters. No draw is
  ever shared between two widths, or between a width and train/val (asserted in
  `generate_predgen_splits.py`). This is what makes the OOD evaluation sets genuinely held out, and
  it is the role behind every $\tau$ / $R^2$ / L1 number in this document.
- **`paired`** — one width-independent seed, so the *same* 200 draws are instantiated and trained at
  every width. `paired[i]` at w128 is the same hyperparameter configuration as `paired[i]` at w16,
  differing only in channel count. Sharing draws across widths is the point, so this role is
  excluded from the disjointness assertion, is never read by `train_sizegen_predgen.py`, and is used
  for nothing except the `τ_HP` baseline below.

### Metrics

Four numbers are reported for every (condition, width, split). Write $y_i$ for the CNN's true test
accuracy and $\hat y_i = \mathrm{sigmoid}(\mathrm{net}(\cdot))$ for the prediction, $i = 1,\dots,n$, with
$n = 1000$ per width per arm on the `test` role.

Implementation:
`regression_metrics()` in [scripts/train_sizegen_predgen.py:152](../scripts/train_sizegen_predgen.py#L152).

**Kendall $\tau$** — the headline number, because the task is a ranking task ("which of these two CNNs
generalizes better"), not a calibration task. Over all $n(n-1)/2 = 499500$ pairs,

$$\tau_b = \frac{C - D}{\sqrt{(n_0-n_1)(n_0-n_2)}}, \qquad n_0 = n(n-1)/2$$

with $C$ concordant pairs ($y_i < y_j$ and $\hat y_i < \hat y_j$, or both reversed), $D$ discordant, and
$n_1$/$n_2$ the tie corrections in $\hat y$ / $y$ respectively (`scipy.stats.kendalltau`, default
`variant='b'`; ties are rare here — both quantities are continuous). Range $[-1, +1]$; $0$ = chance
ordering, $1$ = the exact ranking. It depends on the predictions only through their order, so it
is invariant to any monotone reparameterization — in particular to the affine scale drift that
dominates the OOD widths. That invariance is why $\tau$ is the primary metric and simultaneously why
it cannot be the only one: a model whose predictions are all off by $+0.1$ has a perfect $\tau$.

**$R^2$** — the coefficient of determination, `sklearn.metrics.r2_score`:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

i.e. the MSE relative to the constant predictor $\bar y$ (that width's mean label): $R^2 = 0.8$ means
the MSE is $20\%$ of what the constant-mean predictor would incur; $R^2 < 0$ means worse prediction
than the constant-mean predictor.

**L1** — mean absolute error, $\mathrm{mean}\,|\hat y_i - y_i|$, in **accuracy units**: $L1 = 0.0341$ is
3.4 accuracy points of average miss. The one metric with a directly physical scale; a sanity check on
$R^2$, which mixes error and label spread.

**$R^2_{recal}$** — $R^2$ *after* the best affine recalibration of the predictions, a deliberately
cheating diagnostic:

$$R^2_{recal} = \max_{a,b} R^2(y, a\hat y + b) = \mathrm{corr}(y,\hat y)^2 \quad \text{(squared Pearson)}$$

$R^2 \le R^2_{recal} \le 1$ always, $R^2_{recal}$ is invariant to affine (but not general monotone)
drift, and it is the *linear*-correlation counterpart of $\tau$'s rank correlation. Reading the pair:

| | `R²_recal` high | `R²_recal` low |
|---|---|---|
| **R² high** | model is right and calibrated | (impossible, `R² ≤ R²_recal`) |
| **R² low** | pure calibration failure — the signal is read, the scale drifted | genuine failure to read wide weights |

Experiment 2 additionally reports **paired** metrics against the w16 predictions (`τ(y₁₆, y_w)`,
`max |Δy|`, `mean |Δy|`) rather than against labels; those are a sharper test available only there,
because widening pairs each wide CNN with a specific w16 original — see
[Experiment 2](#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check).

Model selection is always on **val Kendall $\tau$ at the train width only**. No OOD width may influence
which LR or which checkpoint is kept.

---

## Input-network zoo quality

**Status: done 2026-08-22**, on the finished zoo (gate V5 cleared 2026-08-21, generation finished
2026-08-22 00:41). All numbers below are read off the `meta_{role}.jsonl` records
`generate_cnn_zoo.py` writes — no model is loaded and no CNN is re-run — by

```bash
source .env && python scripts/summarize_cnn_zoo_quality.py
```

on the `test` role (n = 1000 per width per arm), except `τ_HP`, which needs the `paired` role.

### Per-width accuracy distributions (labels)

The target *is* accuracy, and accuracy systematically improves with width. A metanetwork trained only
at w16 has never seen a label above w16's range, so the best it can do is rank within the range it was
trained on. The label distributions are a prerequisite for reading the Experiment 3 numbers: they say
how much of any OOD degradation is a shifted target rather than a failure to generalize across width,
and why the headline metric is a rank correlation ($\tau$) rather than an error on the accuracy itself.


| Width | arm | n | min | q25 | median | q75 | max | mean | frac ≤ 0.11 |
|---|---|---|---|---|---|---|---|---|---|
| w16 | SP ≡ muP16 | 1000 | 0.088 | 0.162 | 0.258 | 0.330 | 0.527 | 0.251 | 0.169 |
| w32 | SP | 1000 | 0.085 | 0.205 | 0.289 | 0.366 | 0.581 | 0.286 | 0.133 |
| w48 | SP | 1000 | 0.100 | 0.244 | 0.322 | 0.398 | 0.608 | 0.317 | 0.107 |
| w64 | SP | 1000 | 0.100 | 0.248 | 0.338 | 0.418 | 0.632 | 0.328 | 0.118 |
| w96 | SP | 1000 | 0.100 | 0.275 | 0.364 | 0.447 | 0.644 | 0.350 | 0.108 |
| w128 | SP | 1000 | 0.100 | 0.280 | 0.370 | 0.455 | 0.653 | 0.360 | 0.088 |
| w192 | SP | 1000 | 0.100 | 0.305 | 0.387 | 0.481 | 0.641 | 0.381 | 0.066 |
| w256 | SP | 1000 | 0.100 | 0.304 | 0.394 | 0.482 | 0.663 | 0.384 | 0.074 |
| w384 | SP | 1000 | 0.100 | 0.317 | 0.419 | 0.506 | 0.664 | 0.401 | 0.057 |
| w512 | SP | 1000 | 0.099 | 0.313 | 0.412 | 0.501 | 0.672 | 0.400 | 0.054 |
| w32 | muP16 | 1000 | 0.098 | 0.193 | 0.280 | 0.360 | 0.580 | 0.277 | 0.118 |
| w48 | muP16 | 1000 | 0.099 | 0.218 | 0.291 | 0.376 | 0.578 | 0.293 | 0.093 |
| w64 | muP16 | 1000 | 0.091 | 0.220 | 0.301 | 0.383 | 0.630 | 0.305 | 0.099 |
| w96 | muP16 | 1000 | 0.096 | 0.235 | 0.312 | 0.391 | 0.617 | 0.314 | 0.070 |
| w128 | muP16 | 1000 | 0.099 | 0.232 | 0.312 | 0.396 | 0.655 | 0.315 | 0.071 |
| w192 | muP16 | 1000 | 0.099 | 0.234 | 0.320 | 0.400 | 0.664 | 0.319 | 0.069 |
| w256 | muP16 | 1000 | 0.100 | 0.233 | 0.318 | 0.405 | 0.671 | 0.323 | 0.069 |
| w384 | muP16 | 1000 | 0.100 | 0.246 | 0.321 | 0.403 | 0.664 | 0.328 | 0.044 |
| w512 | muP16 | 1000 | 0.097 | 0.236 | 0.320 | 0.404 | 0.670 | 0.326 | 0.053 |

Support overlap with the training width, same role:

| Width | arm | `W₁(y₁₆, y_w)` | frac inside w16's [q1, q99] | median shift |
|---|---|---|---|---|
| w32 | SP | 0.0350 | 0.942 | +0.0312 |
| w48 | SP | 0.0662 | 0.889 | +0.0634 |
| w64 | SP | 0.0768 | 0.859 | +0.0793 |
| w96 | SP | 0.0994 | 0.818 | +0.1062 |
| w128 | SP | 0.1093 | 0.792 | +0.1115 |
| w192 | SP | 0.1298 | 0.728 | +0.1284 |
| w256 | SP | 0.1335 | 0.726 | +0.1361 |
| w384 | SP | 0.1505 | **0.654** | +0.1606 |
| w512 | SP | 0.1487 | **0.678** | +0.1536 |
| w32 | muP16 | 0.0257 | 0.958 | +0.0215 |
| w48 | muP16 | 0.0424 | 0.921 | +0.0331 |
| w64 | muP16 | 0.0540 | 0.884 | +0.0423 |
| w96 | muP16 | 0.0626 | 0.892 | +0.0538 |
| w128 | muP16 | 0.0645 | 0.880 | +0.0532 |
| w192 | muP16 | 0.0682 | 0.863 | +0.0621 |
| w256 | muP16 | 0.0719 | 0.863 | +0.0599 |
| w384 | muP16 | 0.0771 | 0.869 | +0.0630 |
| w512 | muP16 | 0.0749 | 0.864 | +0.0618 |

Through w128, the labels shift but stay inside w16's support: 79% (SP) / 88% (muP16) of w128
accuracies sit inside w16's central 98%.

Past w128 the arms diverge. muP16 stays flat — 0.863–0.869 inside w16's central 98% from w192 to
w512, `W₁ ≤ 0.077`, median shift only $+0.062$ — a valid target distribution over the full $32\times$
ladder, so an OOD $\tau$ drop there is a real metanetwork failure. SP is not: the fraction inside
support falls monotonically to **0.654 at w384** with the median shifted **$+0.16$**, i.e. a third of
the w384 SP labels are outside the range the metanetwork was trained on. Diagnostic 1 fires for SP at
w384/w512 — part of any SP decay past w128 is a shifted target, not a generalization failure. The
w512 row is marginally better than w384 on both metrics (0.678 vs 0.654): SP's mean accuracy is 0.401
at w384 and 0.400 at w512, i.e. the label distribution saturating, not a recovery.

### Per-width normalized spectral norms

$\|w\|_{Vn} = \sum_l \sqrt{n_{l-1}/n_l}\cdot\|W^{(l)}\|_2$ with convs flattened to
`[c_out, c_in·k_h·k_w]`, computed on the **effective** weights (a property of the represented function,
not of muP's internal variables), plus `W₁(μ₁₆, μ_w)`, the 1-Wasserstein distance between the w16 and
w_w distributions of that scalar. This is what the muP16 arm holds flat and the SP arm does not, and
what the Experiment 3 SP-vs-muP16 comparison rests on.

| Width | SP mean `‖w‖_Vn` | SP `W₁(μ₁₆, μ_w)` | muP16 mean `‖w‖_Vn` | muP16 `W₁(μ₁₆, μ_w)` |
|---|---|---|---|---|
| w16 | 38.84 | 0 (by definition) | 38.84 | 0 (by definition) |
| w32 | 53.04 | 14.20 | 39.41 | 1.34 |
| w48 | 67.16 | 28.32 | 39.44 | 1.47 |
| w64 | 79.11 | 40.27 | 40.66 | 1.89 |
| w96 | 94.44 | 55.60 | 40.12 | 1.38 |
| w128 | 111.12 | 72.27 | 39.49 | 1.64 |
| w192 | 142.63 | 103.79 | 39.23 | 1.33 |
| w256 | 176.45 | 137.61 | 39.53 | 1.22 |
| w384 | 242.08 | 203.24 | 38.78 | 2.11 |
| w512 | 300.21 | 261.37 | 38.14 | 1.92 |

Both arms hold over the full $32\times$ ladder. muP16's `W₁` is 1.2–2.1 at every width — within the
width-to-width noise of a 1000-model sample, and no larger at w512 than at w32 — against SP's 14 → 261,
i.e. about $40\times$ smaller at w128 and $136\times$ smaller at w512. muP16's mean `‖w‖_Vn` moves
from 38.84 to 38.14 over a $32\times$ width increase, a $-1.8\%$ drift; SP's grows $7.7\times$, against
$\sqrt{32} = 5.66$ predicted by $\Theta(\sqrt n)$ scaling. Per-layer means (conv1 / conv2 / conv3 / fc)
locate the growth in the two inner convs and the readout, the layers with two infinite dimensions or
an infinite fan-in:

- SP: w16 `1.63 / 19.19 / 14.95 / 3.08` → w128 `1.18 / 55.61 / 43.39 / 10.94` → w512 `1.04 / 149.05 / 121.31 / 28.80`
- muP16: w16 `1.63 / 19.19 / 14.95 / 3.08` → w128 `1.60 / 20.89 / 13.59 / 3.41` → w512 `1.71 / 20.62 / 12.39 / 3.42`

This is the CNN analogue of the INR-side result that $\|w\|_{Vn} = \Theta(\sqrt n)$ under SP and
$\Theta(1)$ under muP ($2.9\times$ over an $8\times$ width increase, against $\sqrt 8 = 2.83$;
$7.7\times$ over $32\times$, against $5.66$).

muP16 holds the weight norms and label support flat, but `τ_HP` still decays with width (next table):
the accuracy ranking itself is width-dependent regardless of parameterization, so a condition can be
reading the weights perfectly and still lose $\tau$.

### Baseline: ranking via hyperparameters (`τ_HP`)

A predictor could rank OOD wider models without reading the wide weights at all — recognizing which
hyperparameter draw produced the network (learning rate and training budget leave visible traces in
the weights) and reusing the accuracy that draw earned at w16. Whatever ranking survives from w16 to
w_w for a fixed draw is free, and a w16-trained model gets it for nothing. `τ_HP` measures exactly that
free part: the floor a result must clear before "it generalizes across width" is the right description.
$\tau_{HP} \approx 1$ would make the OOD problem trivial; $\tau_{HP} \approx 0$ would make a low
Experiment 3 $\tau$ say nothing about the metanetwork — only the intermediate regime makes the
experiment informative.

Computing this needs the *same* hyperparameters instantiated at both widths, which is why the
[`paired` role](#roles) exists — it is used for nothing else. `τ_HP(16, w)` is the Kendall $\tau$
between its w16 and w_w accuracy vectors, $n = 200$, standard error $\approx \pm 0.05$ (the
non-monotonicity between w32 and w48 is within noise).

| Width | SP `τ_HP` | muP16 `τ_HP` |
|---|---|---|
| w32 | 0.7315 | 0.7788 |
| w48 | 0.7752 | 0.7481 |
| w64 | 0.7006 | 0.6414 |
| w96 | 0.6360 | 0.6463 |
| w128 | 0.6401 | 0.5985 |
| w192 | 0.6020 | 0.5746 |
| w256 | 0.5303 | 0.5791 |
| w384 | 0.5064 | 0.5413 |
| w512 | 0.4573 | 0.5238 |

The ceiling keeps falling to $32\times$: `τ_HP` reaches **0.457 (SP) / 0.524 (muP16)** at w512, so at
w512 a hyperparameter-only predictor retains less than half the ranking. Two consequences for reading
the v2 columns:

1. The bar to clear drops as width grows, so a condition holding $\tau \approx 0.91$ at w512 (as the
   conv matrix-product ScaleGMN does on muP16) clears its ceiling by **0.39** — a stronger statement
   than the same $\tau$ at w128. The `τ_HP` row, not the $\tau$ row, is what makes $32\times$ worth
   running.
2. The arms cross over: through w128 SP's ceiling is at or above muP16's (0.640 vs 0.599 at w128);
   from w256 on it is below (0.457 vs 0.524 at w512). The two arms' conditions must each be compared
   to their own ceiling at each width, not a single study-wide bar.

$n = 200$ per width gives $\pm 0.05$ standard error: the w128 → w512 decline (0.64 → 0.46 SP) is
$\approx 3.5$ standard errors.

---

## Experiment 1 — Reproduce Accuracy Prediction at the Zoo's Native Width (w16)

**TL;DR**: ScaleGMN reaches **$\tau = 0.9316$** on upstream's data and splits, matching the published
$\approx 0.93$ bar, with the plain GMN below it at 0.9266.

**Goal**: confirm the plumbing reproduces ScaleGMN's published accuracy-prediction number before
anything is changed. This is the only experiment that uses upstream's trainer
(`src/scalegmn/predicting_generalization.py`, via `scripts/train_predgen.py`), upstream's data
(`$ANYDIM_DATA_ROOT/cifar10_zoo/weights.npy` + `cifar10_split.csv`) and upstream's splits;
everything after it goes through `scripts/train_sizegen_predgen.py` and our reader.

**Dataset**: upstream small CNN zoo at the native width w16, split **6024 train / 1535 val /
7567 test**. 200 epochs, `--direction forward`.
**Configs**: `configs/cifar10_predgen/{scalegmn,gmn}_predgen_sp.yml`
**Date**: 2026-08-21

### Reproduction

```bash
source .env

screen -dmS predgen_sgmn bash -c 'source .env && CUDA_VISIBLE_DEVICES=0 \
  python scripts/train_predgen.py --conf configs/cifar10_predgen/scalegmn_predgen_sp.yml --wandb True'
screen -dmS predgen_gmn bash -c 'source .env && CUDA_VISIBLE_DEVICES=1 \
  python scripts/train_predgen.py --conf configs/cifar10_predgen/gmn_predgen_sp.yml --wandb True'
```

### Results

| Variant | Best val τ | Test τ | Test R² | Test L1 | Run ID |
|---|---|---|---|---|---|
| ScaleGMN (forward, scale) | 0.9314 | **0.9316** | 0.9915 | 0.0078 | [`aqjwzgsv`](https://wandb.ai/yuxinma/cifar10_predgen/runs/aqjwzgsv) |
| Plain GMN (forward, permutation) | 0.9291 | 0.9266 | 0.9912 | 0.0082 | [`66qfzg9s`](https://wandb.ai/yuxinma/cifar10_predgen/runs/66qfzg9s) |

**PASS.** Both test numbers are taken at the best-val-$\tau$ epoch, not as a running max over epochs
(upstream's `test/best_tau` panel is a running max; for ScaleGMN that would read 0.9319 instead of
0.9316, for the plain GMN it coincides). The scale-equivariant inductive bias buys $\Delta\tau \approx
0.005$ on this fixed-width benchmark, against the $\approx 1.5$ pp accuracy gap it buys on MNIST-INR
classification.

---

## Experiment 2 — Size Generalization on Wider Equivalent CNNs (sanity check)

**Goal**: take the zoo's w16 test CNNs, widen them to w32, w48, w64, w96, w128 by a
*function-preserving* construction, and predict accuracy from the widened weights. The widened CNN
computes exactly the same function as its w16 source and carries that source's test accuracy as its
label, so every change in $\tau$ is the metanetwork failing to be invariant to the widening — never
distribution shift, never a label shift. Compare against
[Experiment 3](#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512),
where the wider CNNs are trained from random initialization, and with
[MNIST Experiment 2](EXPERIMENTS_MNIST_INR_classification.md#experiment-2--size-generalization-on-wider-equivalent-networks-sanity-check).

Two widening families are scored, both defined in
[VERIFICATION_gmn_properties.md](VERIFICATION_gmn_properties.md) § Equivalence classes (CNN graphs).
Writing the widened kernel blockwise per kernel offset $s$,
$W^{(l)}_{\mathrm{up}}[i k_l + a,\, j k_{l-1} + b,\, s] = B^{(l)}_{ij}[a, b, s]$, with
$b^{(l)}_{\mathrm{up}} = b^{(l)} \otimes \mathbf{1}_{k_l}$ and $k_0 = k_L = 1$ (the input channel and
the $10$ logits are never widened):

| Family | $B^{(l)}_{ij}[\cdot,\cdot,s]$ | Conditions |
|---|---|---|
| `general` | independent random block per $(i,j)$, **row** condition only | $\sum_b B^{(l)}_{ij}[a,b,s] = W^{(l)}[i,j,s]$ |
| `uniform` | $W^{(l)}[i,j,s]\,\mathbf{1}\mathbf{1}^{\mathsf{T}}/k_{l-1}$ (channel Kronecker) | row **and** $\sum_a B^{(l)}_{ij}[a,b,s] = (k_l/k_{l-1})\,W^{(l)}[i,j,s]$ |

The row condition alone preserves the function, so `general` is the **largest** function-preserving
widening and `uniform` is a measure-zero special case of it. `uniform` is the reference: it is the
family the dup-equiv modifications (fan-in rescaling + mean aggregation + `LastLayerReadout`) are
built for, so reading a `general` number against it separates "not invariant to this family" from
"cannot handle a wide graph at all".

**9 in-scope conditions of 12**: ScaleGMN and plain GMN each contribute
$\{\text{baseline}, \text{dup-equiv}\}$, mp-GMN and mp-ScaleGMN contribute dup-equiv only (the
matrix-product MSG constraint is defined on top of fan-in rescaling + mean aggregation, so there is no
baseline variant), each crossed with $\{\text{forward}, \text{bidirectional}\}$:

| Condition | forward | bidirectional |
|---|---|---|
| ScaleGMN baseline / dup-equiv | ✓ / ✓ | out of scope |
| plain GMN baseline / dup-equiv | ✓ / ✓ | ✓ / ✓ |
| mp-GMN dup-equiv | ✓ | ✓ |
| mp-ScaleGMN dup-equiv | ✓ | out of scope |

**mp-ScaleGMN is forward-only in every experiment**, and the bidirectional `symmetry: scale` cells are
out of scope on CNN graphs — the reciprocal backward edge feature overflows float32 on this zoo
([`src/models/bidir_reciprocal.py`](../src/models/bidir_reciprocal.py); the ScaleGMN paper's App. A.2
reports the same). Those cells are scope, not pending runs.

**No SP / muP distinction here**: the parameterization only affects how CNNs of *different* widths are
*trained*, and no wide CNN is ever trained in this experiment. All 9 metanetworks are trained at w16
on `w16_zoo_dup/` — the one base-width copy on disk, shared by both families since $k = 1$ pins every
block to the base kernel — and are then scored on both families' widened trees, $n = 2000$ per width.
**No LR sweep**: this is a property check, not a performance comparison, so every condition uses the
config's LR ($10^{-3}$), and model selection is on val $\tau$ at w16, the only width trained on.

**Function preservation, measured** on the stored float32 tensors ($10$ models per width, logits of a
$\mathcal{N}(0, 1)$ probe): `general` gives $\max|\Delta\text{logits}|$ $2.9\times10^{-6}$ (w32) to
$6.5\times10^{-5}$ (w128), with row residual $\le 9.5\times10^{-7}$, column residual $4.92$–$12.5$ and
minimum non-uniformity $2.62$–$4.18$ — it genuinely violates the column condition and never
degenerates to the uniform block. `uniform` gives $\le 1.5\times10^{-6}$, nonzero only at $k = 3, 6$,
where $1/k_{l-1}$ is not representable in float32. Blocks are drawn in float64 and cast on write:
`general`'s are $O(1)$ draws whose row sums cancel to an $O(10^{-2})$ kernel, so a float32
construction would lose the row condition outright.

The evaluator runs in **full fp32, not TF32**. The `max |Δy|` floor is $\approx 10^{-7}$ on `uniform`
and larger on `general`, whose row residual is $\approx 10^{-4}$ *relative* on an $O(10^{-2})$ kernel;
TF32 would add $\approx 10^{-4}$ of its own noise on top. `duplication_equiv` and `readout` are read
back off each checkpoint rather than re-specified.

**Scripts**: `scripts/generate_duplicated_cnns.py --family` (data),
`scripts/eval_sizegen_predgen_duplicated.py` (eval), `scripts/run_predgen_widen_families.sh`
(orchestration).
**Configs**: `configs/cifar10_predgen/{scalegmn,gmn,mpgmn,mpsgmn}_dup_w16.yml`
**Date**: 2026-09-12 (17:22–18:22 on GPUs 0/1/2: 5 trainings, then 4 eval jobs). Supersedes the
2026-08-21 forward-only `uniform` run of 4 conditions, whose numbers the forward `uniform` rows below
reproduce exactly from the same four checkpoints.

### Reproduction

```bash
source .env

# Data: the `general` tree (w32-w128, 4.7 GB); `uniform` is w{16..128}_zoo_dup/, already on disk
python scripts/generate_duplicated_cnns.py --family general

# Phase 1 trains the 5 conditions the 4-condition run never covered (mp-GMN fw, mp-ScaleGMN fw,
# plain GMN bd x2, mp-GMN bd); phase 2 scores {general,uniform} x {forward,bidirectional}, each
# job with --model-type all. Resumable: re-running skips finished jobs.
screen -dmS widen_cnn bash -c 'bash scripts/run_predgen_widen_families.sh cifar10 "0 1 2"'
```

**Logs**: `/tmp/predgen_cifar10_widen_families/` (`SUMMARY.md`, per-job logs,
`preds_{family}_{direction}.npz`; one worker per GPU draining a job queue).

Two families of numbers per width: against the **labels** ($\tau$ / $R^2$ / L1 / $R^2_{recal}$), and
against the **w16 predictions** (`τ(y₁₆, y_w)`, `max |Δy|`, `mean |Δy|`) — the sharp form, since
pairing is by index and every width's split lists the same base models in the same order. `max_drop`
is $\tau(\text{w16}) - \min_w \tau(w)$.

### `general` widening — the largest function-preserving family

Eval runs [`q5aqjcna`](https://wandb.ai/yuxinma/cifar10_predgen/runs/q5aqjcna) (forward) and
[`sehq7vhu`](https://wandb.ai/yuxinma/cifar10_predgen/runs/sehq7vhu) (bidirectional), extended
2026-09-15 to w192/w256/w384/w512 (superseding the earlier w16–w128 run of the same name).

#### ScaleGMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | +0.990 | +0.657 | +0.566 | +0.469 | +0.344 | +0.248 | +0.138 | +0.074 | +0.003 | −0.033 | 1.023 |
| ScaleGMN baseline, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| ScaleGMN dup-equiv, fw | +0.990 | +0.448 | +0.164 | −0.020 | −0.194 | −0.286 | −0.375 | −0.421 | −0.474 | −0.504 | 1.495 |
| ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **mp-ScaleGMN dup-equiv, fw** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **0.000** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.990 \to +0.248$ / $0.0084 \to 0.0919$ /
$+0.991 \to +0.390$; dup-equiv $+0.990 \to -0.286$ / $0.0083 \to 0.1216$ / $+0.990 \to +0.308$;
mp-ScaleGMN $+0.992$ / $0.0075$ / $+0.992$ at every width.

#### Plain GMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | +0.990 | +0.042 | −0.399 | −0.580 | −0.752 | −0.830 | −0.901 | −0.933 | −0.962 | −0.976 | 1.966 |
| GMN baseline, bd | +0.991 | −0.046 | −0.423 | −0.597 | −0.788 | −0.895 | −0.996 | −1.044 | −1.092 | −1.118 | 2.109 |
| GMN dup-equiv, fw | +0.991 | −1.076 | −1.486 | −1.649 | −1.777 | −1.828 | −1.875 | −1.893 | −1.911 | −1.919 | 2.910 |
| GMN dup-equiv, bd | +0.991 | −1.089 | −1.373 | −1.489 | −1.612 | −1.662 | −1.720 | −1.744 | −1.771 | −1.784 | 2.775 |
| **mp-GMN dup-equiv, fw** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| mp-GMN dup-equiv, bd | +0.991 | −0.434 | −0.808 | −0.904 | −1.083 | −1.133 | −1.176 | −1.156 | −1.130 | −1.092 | 2.167 |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.990 \to -0.830$ / $0.0092 \to 0.1507$ /
$+0.991 \to +0.349$ (fw) and $+0.991 \to -0.895$ / $0.0086 \to 0.1508$ / $+0.991 \to +0.235$ (bd);
dup-equiv $+0.991 \to -1.828$ / $0.0085 \to 0.1984$ / $+0.991 \to +0.409$ (fw) and
$+0.991 \to -1.662$ / $0.0084 \to 0.1856$ / $+0.991 \to +0.112$ (bd); mp-GMN $+0.991$ / $0.0079$ /
$+0.992$ at every width (fw) and $+0.991 \to -1.133$ / $0.0084 \to 0.1627$ / $+0.991 \to +0.001$ (bd).

#### Agreement with the w16 predictions

`max |Δy|`:

| Condition | w32 | w48 | w64 | w96 | w128 |
|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 2.8e-01 | 2.8e-01 | 3.0e-01 | 3.1e-01 | 3.0e-01 |
| ScaleGMN dup-equiv, fw | 3.2e-01 | 3.3e-01 | 3.3e-01 | 3.3e-01 | 3.5e-01 |
| **mp-ScaleGMN dup-equiv, fw** | **5.3e-06** | **2.6e-05** | **5.5e-05** | **1.0e-04** | **2.4e-04** |
| GMN baseline, fw | 3.9e-01 | 3.8e-01 | 3.8e-01 | 4.0e-01 | 4.0e-01 |
| GMN baseline, bd | 3.0e-01 | 3.3e-01 | 3.4e-01 | 3.4e-01 | 3.5e-01 |
| GMN dup-equiv, fw | 3.9e-01 | 4.0e-01 | 3.9e-01 | 4.0e-01 | 4.0e-01 |
| GMN dup-equiv, bd | 4.1e-01 | 4.3e-01 | 4.2e-01 | 4.2e-01 | 4.2e-01 |
| **mp-GMN dup-equiv, fw** | **7.0e-06** | **3.3e-05** | **6.1e-05** | **9.3e-05** | **2.5e-04** |
| mp-GMN dup-equiv, bd | 4.4e-01 | 4.4e-01 | 4.3e-01 | 4.3e-01 | 4.3e-01 |

The two forward matrix-product rows are the `general` floor, not zero: the row condition itself holds
only to $\approx 10^{-4}$ relative in float32, giving `mean |Δy|` $1.4\times10^{-7} \to
4.7\times10^{-6}$ (mp-ScaleGMN) / $2.1\times10^{-7} \to 9.1\times10^{-6}$ (mp-GMN) and
$\tau(y_{16}, y_w) \ge 0.9948$ / $\ge 0.9986$. Every other row's $\tau(y_{16}, y_w)$ is within
$0.008$ of its label $\tau$, with `mean |Δy|` $6.0\times10^{-2}$–$2.0\times10^{-1}$.

### `uniform` (Kronecker) widening — the reference family

Eval runs [`b6aqp2d2`](https://wandb.ai/yuxinma/cifar10_predgen/runs/b6aqp2d2) (forward) and
[`7qmcrmox`](https://wandb.ai/yuxinma/cifar10_predgen/runs/7qmcrmox) (bidirectional), extended
2026-09-15 to w192/w256/w384/w512 (superseding the earlier w16–w128 run of the same name).

#### ScaleGMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | +0.990 | +0.988 | +0.979 | +0.968 | +0.939 | +0.906 | +0.834 | +0.752 | +0.570 | +0.380 | 0.611 |
| ScaleGMN baseline, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **ScaleGMN dup-equiv, fw** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **+0.990** | **0.000** |
| ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **mp-ScaleGMN dup-equiv, fw** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **0.000** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.990 \to +0.906$ / $0.0084 \to 0.0341$ /
$+0.991 \to +0.972$; dup-equiv $+0.990$ / $0.0083$ / $+0.990$ and mp-ScaleGMN $+0.992$ / $0.0075$ /
$+0.992$ at every width. The baseline's predictions shrink toward the mean (`pred_mean`
$0.2828 \to 0.2492$ against `actual_mean` $0.2809$).

#### Plain GMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | +0.990 | +0.877 | +0.717 | +0.586 | +0.430 | +0.360 | +0.309 | +0.294 | +0.286 | +0.285 | 0.705 |
| GMN baseline, bd | +0.991 | +0.864 | +0.667 | +0.506 | +0.336 | +0.277 | +0.248 | +0.246 | +0.253 | +0.260 | 0.745 |
| **GMN dup-equiv, fw** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| **GMN dup-equiv, bd** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| **mp-GMN dup-equiv, fw** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| **mp-GMN dup-equiv, bd** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.990 \to +0.360$ / $0.0092 \to 0.0814$ /
$+0.991 \to +0.834$ (fw) and $+0.991 \to +0.277$ / $0.0086 \to 0.0877$ / $+0.991 \to +0.832$ (bd);
dup-equiv $+0.991$ / $0.0085$ / $+0.991$ (fw) and $+0.991$ / $0.0084$ / $+0.991$ (bd), mp-GMN
$+0.991$ / $0.0079$ / $+0.992$ (fw) and $+0.991$ / $0.0084$ / $+0.991$ (bd), all flat.

#### Agreement with the w16 predictions

`max |Δy|`:

| Condition | w32 | w48 | w64 | w96 | w128 |
|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 7.7e-02 | 9.0e-02 | 9.7e-02 | 1.1e-01 | 1.2e-01 |
| **ScaleGMN dup-equiv, fw** | **6.0e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** |
| **mp-ScaleGMN dup-equiv, fw** | **6.0e-08** | **6.0e-08** | **6.0e-08** | **6.0e-08** | **6.7e-08** |
| GMN baseline, fw | 1.1e-01 | 1.9e-01 | 2.1e-01 | 2.8e-01 | 3.1e-01 |
| GMN baseline, bd | 1.7e-01 | 1.9e-01 | 2.7e-01 | 3.0e-01 | 3.2e-01 |
| **GMN dup-equiv, fw** | **8.9e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** |
| **GMN dup-equiv, bd** | **8.9e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** | **8.9e-08** |
| **mp-GMN dup-equiv, fw** | **8.9e-08** | **8.9e-08** | **6.7e-08** | **6.0e-08** | **8.9e-08** |
| **mp-GMN dup-equiv, bd** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **1.2e-07** |

All six invariant rows have $\tau(y_{16}, y_w) = 1.0000$ and `mean |Δy|`
$7.4\times10^{-9}$–$9.7\times10^{-9}$ at every width — the float32 floor, from the $\approx 10^{-7}$ relative
rounding at $k = 3, 6$ plus summation reordering, not zero. The baselines:
$\tau(y_{16}, y_w)$ $0.9730 \to 0.9174$ with `mean |Δy|` $7.5\text{e-}03 \to 3.5\text{e-}02$
(ScaleGMN fw), $0.9132 \to 0.6925$ / $3.4\text{e-}02 \to 7.8\text{e-}02$ (GMN fw),
$0.8948 \to 0.6947$ / $3.8\text{e-}02 \to 8.6\text{e-}02$ (GMN bd).

### Checkpoints

All 9 trained at w16 on `w16_zoo_dup/`, selected on val $\tau$ at w16, and scored on both families.

| Condition | Checkpoint | Training run | Best val $\tau$ (w16) |
|---|---|---|---|
| ScaleGMN baseline, fw | `predgen-dup-sgmn-base_zrj81jjz` | [`zrj81jjz`](https://wandb.ai/yuxinma/cifar10_predgen/runs/zrj81jjz) | 0.9301 (ep 186) |
| ScaleGMN dup-equiv, fw | `predgen-dup-sgmn-deq_uqvstjby` | [`uqvstjby`](https://wandb.ai/yuxinma/cifar10_predgen/runs/uqvstjby) | 0.9270 (ep 190) |
| mp-ScaleGMN dup-equiv, fw | `predgen-dup-mpsgmn-deq_yuv9n90k` | [`yuv9n90k`](https://wandb.ai/yuxinma/cifar10_predgen/runs/yuv9n90k) | 0.9312 (ep 180) |
| GMN baseline, fw | `predgen-dup-gmn-base_62c9p3pf` | [`62c9p3pf`](https://wandb.ai/yuxinma/cifar10_predgen/runs/62c9p3pf) | 0.9255 (ep 134) |
| GMN baseline, bd | `predgen-dup-gmn-base-bidir_osanjg9n` | [`osanjg9n`](https://wandb.ai/yuxinma/cifar10_predgen/runs/osanjg9n) | 0.9275 (ep 196) |
| GMN dup-equiv, fw | `predgen-dup-gmn-deq_6fj34ie1` | [`6fj34ie1`](https://wandb.ai/yuxinma/cifar10_predgen/runs/6fj34ie1) | 0.9239 (ep 187) |
| GMN dup-equiv, bd | `predgen-dup-gmn-deq-bidir_6esfucjz` | [`6esfucjz`](https://wandb.ai/yuxinma/cifar10_predgen/runs/6esfucjz) | 0.9265 (ep 187) |
| mp-GMN dup-equiv, fw | `predgen-dup-mpgmn-deq_nin0y3zi` | [`nin0y3zi`](https://wandb.ai/yuxinma/cifar10_predgen/runs/nin0y3zi) | 0.9291 (ep 190) |
| mp-GMN dup-equiv, bd | `predgen-dup-mpgmn-deq-bidir_hfucf5za` | [`hfucf5za`](https://wandb.ai/yuxinma/cifar10_predgen/runs/hfucf5za) | 0.9283 (ep 192) |

Predictions: `/tmp/predgen_cifar10_widen_families/preds_{general,uniform}_{forward,bidirectional}.npz`.

### Figures

`scripts/plot_widen_families.py --dataset cifar10` (reads the tables above):

| Figure | Contents |
|---|---|
| `results/figures/widen_cnn_gmn.png` | GMN baseline / dup-equiv / mp-GMN, `general` \| `uniform` panels |
| `results/figures/widen_cnn_scalegmn.png` | ScaleGMN baseline / dup-equiv / mp-ScaleGMN, same panels |

Solid forward, dashed bidirectional; colour + marker carry the model, matching every other figure in
`results/figures/`.

---

## Experiment 3 — Size Generalization on Our Multi-Width Zoo (train w16, test w16–w512)

**Goal**: the flagship result — train the metanetwork on w16 CNNs only and predict the test accuracy
of w32–w512 CNNs it has never seen, on a zoo we generate ourselves at ten widths so that width is a
controlled variable. Same question as
[MNIST Experiment 3](EXPERIMENTS_MNIST_INR_classification.md#experiment-3--size-generalization-train-w24-test-w24w1024),
on a task where the label is a property of the input network rather than of the data it encodes.

**Setup**:
- Train on **w16**, the zoo's native width, so the training set is the same distribution Experiment 1
  already reproduced a published number on.
- muP16 uses **base width 16** = the training width, so w16 muP16 $\equiv$ w16 SP exactly and the two
  data arms differ only at the OOD widths.
- Test widths w16, w32, w48, w64, w96, w128 ($8\times$ the training width), extended to w192, w256,
  w384, w512 ($32\times$) by re-scoring the same checkpoints — no retraining (see Data below). Eval
  batch size scales as $bs\cdot(16/w)^2$ to keep activation memory flat, flooring at 1 from w256 on.
- Both message-passing directions were run: the forward variant in each model's `#### Forward`
  subsection below (tables run the full w16–w512 ladder), and the bidirectional variant in each
  model's `#### Bidirectional` subsection. The six valid `symmetry: permutation` bidirectional
  conditions (plain GMN, conv matrix-product GMN) were extended to the full w16–w512 ladder by the
  same eval-only re-score as forward (2026-08-31); the four invalid `symmetry: scale` (ScaleGMN)
  bidirectional conditions remain w16–w128 only and out of scope — see the warning below. The **conv
  matrix-product ScaleGMN** (`mpsgmn`, dup-equiv only) was added forward-only,
  already scored on the full ladder — see
  [its own subsection](#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward).

### Data

- **SP**: `$ANYDIM_DATA_ROOT/cnn_zoo/w{16,32,48,64,96,128}_sp/` plus `w{192,256,384,512}_sp/` — the
  sampled hyperparameters instantiated as-is at every width.
- **muP16**: `w{16,...,128}_mup16/` plus `w{192,...,512}_mup16/` — muP with base width 16,
  implemented in reference coordinates with an explicit readout multiplier `1/wm` in the forward
  pass; the exported `.pth` holds the effective weights `W_eff = W/wm` (see
  `src/data/train_zoo_cnns.py`).

Splits: `cifar10_predgen_splits.json` per width directory (`scripts/generate_predgen_splits.py`).
Generation: `scripts/generate_cnn_zoo.py` (w16–w128), `scripts/run_cifar10_zoo_gen_v2.sh` (w192–w512).

The width range was built in two stages, which is only a data-provenance detail — all w16 → w512
numbers in the tables below come from one evaluation protocol on checkpoints trained once:

1. **w16–w128** (2026-08-21 19:00 → 2026-08-22 00:41): 5.9 h wall on five GPUs, $\approx 26$ GPU-hours
   (against $\approx 100$ predicted from $c^2$ scaling — the small widths are latency- rather than
   FLOP-bound). 8000 train / 1000 val at w16 only, 1000 test + 200 paired at each of the six widths,
   per arm; 675 GB on disk before the clone fix below. w16 muP16 is hard-linked from w16 SP, so the
   base-width identity is exact by construction rather than by numerics.
2. **w192–w512** (2026-08-25 22:14 → 2026-08-29 02:35): one worker on GPU 4, widest-first, since the
   rest of the pool was running the FMNIST bidirectional queue and other users' jobs. **Nothing is
   retrained**: each new width's test hyperparameters come from a width-specific seed (`3000 +
   width`), so a new width adds fresh draws without touching any existing width's train/val/test —
   verified rather than assumed, since regenerating both arms' splits over the six pre-existing widths
   reproduced **all 12 JSONs byte-identically** (w16 md5 `dd33c6c16db241b3ca05e4e35e8afa8f`, unchanged)
   with `disjointness: OK (15000 distinct draws across train/val/test)`. Both `finalize()` split
   rewrites confirmed this on the real generation too: `splits_md5_changed.txt` came out **empty** —
   none of the eight pre-existing splits changed. Each condition therefore reaches w512 via one
   `--eval-only-ckpt` forward pass, the same reuse MNIST v4 made of the v3 checkpoints.

Quality diagnostics over the full ladder: [Input-network zoo quality](#input-network-zoo-quality).

### Model and training setup

**Model variants**: ScaleGMN (`symmetry: scale`) and plain GMN (`symmetry: permutation`), each in
{baseline, dup-equiv}, on each of {SP, muP16} = 8 conditions, plus the **conv matrix-product GMN**
(`message_fn_type: matrix_product_conv`, dup-equiv only, no baseline variant) on {SP, muP16} = **10
conditions total**, each run in both message-passing directions (forward:
[ScaleGMN](#scalegmn-results-symmetry-scale-forward-and-bidirectional) /
[Plain GMN](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional) /
[Conv matrix-product GMN](#conv-matrix-product-gmn-results-message_fn_type-matrix_product_conv-forward-and-bidirectional);
bidirectional: each model's own `#### Bidirectional` subsection). The **conv matrix-product ScaleGMN**
(`mpsgmn`, dup-equiv only) was added on top of these ten, forward only — see
[its subsection](#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward).

Baseline = `aggregator: add`, raw edge features, no fan-in rescaling.
Dup-equiv = `aggregator: mean` + fan-in rescaled edges. The CNN fan-in factor is **`n_in`, not
`n_in·k_h·k_w`**: the graph puts one edge per input channel carrying the whole zero-padded R⁹ kernel,
and mean aggregation averages over edges.
All variants: d_hid=128, 4 GNN layers, batch_size=64 at w16, `readout_range` forced to `full_graph`.

**Readout**: `LastLayerReadout` (`src/models/last_layer_readout.py`) — reads only the 10 output
nodes, selected via `node2type`. It is duplication-invariant *by construction* (`k_L = 1`), so
dup-equiv needs only fan-in + mean here, unlike the INR side where `LayerWiseMeanReadout` is
required. Upstream's `readout_range: last_layer` path is unusable OOD (it reshapes with the *train*
width's `num_nodes`); both replacements install into the `full_graph` slot.
`edge_pos_embed: False` everywhere — upstream's `get_edge_types()` disagrees with `cnn_to_tg_data`'s
edge order once the output layer has more than one node.

**Bidirectional works on CNN graphs — verified, not assumed.** `cnn_to_tg_data` fills `bw_edge_attr`
with `reciprocal(edge_attr)`, which on a CNN graph is largely `inf` (the R⁹ kernel slots outside the
3×3 support are zero-padded). That field is **dead**: `ScaleGMN_GNN_bidir.forward` sets
`bw_edge_attr = batch.edge_attr` at layer 0 and reads only `batch.bw_edge_index`. Smoke-tested before
launch across all five (model, variant) pairs at w16 and w128 — all ten forwards finite, while
`batch.bw_edge_attr` carried 20480 `inf`s of 615168 entries at w128.

> **⚠ The four bidirectional `symmetry: scale` conditions (ScaleGMN, both variants, both arms) are
> invalid** — a bidirectional ScaleGMN is *defined* with the reciprocal backward edge feature
> `1/W[u,v]`: with `bw_edge_attr = batch.edge_attr` a backward message into `u` carries `λ_v²/λ_u`
> instead of `λ_u`, so the model has no scale equivariance at all. Upstream asks for exactly that
> feature with `reciprocal: True`, assigns it at `models.py:305`, and never applies it. Measured in
> [VERIFICATION_gmn_properties.md § Scale equivariance on CNN graphs](VERIFICATION_gmn_properties.md#verification-of-scale-equivariance-1):
> the shipped code path fails by 4–5 orders of magnitude on conv graphs (2.0e-3 node / 2.7e-2 graph),
> and with `src/models/bidir_reciprocal.py` installed (the `1/0 := 0` pseudo-inverse convention, finite
> even though `torch.reciprocal(edge_attr)` is `inf` on 20.7% of components at c=16) the same weights
> are exactly equivariant (1.5e-8 / 1.9e-7, the float32 floor). `train_sizegen_predgen.py` and
> `eval_sizegen_predgen_duplicated.py` install the fix by default (`--bidir-reciprocal False` restores
> the old behaviour). **A retrain was attempted (`scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh`)
> and dropped, not queued** — every LR NaNs by training step ~13. Root-cause trace: the smallest
> nonzero `|W|` across the whole `cnn_zoo/w16_mup16` zoo is ~1.2e-15, whose reciprocal (~8.4e14)
> overflows float32 inside `EquivariantNet`'s unnormalized io-node branch (`mlp_on_io: True`,
> `src/scalegmn/src/scalegmn/layers.py:457`) after a few rounds of `bw_update_edge_attr_fn` threading;
> patching that branch-selection to `torch.where` (avoiding upstream's `0 * inf/nan = nan` masking
> leak) reduces but does not eliminate it — by round 3 the selected branch overflows directly, not
> just via the leak. This is not a bug unique to this repo: the ScaleGMN paper (Kalogeropoulos et al.,
> NeurIPS'24, arXiv:2406.10685) Appendix A.2 defines the identical construction and reports the same
> class of instability for elementwise-division backward messages under positive-scale symmetry,
> works around it only for `symmetry: sign` (where `1/q=q` needs no division), and reports no
> bidirectional positive-scale result anywhere in the paper. Implementation:
> [`src/models/bidir_reciprocal.py`](../src/models/bidir_reciprocal.py).
> This instability is CNN-specific — the INR-side retrains
> ([MNIST](EXPERIMENTS_MNIST_INR_classification.md), [FMNIST](EXPERIMENTS_FMNIST_INR_classification.md))
> hit no such issue and stand as recorded. The six `symmetry: permutation` bidirectional rows (plain
> GMN, conv matrix-product GMN) are unaffected — the fix is a no-op there — and the
> direction-neutrality finding below rests on those six rows alone.

**Every condition trains once and is scored under both arms.** Because base width = train width,
`w16_mup16/` is hard-linked from `w16_sp/`, and both directories carry the same split JSON, so the two
arms consume byte-identical inputs under the same seed — verified for every forward pair (same
selected LR, same `val_tau` at every epoch, `max |Δw| = 0.00e+00` across every parameter tensor,
bit-identical w16 test predictions) and audited per bidirectional condition with a printed
**`w16 identity`** line (the two arms' w16 rows must agree to the last printed digit). The
bidirectional arm goes further and trains only **five** jobs for all ten conditions (one per model ×
variant, under the muP16 config, then `--eval-only-ckpt` under the SP config), which is why it cost
75.9 GPU-h against the forward arm's $\approx 300$ for the same ten conditions. The conv
matrix-product ScaleGMN uses the same trick across its two *width ranges* rather than its two arms:
one training under the muP16 config, scored under {SP, muP16} × {w16–w128, w16–w512}, each pair
audited the same way (`v1 agreement: OK` — a w16–w512 eval's w16–w128 $\tau$ must reproduce the
w16–w128 log's).

**Training**: 200 epochs per condition, model selection on val $\tau$ at w16 only, then one eval-only
re-score that dumps the per-width `(pred, actual)` arrays the diagnostics need (`--save-predictions`).

**Metanetwork LR protocol**: a 7-value sweep (6.25e-5 to 4e-3, $\times 2$ grid, 8600 steps) per
condition, selected on best val $\tau$ at w16, then full training at that LR — same protocol as MNIST
Experiment 3, selected in-distribution only, which is what keeps the OOD numbers honest.

**Date**: zoo (w16–w128) generated 2026-08-21/22; ten forward conditions trained 2026-08-22 00:45 →
2026-08-25 04:37; zoo extended to w192–w512 2026-08-25 22:14 → 2026-08-29 02:35 and the ten forward
checkpoints re-scored to w512 by 2026-08-29 11:45 (no retraining); conv matrix-product ScaleGMN trained
2026-08-28 17:48 → 2026-08-29 07:29, scored on both width ranges; bidirectional arm (five trainings,
ten conditions) 2026-08-26 18:02 → 2026-08-28 14:05, its four `symmetry: scale` conditions invalidated
2026-08-29 and a retrain attempted and dropped as out of scope 2026-08-30/31 (see the warning above).

### Reproduction

```bash
source .env

# Zoo generation (w16-w128) + both arms' splits, one (arm, width) job per GPU worker, longest-first.
bash scripts/run_cifar10_zoo_gen.sh "0 4 5"

# 10 conditions, forward: sweep -> 200-epoch training -> eval-only re-score, one worker per GPU
bash scripts/run_cifar10_predgen_sizegen_v1.sh "0 2"

# Extend to w192-w512: zoo gen (BOTH arms' splits rewritten over all ten widths, finalize() md5s
# the pre-existing splits so a regression would be visible) then re-score the same checkpoints;
# refuses to start without the zoo-gen DONE marker.
bash scripts/run_cifar10_zoo_gen_v2.sh "4 5"
bash scripts/run_cifar10_predgen_sizegen_v2.sh "7"

# 10 conditions, bidirectional: 5 trainings -> score under muP16 -> score under SP -> w16-identity
# audit. Resumable; re-running with a different GPU list ADDS a worker.
bash scripts/run_cifar10_predgen_sizegen_v1_bidir.sh "5"

# Extend the SIX valid (symmetry: permutation) bidirectional conditions to w16-w512 by
# re-scoring the same checkpoints (no retraining); the four ScaleGMN conditions are not
# queued -- out of scope. Requires /tmp/predgen_zoo_v2/DONE.
bash scripts/run_cifar10_predgen_sizegen_v2_bidir.sh "4"

# DO NOT RUN: retrain of the 4 invalid ScaleGMN-bidirectional conditions, dropped as out of
# scope (NaNs by step ~13 on every LR, see the warning above). Kept only as a record.
# bash scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh ""

# Conv matrix-product ScaleGMN: one training, scored on both width ranges and both arms.
bash scripts/run_cifar10_predgen_mpsgmn_sizegen.sh ""

# zoo quality: the three tables this experiment cannot be read without (labels, norms, tau_HP)
python scripts/summarize_cnn_zoo_quality.py
```

The zoo generation and the forward training queue were chained on 2026-08-21 by
`scripts/orchestrate_cnn_first.sh`, which grows the zoo-generation pool onto each GPU as it frees,
waits for the `DONE` sentinel, then seeds and drains the Experiment 3 queue, and finally resumes the
MNIST / FMNIST size-gen queues that were paused to give this study the pool.

**Configs**: `configs/cifar10_predgen/{scalegmn,gmn,mpgmn}_sizegen_{sp,mup16}_v1.yml` for the w16–w128
training and forward re-score (`mpgmn_*` must be run with `--duplication-equiv True`); the `_v2.yml`
variants are identical apart from the four extra test widths;
`mpsgmn_sizegen_{sp,mup16}_{v1,v2}.yml` for the conv matrix-product ScaleGMN.
**Logs**: `/tmp/predgen_sizegen_v1/`, `/tmp/predgen_zoo_v2/`, `/tmp/predgen_sizegen_v2/`,
`/tmp/predgen_sizegen_v1_bidir/`, `/tmp/predgen_sizegen_v2_bidir/`,
`/tmp/predgen_sizegen_v1_bidir_recip/` (idle, dropped — see above),
`/tmp/mpsgmn_predgen/` — each an incremental `SUMMARY.md` plus per-run sweep / full / eval logs; queue
files are created only if missing, so re-running with a different GPU list **adds** workers rather
than duplicating work.

**Status: all forward conditions done (2026-08-25 04:37), extended to w512 by 2026-08-29 11:45.**
Gate V5 cleared 2026-08-21, the w16–w128 zoo and both arms' splits finished 2026-08-22 00:41, and the
ten forward conditions ran on five GPUs (0/1/2/4/5; GPU 3 and GPUs 6/7 left to other users), draining
in **3.2 days** — last two out were `mpgmn-mup16-deq` (2026-08-25 03:25) and `mpgmn-sp-deq` (04:37).
The w192–w512 zoo ([Data](#data) above) then let all ten checkpoints reach w512 by one
`--eval-only-ckpt` re-score each, verified for all five pairs: same selected LR, same `val_tau` at
every one of 200 epochs, same kept epoch (37 / 71 / 99 / 196 / 199 for `sgmn-base` / `gmn-base` /
`gmn-deq` / `sgmn-deq` / `mpgmn-deq`), `max |Δw| = 0.00e+00` across every parameter tensor (142 / 75 /
75 / 142 / 36 tensors), and w16 test predictions equal bit-for-bit (`max |Δy| = 0`). Full logs:
`/tmp/predgen_sizegen_v1/SUMMARY.md`, `/tmp/predgen_sizegen_v2/SUMMARY.md`.

### Plain GMN Results (`symmetry: permutation`, forward and bidirectional)

#### Forward

**LR sweep** (same grid; val $\tau$ at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.9027 / 0.9051 / 0.9062 / 0.9113 / **0.9120** / 0.9086 / 0.9075 → **1e-3**
- SP dup-equiv: 0.9088 / 0.9096 / 0.9095 / **0.9107** / 0.9093 / 0.9080 / 0.9084 → **5e-4**
- muP16 baseline: 0.9027 / 0.9051 / 0.9062 / 0.9113 / **0.9120** / 0.9086 / 0.9075 → **1e-3**
- muP16 dup-equiv: 0.9088 / 0.9096 / 0.9095 / **0.9107** / 0.9093 / 0.9080 / 0.9084 → **5e-4**

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | **0.9131** | 0.9075 | **0.9131** | 0.9075 |
| w32 | 0.8903 | 0.8897 | 0.8876 | **0.9057** |
| w48 | 0.8144 | 0.8175 | 0.8476 | **0.8923** |
| w64 | 0.7472 | 0.7687 | 0.8012 | **0.8740** |
| w96 | 0.6191 | 0.6211 | 0.7388 | **0.8330** |
| w128 | 0.5245 | 0.5090 | 0.7181 | **0.8083** |
| w192 | 0.4055 | 0.3107 | 0.6971 | **0.7786** |
| w256 | 0.3376 | 0.2154 | 0.6614 | **0.7349** |
| w384 | 0.2536 | 0.0932 | 0.6406 | **0.6950** |
| w512 | 0.2013 | 0.0175 | 0.6356 | **0.6614** |
| **OOD mean** (w32–w512) | 0.5326 | 0.4714 | 0.7364 | **0.7981** |

Its baseline is the best in-distribution condition in the study ($\tau = 0.9131$ at w16, above every
ScaleGMN column) and, under SP, the worst OOD by w128 already (below `τ_HP(16,128) = 0.640`) —
ranking CNNs worse than a predictor that ignored the weights entirely. SP dup-equiv is worse still by
w512 ($0.0175$, indistinguishable from random ranking). The concat+MLP MSG function has no way to
absorb the fan-in growth, as Experiment 2 measured on function-identical widenings (`max_drop` 0.2388
baseline, 0.0000 dup-equiv).

Fan-in rescaling + mean aggregation neutralizes the *combinatorial* part of widening (more edges) but
not the *magnitude* part (larger weights); for a permutation-symmetric MSG function the magnitude
part dominates once SP lets `‖w‖_Vn` grow $7.7\times$ by w512. Under SP, dup-equivariance buys nothing
on $\tau$ past w128 and turns actively harmful by w512; under muP16 it buys $+0.062$ OOD mean. Only
muP16 + dup-equiv, which removes both confounds, generalizes.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.980 / 0.0106 / +0.981 | +0.979 / 0.0108 / +0.979 | +0.980 / 0.0106 / +0.981 | +0.979 / 0.0108 / +0.979 |
| w32 | +0.623 / 0.0626 / +0.958 | +0.959 / 0.0157 / +0.964 | +0.656 / 0.0576 / +0.956 | +0.965 / 0.0142 / +0.972 |
| w48 | −0.165 / 0.1166 / +0.884 | +0.885 / 0.0275 / +0.913 | +0.107 / 0.0970 / +0.903 | +0.941 / 0.0187 / +0.963 |
| w64 | −0.777 / 0.1536 / +0.782 | +0.806 / 0.0396 / +0.863 | −0.328 / 0.1252 / +0.842 | +0.903 / 0.0249 / +0.943 |
| w96 | −1.975 / 0.2023 / +0.530 | +0.594 / 0.0608 / +0.676 | −1.086 / 0.1530 / +0.733 | +0.840 / 0.0319 / +0.907 |
| w128 | −2.549 / 0.2218 / +0.370 | +0.404 / 0.0773 / +0.512 | −1.229 / 0.1628 / +0.719 | +0.804 / 0.0377 / +0.877 |
| w192 | −3.465 / 0.2492 / +0.203 | +0.017 / 0.1017 / +0.219 | −1.410 / 0.1732 / +0.686 | +0.755 / 0.0436 / +0.837 |
| w256 | −3.506 / 0.2568 / +0.126 | −0.119 / 0.1134 / +0.119 | −1.607 / 0.1791 / +0.644 | +0.687 / 0.0501 / +0.780 |
| w384 | −3.902 / 0.2767 / +0.061 | −0.407 / 0.1326 / +0.025 | −1.968 / 0.1854 / +0.626 | +0.606 / 0.0553 / +0.725 |
| w512 | −3.983 / 0.2755 / +0.021 | −0.552 / 0.1387 / +0.003 | −1.814 / 0.1837 / +0.630 | +0.565 / 0.0607 / +0.678 |

Both baselines go negative from w48/w64 on, the SP one reaching $R^2 = -3.983$ at w512 — the worst
number in the study — with L1 $= 0.276$ against a label range of $\approx 0.1$–$0.5$; `R²_recal` down
to $+0.021$ means the ordering itself is gone, not merely rescaled. SP dup-equiv follows it into
negative territory by w256 ($R^2=-0.119$, `R²_recal`$=+0.119$) — the one condition in the study whose
dup-equiv variant is worse than its baseline at the widest test width.

| Condition | Checkpoint | Eval run (w16–w128) |
|---|---|---|
| SP baseline | `predgen-sizegen-v1-gmn-sp-base_xzh7c3i2` | [`yis87z88`](https://wandb.ai/yuxinma/cifar10_predgen/runs/yis87z88) |
| SP dup-equiv | `predgen-sizegen-v1-gmn-sp-deq_v87ixou2` | [`7nytce7h`](https://wandb.ai/yuxinma/cifar10_predgen/runs/7nytce7h) |
| muP16 baseline | `predgen-sizegen-v1-gmn-mup16-base_u2h067oj` | [`5wkc5a33`](https://wandb.ai/yuxinma/cifar10_predgen/runs/5wkc5a33) |
| muP16 dup-equiv | `predgen-sizegen-v1-gmn-mup16-deq_ud2bgeq7` | [`quxqdtc3`](https://wandb.ai/yuxinma/cifar10_predgen/runs/quxqdtc3) |

**Training runs** (200 epochs):

| Variant | Run ID | LR | Epochs | Best Val τ |
|---|---|---|---|---|
| SP baseline | [`xzh7c3i2`](https://wandb.ai/yuxinma/cifar10_predgen/runs/xzh7c3i2) | 1e-3 | 200 | 0.9136 (ep 71) |
| SP dup-equiv | [`v87ixou2`](https://wandb.ai/yuxinma/cifar10_predgen/runs/v87ixou2) | 5e-4 | 200 | 0.9113 (ep 99) |
| muP16 baseline | [`u2h067oj`](https://wandb.ai/yuxinma/cifar10_predgen/runs/u2h067oj) | 1e-3 | 200 | 0.9136 (ep 71) |
| muP16 dup-equiv | [`ud2bgeq7`](https://wandb.ai/yuxinma/cifar10_predgen/runs/ud2bgeq7) | 5e-4 | 200 | 0.9113 (ep 99) |

The w192–w512 re-score ran the same way as ScaleGMN's (`v1 agreement: exact`), each condition logged
as its own wandb eval run — SP baseline
[`8e8oty8v`](https://wandb.ai/yuxinma/cifar10_predgen/runs/8e8oty8v), SP dup-equiv
[`x51lrio8`](https://wandb.ai/yuxinma/cifar10_predgen/runs/x51lrio8), muP16 baseline
[`eqsc86f2`](https://wandb.ai/yuxinma/cifar10_predgen/runs/eqsc86f2), muP16 dup-equiv
[`hddc44zn`](https://wandb.ai/yuxinma/cifar10_predgen/runs/hddc44zn).
Predictions: `/tmp/predgen_sizegen_v1/preds_gmn-{sp,mup16}-{base,deq}.npz`.

#### Bidirectional

Selected LRs (7-value $\times 2$ grid, val $\tau$ at w16): baseline **2e-3**, dup-equiv **1e-3**, both
200 epochs (no early stop). Each variant trains **once**, under the muP16 config, and is scored under
both arms (base width = train width, so w16 is byte-identical across arms); `w16 identity` audits OK
for all four rows below.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold), extended from w128 to w512
2026-08-31 by an eval-only re-score of the same four checkpoints — `v1 agreement: OK` (w16–w128 taus
reproduce v1's exactly) and `w16 identity: OK` (the two arms still match at w16) for all four rows,
`/tmp/predgen_sizegen_v2_bidir/SUMMARY.md`:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | **0.9122** | 0.9106 | **0.9122** | 0.9106 |
| w32 | 0.8658 | 0.8996 | 0.8545 | **0.9135** |
| w48 | 0.7979 | 0.8367 | 0.8094 | **0.8980** |
| w64 | 0.7479 | 0.7835 | 0.7700 | **0.8840** |
| w96 | 0.6531 | 0.6480 | 0.7336 | **0.8374** |
| w128 | 0.5633 | 0.5294 | 0.7254 | **0.8170** |
| w192 | 0.4211 | 0.3335 | 0.7176 | **0.7804** |
| w256 | 0.3388 | 0.2077 | 0.6864 | **0.7307** |
| w384 | 0.2298 | 0.0639 | 0.6652 | **0.6784** |
| w512 | 0.1865 | −0.0222 | **0.6692** | 0.6428 |
| **OOD mean** (w32–w512) | 0.5338 | 0.4756 | 0.7368 | **0.7980** |

Both SP conditions fall below the hyperparameter-only ceiling `τ_HP(16,128) = 0.640` by w128 (dup-equiv
$0.5294$, baseline $0.5633$) — the same failure Forward's Plain GMN SP columns show, so it is not a
direction effect. SP dup-equiv's $\tau$ turns negative by w512 ($-0.0222$), indistinguishable from
random ranking, tracking Forward's SP dup-equiv collapse ($0.0175$ at w512). muP16 dup-equiv is the
row-best at every OOD width through w384, but muP16 baseline overtakes it at w512 ($0.6692$ vs
$0.6428$) — the only width in this table where dup-equivariance is not the better choice under muP16.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.982 / 0.0102 / +0.982 | +0.978 / 0.0114 / +0.979 | +0.982 / 0.0102 / +0.982 | +0.978 / 0.0114 / +0.979 |
| w32 | +0.564 / 0.0663 / +0.944 | +0.956 / 0.0172 / +0.969 | +0.585 / 0.0616 / +0.938 | +0.953 / 0.0178 / +0.975 |
| w48 | −0.210 / 0.1182 / +0.872 | +0.882 / 0.0286 / +0.920 | +0.002 / 0.1004 / +0.861 | +0.917 / 0.0241 / +0.966 |
| w64 | −0.728 / 0.1506 / +0.806 | +0.804 / 0.0399 / +0.856 | −0.460 / 0.1289 / +0.791 | +0.879 / 0.0305 / +0.952 |
| w96 | −1.812 / 0.1964 / +0.678 | +0.610 / 0.0583 / +0.670 | −1.155 / 0.1541 / +0.729 | +0.804 / 0.0376 / +0.913 |
| w128 | −2.415 / 0.2172 / +0.548 | +0.436 / 0.0718 / +0.495 | −1.238 / 0.1616 / +0.717 | +0.773 / 0.0421 / +0.889 |
| w192 | −3.369 / 0.2459 / +0.278 | +0.119 / 0.0912 / +0.202 | −1.347 / 0.1699 / +0.690 | +0.721 / 0.0479 / +0.846 |
| w256 | −3.439 / 0.2540 / +0.141 | −0.014 / 0.1020 / +0.090 | −1.503 / 0.1745 / +0.633 | +0.648 / 0.0536 / +0.789 |
| w384 | −3.822 / 0.2732 / +0.038 | −0.259 / 0.1221 / +0.008 | −1.792 / 0.1790 / +0.607 | +0.558 / 0.0580 / +0.722 |
| w512 | −3.900 / 0.2722 / +0.018 | −0.410 / 0.1279 / +0.001 | −1.641 / 0.1773 / +0.621 | +0.525 / 0.0628 / +0.678 |

The SP baseline's $R^2$ collapses to $-2.415$ at w128 and further to $-3.900$ at w512 (`R²_recal`
$+0.018$) — within $2\%$ of Forward's $-3.983$, the worst number in the study — the same
calibration-failure-not-ranking-failure signature Forward's baselines show. SP dup-equiv follows the
same path as Forward: negative $R^2$ from w256 on ($-0.014$), reaching $-0.410$ at w512.

| Condition | Checkpoint | Eval run (w16–w128) | Eval run (w192–w512) |
|---|---|---|---|
| SP baseline | `predgen-sizegen-v1-bidir-gmn-base-bidir_uoxx4ziu` | [`ioipkd4g`](https://wandb.ai/yuxinma/cifar10_predgen/runs/ioipkd4g) | [`i036wpxv`](https://wandb.ai/yuxinma/cifar10_predgen/runs/i036wpxv) |
| SP dup-equiv | `predgen-sizegen-v1-bidir-gmn-deq-bidir_dh3gh55b` | [`87og968y`](https://wandb.ai/yuxinma/cifar10_predgen/runs/87og968y) | [`sdalxomv`](https://wandb.ai/yuxinma/cifar10_predgen/runs/sdalxomv) |
| muP16 baseline | `predgen-sizegen-v1-bidir-gmn-base-bidir_uoxx4ziu` | [`t0k5h7iw`](https://wandb.ai/yuxinma/cifar10_predgen/runs/t0k5h7iw) | [`y7odppaz`](https://wandb.ai/yuxinma/cifar10_predgen/runs/y7odppaz) |
| muP16 dup-equiv | `predgen-sizegen-v1-bidir-gmn-deq-bidir_dh3gh55b` | [`x52efl13`](https://wandb.ai/yuxinma/cifar10_predgen/runs/x52efl13) | [`zb5yuwby`](https://wandb.ai/yuxinma/cifar10_predgen/runs/zb5yuwby) |

**Training runs** (200 epochs, one per variant — trained under muP16, scored under both arms):

| Variant | Run ID | LR | Epochs | Best Val τ (w16) |
|---|---|---|---|---|
| baseline | [`uoxx4ziu`](https://wandb.ai/yuxinma/cifar10_predgen/runs/uoxx4ziu) | 2e-3 | 200 | 0.9117 |
| dup-equiv | [`dh3gh55b`](https://wandb.ai/yuxinma/cifar10_predgen/runs/dh3gh55b) | 1e-3 | 200 | 0.9100 |

Full sweep curves: `/tmp/predgen_sizegen_v1_bidir/SUMMARY.md`.

### ScaleGMN Results (`symmetry: scale`, forward and bidirectional)

#### Forward

**LR sweep** (7 LRs from 6.25e-5 to 4e-3, $\times 2$ grid, 8600 steps each; val $\tau$ at
6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.9043 / **0.9064** / 0.9055 / 0.9026 / 0.9035 / 0.9024 / 0.9007 → **1.25e-4**
- SP dup-equiv: 0.8934 / 0.8993 / 0.9008 / **0.9025** / 0.9012 / 0.9000 / 0.8982 → **5e-4**
- muP16 baseline: 0.9043 / **0.9064** / 0.9055 / 0.9026 / 0.9035 / 0.9024 / 0.9007 → **1.25e-4**
- muP16 dup-equiv: 0.8934 / 0.8993 / 0.9008 / **0.9025** / 0.9012 / 0.9000 / 0.8982 → **5e-4**

Each SP row equals its muP16 row because it *is* the same training run (base width = train width); the
grid is flat to $\pm 0.01$ $\tau$ across a $64\times$ LR range.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | **0.9041** | 0.9039 | **0.9041** | 0.9039 |
| w32 | 0.9090 | 0.9066 | 0.9155 | **0.9161** |
| w48 | 0.8820 | 0.8873 | 0.8978 | **0.9063** |
| w64 | 0.8590 | 0.8763 | 0.8879 | **0.8992** |
| w96 | 0.7697 | 0.8357 | 0.8589 | **0.8691** |
| w128 | 0.7190 | 0.8350 | 0.8455 | **0.8616** |
| w192 | 0.5975 | 0.7980 | 0.8278 | **0.8514** |
| w256 | 0.5001 | 0.7764 | 0.8079 | **0.8260** |
| w384 | 0.3488 | 0.7456 | 0.7600 | **0.7879** |
| w512 | 0.2559 | 0.6969 | **0.7609** | 0.7560 |
| **OOD mean** (w32–w512) | 0.6490 | 0.8176 | 0.8402 | **0.8526** |

Every cell clears the hyperparameter-only ceiling `τ_HP(16,128) = 0.640` SP / `0.599` muP16, so these
columns are reading the weights, not the hyperparameters — though SP baseline is the closest any
ScaleGMN column comes to it, and its $\tau$ at w512 ($0.2559$) is still above the w512 ceiling
(`τ_HP(16,512) = 0.457`).

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.973 / 0.0123 / +0.977 | +0.976 / 0.0109 / +0.977 | +0.973 / 0.0123 / +0.977 | +0.976 / 0.0109 / +0.977 |
| w32 | +0.850 / 0.0386 / +0.973 | +0.963 / 0.0159 / +0.975 | +0.857 / 0.0370 / +0.979 | +0.970 / 0.0133 / +0.978 |
| w48 | +0.648 / 0.0642 / +0.952 | +0.926 / 0.0235 / +0.964 | +0.678 / 0.0592 / +0.964 | +0.950 / 0.0179 / +0.972 |
| w64 | +0.519 / 0.0820 / +0.918 | +0.894 / 0.0308 / +0.957 | +0.546 / 0.0754 / +0.956 | +0.927 / 0.0228 / +0.965 |
| w96 | +0.145 / 0.1129 / +0.749 | +0.820 / 0.0421 / +0.928 | +0.180 / 0.0997 / +0.920 | +0.884 / 0.0279 / +0.944 |
| w128 | −0.118 / 0.1301 / +0.660 | +0.776 / 0.0491 / +0.922 | +0.008 / 0.1135 / +0.895 | +0.865 / 0.0311 / +0.933 |
| w192 | −0.701 / 0.1613 / +0.396 | +0.671 / 0.0609 / +0.877 | −0.282 / 0.1325 / +0.848 | +0.839 / 0.0347 / +0.920 |
| w256 | −0.933 / 0.1756 / +0.260 | +0.613 / 0.0693 / +0.836 | −0.622 / 0.1482 / +0.812 | +0.802 / 0.0391 / +0.898 |
| w384 | −1.439 / 0.2018 / +0.082 | +0.491 / 0.0835 / +0.794 | −1.314 / 0.1707 / +0.734 | +0.743 / 0.0423 / +0.854 |
| w512 | −1.836 / 0.2165 / +0.022 | +0.404 / 0.0893 / +0.707 | −1.450 / 0.1782 / +0.682 | +0.714 / 0.0471 / +0.825 |

Both baselines' $R^2$ collapse well below 0 by w512 (SP $-1.836$, muP16 $-1.450$) while $\tau$ and
`R²_recal` stay comparatively high through most of the ladder — the Experiment 2 signature of a scale
drift recoverable by per-width affine recalibration. Dup-equiv keeps $R^2$ itself, without
recalibration ($+0.404$ SP / $+0.714$ muP16 at w512).

| Condition | Checkpoint | Eval run (w16–w128) |
|---|---|---|
| SP baseline | `predgen-sizegen-v1-sgmn-sp-base_o8xjd542` | [`l3kc8whp`](https://wandb.ai/yuxinma/cifar10_predgen/runs/l3kc8whp) |
| SP dup-equiv | `predgen-sizegen-v1-sgmn-sp-deq_jaehou9t` | [`hrvc7l33`](https://wandb.ai/yuxinma/cifar10_predgen/runs/hrvc7l33) |
| muP16 baseline | `predgen-sizegen-v1-sgmn-mup16-base_pri6dpcy` | [`tsmax75v`](https://wandb.ai/yuxinma/cifar10_predgen/runs/tsmax75v) |
| muP16 dup-equiv | `predgen-sizegen-v1-sgmn-mup16-deq_svjxn4qb` | [`7a2kehdw`](https://wandb.ai/yuxinma/cifar10_predgen/runs/7a2kehdw) |

**Training runs** (200 epochs):

| Variant | Run ID | LR | Epochs | Best Val τ |
|---|---|---|---|---|
| SP baseline | [`o8xjd542`](https://wandb.ai/yuxinma/cifar10_predgen/runs/o8xjd542) | 1.25e-4 | 200 | 0.9064 (ep 37) |
| SP dup-equiv | [`jaehou9t`](https://wandb.ai/yuxinma/cifar10_predgen/runs/jaehou9t) | 5e-4 | 200 | 0.9040 (ep 196) |
| muP16 baseline | [`pri6dpcy`](https://wandb.ai/yuxinma/cifar10_predgen/runs/pri6dpcy) | 1.25e-4 | 200 | 0.9064 (ep 37) |
| muP16 dup-equiv | [`svjxn4qb`](https://wandb.ai/yuxinma/cifar10_predgen/runs/svjxn4qb) | 5e-4 | 200 | 0.9040 (ep 196) |

The w192–w512 re-score of these same checkpoints ran 2026-08-29 on one worker (GPU 7); each
condition's re-score is logged as its own wandb eval run too — SP baseline
[`qs01lv2v`](https://wandb.ai/yuxinma/cifar10_predgen/runs/qs01lv2v), SP dup-equiv
[`mmucpare`](https://wandb.ai/yuxinma/cifar10_predgen/runs/mmucpare), muP16 baseline
[`gk0djs49`](https://wandb.ai/yuxinma/cifar10_predgen/runs/gk0djs49), muP16 dup-equiv
[`5z8dcv8i`](https://wandb.ai/yuxinma/cifar10_predgen/runs/5z8dcv8i) — transcribed here from the
consolidated `/tmp/predgen_sizegen_v2/SUMMARY.md` (`v1 agreement: exact` for every condition — its
w16–w128 numbers reproduce the eval runs above bit-for-bit). Predictions:
`/tmp/predgen_sizegen_v1/preds_sgmn-{sp,mup16}-{base,deq}.npz`.

#### Bidirectional

⚠ **All four conditions here are invalid — see the warning in
[Model and training setup](#model-and-training-setup) above.** They are not a bidirectional ScaleGMN
with a degraded property; they are a different, non-equivariant model that happens to share the config
file, because `bw_edge_attr` was never replaced with the reciprocal edge feature a bidirectional
ScaleGMN is defined with. A retrain is queued but blocked on a NaN. The numbers below are kept for the
record, not as a property measurement, and — unlike the plain GMN and conv matrix-product GMN
bidirectional conditions — were **not extended to w512**: re-scoring a non-equivariant model further
out of distribution adds nothing, so these four remain w16–w128 only, out of scope.

Selected LRs (7-value $\times 2$ grid, val $\tau$ at w16): baseline **2e-3** (200 epochs), dup-equiv
**4e-3** (crashed on a NaN at epoch ~115, best checkpoint saved at epoch 104). Each variant trains **once**, under the muP16 config, and is
scored under both arms; `w16 identity` audits OK for all four rows below.

**Per-width test Kendall $\tau$, w16 → w128** (row-best in bold; no run exists past w128):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 0.8973 | **0.8984** | 0.8973 | **0.8984** |
| w32 | 0.9013 | 0.8931 | 0.9091 | **0.9131** |
| w48 | 0.8799 | 0.8703 | 0.8945 | **0.9080** |
| w64 | 0.8754 | 0.8585 | 0.8934 | **0.9038** |
| w96 | 0.8477 | 0.8218 | 0.8745 | **0.8789** |
| w128 | 0.8307 | 0.8006 | 0.8594 | **0.8748** |
| **OOD mean** (w32–w128) | 0.8670 | 0.8489 | 0.8862 | **0.8957** |

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.973 / 0.0120 / +0.974 | +0.973 / 0.0122 / +0.975 | +0.973 / 0.0120 / +0.974 | +0.973 / 0.0122 / +0.975 |
| w32 | +0.919 / 0.0270 / +0.970 | +0.945 / 0.0202 / +0.968 | +0.921 / 0.0262 / +0.975 | +0.963 / 0.0154 / +0.976 |
| w48 | +0.800 / 0.0466 / +0.955 | +0.894 / 0.0301 / +0.955 | +0.804 / 0.0446 / +0.965 | +0.944 / 0.0198 / +0.972 |
| w64 | +0.709 / 0.0621 / +0.945 | +0.848 / 0.0397 / +0.948 | +0.712 / 0.0587 / +0.960 | +0.924 / 0.0240 / +0.966 |
| w96 | +0.486 / 0.0858 / +0.912 | +0.746 / 0.0541 / +0.927 | +0.461 / 0.0798 / +0.942 | +0.884 / 0.0289 / +0.947 |
| w128 | +0.280 / 0.1033 / +0.879 | +0.646 / 0.0654 / +0.903 | +0.313 / 0.0934 / +0.918 | +0.867 / 0.0318 / +0.938 |

The baseline's $R^2$ collapses to $+0.280$ (SP) / $+0.313$ (muP16) at w128 while $\tau$ and `R²_recal`
stay high — the same calibration-failure-not-ranking-failure signature Forward's ScaleGMN baseline
shows.

| Condition | Checkpoint | Eval run (w16–w128) |
|---|---|---|
| SP baseline | `predgen-sizegen-v1-bidir-sgmn-base-bidir_drzplfn0` | [`mxsowscd`](https://wandb.ai/yuxinma/cifar10_predgen/runs/mxsowscd) |
| SP dup-equiv | `predgen-sizegen-v1-bidir-sgmn-deq-bidir_sddltcue` | [`cjghqcyy`](https://wandb.ai/yuxinma/cifar10_predgen/runs/cjghqcyy) |
| muP16 baseline | `predgen-sizegen-v1-bidir-sgmn-base-bidir_drzplfn0` | [`4uqfw2o4`](https://wandb.ai/yuxinma/cifar10_predgen/runs/4uqfw2o4) |
| muP16 dup-equiv | `predgen-sizegen-v1-bidir-sgmn-deq-bidir_sddltcue` | [`9h3mu5yo`](https://wandb.ai/yuxinma/cifar10_predgen/runs/9h3mu5yo) |

**Training runs** (one per variant — trained under muP16, scored under both arms):

| Variant | Run ID | LR | Epochs | Best Val τ (w16) |
|---|---|---|---|---|
| baseline | [`drzplfn0`](https://wandb.ai/yuxinma/cifar10_predgen/runs/drzplfn0) | 2e-3 | 200 | 0.9025 |
| dup-equiv | [`sddltcue`](https://wandb.ai/yuxinma/cifar10_predgen/runs/sddltcue) | 4e-3 | 114 (crashed on NaN, ep 115) | 0.8952 (ep 104) |

Full sweep curves: `/tmp/predgen_sizegen_v1_bidir/SUMMARY.md`. A retrain was attempted and dropped
as out of scope — see the ⚠ warning above and
[`src/models/bidir_reciprocal.py`](../src/models/bidir_reciprocal.py).
Logs of the attempt: `/tmp/predgen_sizegen_v1_bidir_recip/`.

### Conv Matrix-Product GMN Results (`message_fn_type: matrix_product_conv`, forward and bidirectional)

Same protocol and LR grid as above. The conv matrix-product GMN is **dup-equiv only** — its MSG
function only becomes a sum of matrix products on top of fan-in rescaling + mean aggregation, so
there is no baseline variant. On a CNN graph an edge carries a whole kernel in R⁹, so the scalar
matrix-product message becomes bilinear, with one learned map per kernel offset:
`msg_{u→v} = Σ_s a_{u,v}[s] · (h_u A_s)`, giving
`Z^(l) = (1/n_{l−1}) Σ_s W̄_s^(l) H^(l−1) A_s` under `aggregator: mean` — the input weights enter
only through matrix multiplication. Layer: `src/models/conv_matrix_product_layer.py`; installed by
name-swapping via `conv_matrix_product_context(conf)`, not by editing the subtree. Properties
verified by `scripts/check_matrix_product_gmn_cnn.py` (all 3 tests PASS).

#### Forward

**LR sweep** (same grid; val $\tau$ at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP: 0.8863 / 0.8951 / 0.8975 / 0.8995 / **0.9017** / 0.9015 / 0.9012 → **1e-3**
- muP16: 0.8863 / 0.8951 / 0.8975 / 0.8995 / **0.9017** / 0.9015 / 0.9012 → **1e-3**

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | 0.9057 | 0.9057 |
| w32 | 0.9072 | **0.9207** |
| w48 | 0.8793 | **0.9170** |
| w64 | 0.8617 | **0.9146** |
| w96 | 0.8356 | **0.9044** |
| w128 | 0.7776 | **0.9006** |
| w192 | 0.7124 | **0.8988** |
| w256 | 0.6716 | **0.8889** |
| w384 | 0.6147 | **0.8657** |
| w512 | 0.5386 | **0.8628** |
| **OOD mean** (w32–w512) | 0.7554 | **0.8971** |

muP16's $\tau$ goes $0.9057$ at w16 $\to 0.8628$ at w512, a $0.043$ drop over a $32\times$ widening,
against SP's $0.367$ over the same range. Constraining the MSG function to a sum of matrix products
in the raw weights makes the metanetwork's output move continuously with the normalized spectral
norm, which muP16 holds flat.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | +0.976 / 0.0111 / +0.977 | +0.976 / 0.0111 / +0.977 |
| w32 | +0.971 / 0.0134 / +0.974 | +0.978 / 0.0118 / +0.981 |
| w48 | +0.945 / 0.0183 / +0.954 | +0.969 / 0.0140 / +0.978 |
| w64 | +0.913 / 0.0245 / +0.933 | +0.960 / 0.0171 / +0.976 |
| w96 | +0.873 / 0.0306 / +0.906 | +0.943 / 0.0193 / +0.968 |
| w128 | +0.766 / 0.0416 / +0.832 | +0.934 / 0.0210 / +0.962 |
| w192 | +0.588 / 0.0579 / +0.730 | +0.924 / 0.0228 / +0.961 |
| w256 | +0.479 / 0.0689 / +0.685 | +0.908 / 0.0249 / +0.951 |
| w384 | +0.229 / 0.0909 / +0.599 | +0.881 / 0.0269 / +0.933 |
| w512 | −0.118 / 0.1089 / +0.489 | +0.879 / 0.0285 / +0.935 |
| **OOD mean** (w32–w128) | +0.894 | **+0.957** |

muP16 holds $R^2 \ge +0.879$ at every width and needs essentially no recalibration (`R²_recal −
R²` $\le 0.056$ everywhere, even at w512). Under SP, same metanetwork, $R^2$ crosses zero by w512
($-0.118$). Its SP-vs-muP16 OOD $\tau$ gap ($0.142$ over w32–w512) sits above ScaleGMN dup-equiv's
($0.035$) and below plain GMN dup-equiv's ($0.327$): Lipschitz continuity in the normalized spectral
norm bounds how fast the output can move, it does not hold the output constant when the input norms
grow $7.7\times$. The matrix-product constraint and muP are complements, not substitutes.

| Condition | Checkpoint | Eval run (w16–w128) |
|---|---|---|
| SP | `predgen-sizegen-v1-mpgmn-sp-deq_bvjiy19n` | [`7u2iw9w7`](https://wandb.ai/yuxinma/cifar10_predgen/runs/7u2iw9w7) |
| muP16 | `predgen-sizegen-v1-mpgmn-mup16-deq_1ig5pp0x` | [`sq2v72jf`](https://wandb.ai/yuxinma/cifar10_predgen/runs/sq2v72jf) |

**Training runs** (200 epochs):

| Variant | Run ID | LR | Epochs | Best Val τ |
|---|---|---|---|---|
| SP | [`bvjiy19n`](https://wandb.ai/yuxinma/cifar10_predgen/runs/bvjiy19n) | 1e-3 | 200 | 0.9045 (ep 199) |
| muP16 | [`1ig5pp0x`](https://wandb.ai/yuxinma/cifar10_predgen/runs/1ig5pp0x) | 1e-3 | 200 | 0.9045 (ep 199) |

Predictions: `/tmp/predgen_sizegen_v1/preds_mpgmn-{sp,mup16}-deq.npz`. This is the only forward
condition whose kept epoch is the last one (199), i.e. val $\tau$ was still improving at 200 epochs —
the constrained MSG function trains more slowly, so its in-distribution number is the one most likely
to be understated by the fixed 200-epoch budget. The w192–w512 re-score ran the same way as
ScaleGMN's (`v1 agreement: exact`), each condition logged as its own wandb eval run — SP
[`cdxuh1cx`](https://wandb.ai/yuxinma/cifar10_predgen/runs/cdxuh1cx), muP16
[`9nrpbpsn`](https://wandb.ai/yuxinma/cifar10_predgen/runs/9nrpbpsn).

#### Bidirectional

Selected LR (7-value $\times 2$ grid, val $\tau$ at w16): **1e-3**, 200 epochs. Trained **once**, under
the muP16 config, and scored under both arms; `w16 identity` audits OK for both rows below.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold), extended from w128 to w512
2026-08-31 by an eval-only re-score of the same checkpoint — `v1 agreement: OK` and `w16 identity: OK`
for both rows, `/tmp/predgen_sizegen_v2_bidir/SUMMARY.md`:

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | 0.9017 | 0.9017 |
| w32 | 0.9093 | **0.9158** |
| w48 | 0.8913 | **0.9128** |
| w64 | 0.8721 | **0.9104** |
| w96 | 0.8423 | **0.8958** |
| w128 | 0.7953 | **0.8952** |
| w192 | 0.7282 | **0.8860** |
| w256 | 0.6790 | **0.8773** |
| w384 | 0.6025 | **0.8597** |
| w512 | 0.5637 | **0.8592** |
| **OOD mean** (w32–w512) | 0.7648 | **0.8902** |

muP16 stays $\ge 0.859$ over the whole $32\times$ ladder ($\Delta\tau = 0.9017 - 0.8592 = 0.0425$ from
w16 to w512), while SP keeps decaying past w128, reaching $0.5637$ at w512 — a $32\times$-ladder
version of the same SP-vs-muP16 gap Forward's conv matrix-product GMN shows ($0.5386$ SP vs $0.8628$
muP16 at w512).

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | +0.976 / 0.0116 / +0.976 | +0.976 / 0.0116 / +0.976 |
| w32 | +0.970 / 0.0140 / +0.976 | +0.972 / 0.0131 / +0.978 |
| w48 | +0.949 / 0.0191 / +0.966 | +0.962 / 0.0162 / +0.978 |
| w64 | +0.919 / 0.0255 / +0.951 | +0.949 / 0.0197 / +0.975 |
| w96 | +0.875 / 0.0326 / +0.930 | +0.925 / 0.0228 / +0.965 |
| w128 | +0.792 / 0.0419 / +0.881 | +0.913 / 0.0246 / +0.958 |
| w192 | +0.658 / 0.0551 / +0.813 | +0.893 / 0.0273 / +0.948 |
| w256 | +0.583 / 0.0637 / +0.774 | +0.864 / 0.0296 / +0.932 |
| w384 | +0.381 / 0.0833 / +0.682 | +0.834 / 0.0318 / +0.916 |
| w512 | +0.250 / 0.0916 / +0.615 | +0.835 / 0.0332 / +0.922 |
| **OOD mean** (w32–w512) | +0.709 | **+0.905** |

muP16 holds $R^2 \ge +0.834$ at every width, down from the $\ge +0.913$ measured through w128 but
still the best bidirectional condition in the study on OOD mean $\tau$ and the only one whose $R^2$
never goes negative; `R²_recal − R²` grows from $\le 0.045$ through w128 to $0.087$ at w512, still the
smallest such gap of any bidirectional condition. SP's $R^2$ crosses the same territory Forward's SP
baseline does, down to $+0.250$ at w512 ($R^2_{recal}$ $+0.615$) — a calibration failure, not a ranked
one, since $\tau$ stays at $0.5637$.

| Condition | Checkpoint | Eval run (w16–w128) | Eval run (w192–w512) |
|---|---|---|---|
| SP | `predgen-sizegen-v1-bidir-mpgmn-deq-bidir_o8udhbnt` | [`aydgllfh`](https://wandb.ai/yuxinma/cifar10_predgen/runs/aydgllfh) | [`ilvm08li`](https://wandb.ai/yuxinma/cifar10_predgen/runs/ilvm08li) |
| muP16 | `predgen-sizegen-v1-bidir-mpgmn-deq-bidir_o8udhbnt` | [`90oa5cqa`](https://wandb.ai/yuxinma/cifar10_predgen/runs/90oa5cqa) | [`uzwg7dyv`](https://wandb.ai/yuxinma/cifar10_predgen/runs/uzwg7dyv) |

**Training run**:

| Variant | Run ID | LR | Epochs | Best Val τ (w16) |
|---|---|---|---|---|
| dup-equiv | [`o8udhbnt`](https://wandb.ai/yuxinma/cifar10_predgen/runs/o8udhbnt) | 1e-3 | 200 | 0.9058 |

Full sweep curves: `/tmp/predgen_sizegen_v1_bidir/SUMMARY.md`.

### Conv Matrix-Product ScaleGMN Results (`message_fn_type: matrix_product_conv_scale`, forward)

The same bilinear-per-offset MSG constraint on top of **ScaleGMN's** scale-equivariant node states
and node update instead of the plain GMN's concat-MLP (`paper/main.tex` § "Forward matrix-product
ScaleGMN"). Dup-equiv only and forward only; identical LR grid, epochs, splits, readout
(`LastLayerReadout`) and selection metric (val $\tau$ at w16) to the conv matrix-product GMN above,
so these are drop-in columns in the same per-width tables. Trained once, under the muP16 config (base
width = train width), and scored under both arms and both width ranges — the same one-training trick
described in [Model and training setup](#model-and-training-setup) — with a `w16 identity` line
audited per pair.

Properties verified before queueing — see
[VERIFICATION_gmn_properties.md § Conv matrix-product models](VERIFICATION_gmn_properties.md#verification-of-the-matrix-product-models-1):
aggregate identity 1.19e-7, forward equivalence on all five widening families (~4.0e-7) at node and
graph level, the duplication ablation's PASS/FAIL/FAIL pattern under both width-agnostic readouts,
and conv spectral continuity 4.42e-3 → 5.47e-4 over n = 16 → 128 (vs plain ScaleGMN flat at ratio
1.00).

**Date**: queued 2026-08-28 00:46, started 2026-08-28 17:48
**Configs**: `configs/cifar10_predgen/mpsgmn_sizegen_{sp,mup16}_{v1,v2}.yml`, with
`--duplication-equiv True --direction forward`
**Launch**: `bash scripts/run_cifar10_predgen_mpsgmn_sizegen.sh ""` seeds the queue; workers are
armed per GPU via `add_worker_when_free.sh`
**Logs**: `/tmp/mpsgmn_predgen/` (incremental `SUMMARY.md`, per-run sweep / full / eval logs,
per-width prediction dumps `preds_mpsgmn-deq_{v1,v2}_{sp,mup16}.npz`)

**Status: DONE 2026-08-29 07:29** — 13.7 h wall on one GPU (GPU 0). All four conditions recorded: v1
(w16–w128) and v2 (w16–w512) each scored under {SP, muP16}.

**LR sweep** (val $\tau$ at w16, 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
0.8881 / 0.8931 / 0.9024 / 0.9043 / **0.9067** / 0.9022 / 0.9008 → **1e-3**. Then 200 epochs at 1e-3,
best val $\tau$ **0.9078**, all 200 epochs run. Training run
[`m8z4d8t9`](https://wandb.ai/yuxinma/cifar10_predgen/runs/m8z4d8t9), checkpoint
`predgen-sizegen-mpsgmn_m8z4d8t9`. The sweep peak is level with its two references on the same grid
(conv matrix-product *plain* GMN 0.9017, dup-equiv ScaleGMN 0.9025): nothing distinguishes these
three models in distribution — the whole difference is OOD.

| Width | muP16 τ | muP16 R² | muP16 L1 | muP16 R²_recal | SP τ | SP R² | SP L1 | SP R²_recal |
|---|---|---|---|---|---|---|---|---|
| w16 (IN) | 0.9062 | +0.978 | 0.0111 | +0.978 | 0.9062 | +0.978 | 0.0111 | +0.978 |
| w32 | 0.9230 | +0.979 | 0.0114 | +0.981 | 0.9142 | +0.974 | 0.0128 | +0.977 |
| w48 | **0.9277** | +0.974 | 0.0130 | +0.982 | 0.9041 | +0.962 | 0.0158 | +0.970 |
| w64 | 0.9249 | +0.962 | 0.0162 | +0.977 | 0.8969 | +0.943 | 0.0205 | +0.958 |
| w96 | 0.9248 | +0.953 | 0.0172 | +0.974 | 0.8878 | +0.909 | 0.0247 | +0.931 |
| w128 | 0.9256 | +0.945 | 0.0187 | +0.970 | 0.8655 | +0.871 | 0.0296 | +0.903 |
| w192 | 0.9247 | +0.936 | 0.0202 | +0.968 | 0.8323 | +0.777 | 0.0398 | +0.838 |
| w256 | 0.9204 | +0.925 | 0.0219 | +0.963 | 0.8223 | +0.740 | 0.0446 | +0.819 |
| w384 | 0.9137 | +0.914 | 0.0221 | +0.955 | 0.7878 | +0.586 | 0.0604 | +0.750 |
| w512 | 0.9123 | +0.913 | 0.0232 | +0.958 | 0.7471 | +0.468 | 0.0680 | +0.699 |
| **OOD mean w32–w128 (v1)** | **0.9252** | | | | **0.8937** | | | |
| **OOD mean w32–w512 (v2)** | **0.9219** | | | | **0.8509** | | | |

| Condition | Eval run | `w16 identity` |
|---|---|---|
| muP16, v1 (w16–w128) | [`b5xdl6cv`](https://wandb.ai/yuxinma/cifar10_predgen/runs/b5xdl6cv) | OK |
| SP, v1 (w16–w128) | [`podmgefg`](https://wandb.ai/yuxinma/cifar10_predgen/runs/podmgefg) | OK |
| muP16, v2 (w16–w512) | [`8mpi98un`](https://wandb.ai/yuxinma/cifar10_predgen/runs/8mpi98un) | OK |
| SP, v2 (w16–w512) | [`lewewvbu`](https://wandb.ai/yuxinma/cifar10_predgen/runs/lewewvbu) | OK |

All four `w16 identity` audits pass, and each v2 row at w16–w128 reproduces its v1 row to the last
digit, in both arms — the same `v1 agreement` check used for the ten forward re-scores above.

On muP16 this is the best condition in the study: OOD $\tau$ $0.9252$ (w32–w128) against the previous
best $0.9115$ (conv matrix-product GMN, muP16), with `Δτ = w16 − w128 = −0.019` — it ranks CNNs
better $8\times$ out than at the width it trained on, $\tau$ peaking at w48. Over $32\times$ it still
has not turned over — w512 $\tau$ $0.9123$, `Δτ = −0.006` — and $R^2$ holds $+0.913$ with essentially
no recalibration needed (`R²_recal − R²` $\le 0.045$ everywhere). Against `τ_HP(16,128) = 0.599`, it
clears the hyperparameter-only ceiling by $0.33$.

Stacking the bilinear matrix-product MSG function on ScaleGMN's scale-equivariant states, rather than
the plain GMN's concat-MLP, buys $+0.014$ OOD $\tau$ at $8\times$ over `mpgmn` and turns `Δτ` from
$+0.005$ to $-0.019$, at no in-distribution cost ($0.9062$ vs $0.9057$ at w16) — the same ordering the
INR side reports for "architecture > parameterization > readout/aggregation".

Under SP: OOD $\tau$ $0.8937$ vs `mpgmn` SP's $0.8523$, `Δτ` at $8\times$ is $0.041$ vs $0.128$. Its
SP↔muP16 OOD $\tau$ gap is $0.032$ at $8\times$ (against `mpgmn`'s $0.059$, ScaleGMN dup-equiv's
$0.022$, plain GMN dup-equiv's $0.142$); by $32\times$ the gap widens to $0.071$ and SP's $R^2$ is
down to $+0.468$ (`R²_recal` $+0.699$). Lipschitz continuity in the normalized spectral norm bounds
how fast the output can move; it does not hold the output fixed when the input norms grow
$2.9\times$, and the SP arm's labels and norms are themselves drifting.

**The w192–w512 columns carry the same [SP caveat](#data) as the rest of this experiment.** Only
0.654 of w384 SP labels sit inside w16's central 98% (median shifted $+0.16$) and `‖w‖_Vn` has grown
$7.7\times$ by w512, so part of the SP decay past w128 is a shifted target rather than a metanetwork
failure. muP16 passes over the whole ladder (0.863–0.869 inside support, `W₁(‖w‖) ≤ 2.11`), so the
muP16 column is the interpretable one.

### $\tau$ within Hyperparameter Strata

Diagnostic 4, computed 2026-09-12 by `scripts/summarize_predgen_hp_strata.py --dataset cifar10`
from the `preds_*.npz` dumps of the runs already recorded above (no new training, no new wandb
runs). Restricting the pairs to one stratum removes the hyperparameter shortcut: what survives is
what the weights carry beyond the recipe. Read against [`τ_HP`](#baseline-ranking-via-hyperparameters-τ_hp).

Two stratifications, both pooled the way $\tau_b$ is defined — $\sum_s (C_s - D_s)$ over
$\sum_s \sqrt{(n_{0s} - n_{1s})(n_{0s} - n_{2s})}$ — so unequal strata are weighted by the
comparable pairs they contribute and ties are handled as in the unstratified number:

- `lr` — $10$ learning-rate deciles ($100$ CNNs each, $9.9\%$ of all pairs retained).
- `hp` — exact $(\text{dropout}, \text{epochs}, \text{train\_frac})$ crossed with LR terciles
  ($108$ strata, $\approx 9$ CNNs each, $0.92\%$ of pairs). Thin; read as a bound, not a point
  estimate.

The join is positional: `CNNZooDataset` reads `[split]["path"]` in file order and the test loaders
are `shuffle=False`, so dump row $i$ is split row $i$. Verified by asserting `w{N}_actual` equals
that width's split `score` for every width of every condition, and by reproducing each condition's
unstratified per-width $\tau$ against the tables above.

OOD columns are means over w32–w512, except `mpsgmn-deq-v1` (w32–w128). The four `symmetry: scale`
bidirectional conditions are [out of scope](#experiment-status) and not reported.

| Condition | Arm | $\tau$ (w16) | OOD $\tau$ | OOD $\tau$ (`lr`) | $\Delta$ | OOD $\tau$ (`hp`) | $\Delta$ |
|---|---|---|---|---|---|---|---|
| gmn-base | mup16 | $+0.9131$ | $+0.7364$ | $+0.6518$ | $-0.0846$ | $+0.5715$ | $-0.1649$ |
| gmn-base | sp | $+0.9131$ | $+0.5326$ | $+0.6112$ | $+0.0786$ | $+0.4708$ | $-0.0618$ |
| gmn-deq | mup16 | $+0.9075$ | $+0.7981$ | $+0.7994$ | $+0.0012$ | $+0.7404$ | $-0.0578$ |
| gmn-deq | sp | $+0.9075$ | $+0.4714$ | $+0.6420$ | $+0.1706$ | $+0.4262$ | $-0.0453$ |
| sgmn-base | mup16 | $+0.9041$ | $+0.8402$ | $+0.7888$ | $-0.0514$ | $+0.6742$ | $-0.1660$ |
| sgmn-base | sp | $+0.9041$ | $+0.6490$ | $+0.7391$ | $+0.0901$ | $+0.5561$ | $-0.0929$ |
| sgmn-deq | mup16 | $+0.9039$ | $+0.8526$ | $+0.8299$ | $-0.0227$ | $+0.7816$ | $-0.0711$ |
| sgmn-deq | sp | $+0.9039$ | $+0.8176$ | $+0.7754$ | $-0.0421$ | $+0.7170$ | $-0.1005$ |
| mpgmn-deq | mup16 | $+0.9057$ | $+0.8971$ | $+0.8801$ | $-0.0170$ | $+0.8258$ | $-0.0712$ |
| mpgmn-deq | sp | $+0.9057$ | $+0.7554$ | $+0.7723$ | $+0.0169$ | $+0.7091$ | $-0.0463$ |
| gmn-base-bidir | mup16 | $+0.9122$ | $+0.7368$ | $+0.6746$ | $-0.0622$ | $+0.5674$ | $-0.1694$ |
| gmn-base-bidir | sp | $+0.9122$ | $+0.5338$ | $+0.6020$ | $+0.0682$ | $+0.3901$ | $-0.1437$ |
| gmn-deq-bidir | mup16 | $+0.9106$ | $+0.7980$ | $+0.8109$ | $+0.0129$ | $+0.7548$ | $-0.0433$ |
| gmn-deq-bidir | sp | $+0.9106$ | $+0.4756$ | $+0.6505$ | $+0.1750$ | $+0.4115$ | $-0.0641$ |
| mpgmn-deq-bidir | mup16 | $+0.9017$ | $+0.8902$ | $+0.8705$ | $-0.0197$ | $+0.8018$ | $-0.0885$ |
| mpgmn-deq-bidir | sp | $+0.9017$ | $+0.7648$ | $+0.7609$ | $-0.0039$ | $+0.6671$ | $-0.0977$ |
| mpsgmn-deq-v1 | mup16 | $+0.9062$ | $+0.9252$ | $+0.8934$ | $-0.0318$ | $+0.8446$ | $-0.0806$ |
| mpsgmn-deq-v1 | sp | $+0.9062$ | $+0.8937$ | $+0.8686$ | $-0.0251$ | $+0.8289$ | $-0.0648$ |
| mpsgmn-deq-v2 | mup16 | $+0.9062$ | $+0.9219$ | $+0.8908$ | $-0.0311$ | $+0.8392$ | $-0.0827$ |
| mpsgmn-deq-v2 | sp | $+0.9062$ | $+0.8509$ | $+0.8188$ | $-0.0321$ | $+0.7712$ | $-0.0797$ |

The ordering of conditions is unchanged under `lr` on muP16: `mpsgmn-deq` $>$ `mpgmn-deq` $>$
`sgmn-deq` $>$ `gmn-deq` $>$ `gmn-base`, as in the unstratified tables. Every muP16 condition stays
at or above every width's `τ_HP` ($0.599$ at w128, $0.524$ at w512) after `lr` stratification except
`gmn-base` ($0.6518$) and `gmn-base-bidir` ($0.6746$), which sit at it.

Five SP conditions **gain** under `lr` stratification, largest `gmn-deq-bidir` $+0.1750$ and
`gmn-deq` $+0.1706$ — the two the diagnostic was requested for. The predictions do not collapse
within a recipe; they stop reproducing the accuracy gaps *between* recipes. At w512 `gmn-deq — sp`
has unstratified $\tau = +0.0175$ but per-decile $\tau$ of $+0.675$/$+0.598$ in the two lowest LR
deciles, decaying to $+0.025$/$+0.060$ in the two highest, while its predicted mean stays in
$[+0.250, +0.389]$ across deciles against a true mean spanning $[+0.197, +0.479]$. So the SP
collapse past w128 is a compression of the prediction range, not a loss of within-recipe ordering,
which is consistent with the [SP caveat](#data) — the input norms have grown $7.7\times$ and the
predictions saturate.

Under the stricter `hp` strata every condition loses ($\Delta \in [-0.1694, -0.0433]$). Plain GMN
baseline loses most on both arms and in both directions ($-0.1649$, $-0.1694$ muP16); the two
matrix-product families lose least. Nothing inverts: `mpsgmn-deq-v1 — mup16` is still the best
condition in the study at $+0.8446$, i.e. $91\%$ of its unstratified OOD $\tau$ survives with the
recipe held exactly fixed and only $\approx 9$ comparable CNNs per stratum.

### Mandatory Diagnostics

A per-width $\tau$ table on its own does not identify what OOD size generalization means here, because
the width axis moves three things at once (the CNNs' own accuracies, their weight norms, and the
hyperparameter→accuracy map). Each of the following is required before any claim is made.

1. **Per-width accuracy distributions** — quantiles and histograms of the labels at every width. If
   the label distribution at w128 barely overlaps w16's, a $\tau$ drop is a support problem, not a
   generalization failure. **Done for w32–w512** (2026-08-22, extended 2026-08-29). Fires for SP past
   w128 (0.654 at w384, 0.678 at w512, median shift $+0.16$) but not for muP16 (flat 0.863–0.869 over
   the whole $32\times$ ladder). Table: [Input-network zoo quality](#input-network-zoo-quality).
2. **Baseline: ranking via hyperparameters** — `τ_HP(16, w)`: the Kendall $\tau$ between the
   w16 and w_w accuracies of the *same* hyperparameter draw, on the paired set, plus
   accuracy-vs-base-LR curves per arm. This is the ceiling a hyperparameter-only predictor would hit.
   **Done for w32–w512** (2026-08-22, extended 2026-08-29): `τ_HP(16, 128) = 0.640` (SP) / `0.599`
   (muP16), falling to `0.457` (SP) / `0.524` (muP16) at w512, with the arms crossing over from w256
   on (SP below muP16 thereafter). Table: [Input-network zoo quality](#input-network-zoo-quality).
3. **Per-width $\tau$ / $R^2$ / L1 / $R^2_{recal}$** for all twelve conditions ([definitions](#metrics)).
   `R²_recal` vs `R²` separates a collapsed ordering from a drifted scale. **Done for every forward
   condition** (2026-08-25, extended to w512 2026-08-29), inlined in the per-model tables above: every
   baseline keeps $\tau$ while $R^2$ collapses (worst: plain GMN SP baseline, $R^2 = -3.983$ at w512)
   with `R²_recal` recovering most of it.
4. **$\tau$ within fixed-LR HP strata** — $\tau$ computed inside groups that share the sampled
   hyperparameters, which removes the HP signal and leaves only what the weights carry. **Done
   2026-09-12 for all 20 in-scope conditions**: no condition depends on the shortcut. Every muP16
   condition keeps its OOD $\tau$ to within $[-0.085, +0.013]$ under LR-decile strata, and the five
   SP conditions that fall below `τ_HP` **gain** (up to $+0.175$), so their unstratified collapse is
   a compressed prediction range across recipes, not lost within-recipe ordering. Table:
   [$\tau$ within hyperparameter strata](#tau-within-hyperparameter-strata).
5. **Per-width normalized spectral norms** $\sqrt{n_{l-1}/n_l}\cdot\|W^{(l)}\|_2$ (convs flattened to
   `[c_out, c_in·k_h·k_w]`), and `W₁(μ₁₆, μ_w)`, the 1-Wasserstein distance between the w16 and w_w
   norm distributions. This is what the muP16 arm holds flat and the SP arm does not. **Done for
   w32–w512** (2026-08-22, extended 2026-08-29): muP16 `W₁ ≤ 2.11` over the whole ladder (mean norm
   $38.84 \to 38.14$, i.e. $-1.8\%$ across $32\times$ of width) against SP's $14.20 \to 261.37$
   ($38.84 \to 300.21$, a $7.7\times$ growth against $\sqrt{32} = 5.66$). Table:
   [Input-network zoo quality](#input-network-zoo-quality).
6. **wandb run IDs for every training run, w16–w128 eval run, and w192–w512 eval run reported** (the
   w192–w512 re-scores are each logged as their own wandb eval run too — e.g. ScaleGMN SP baseline
   `qs01lv2v` — transcribed here from the consolidated `/tmp/predgen_sizegen_v2/SUMMARY.md` and
   audited against its w16–w128 wandb run via `v1 agreement`).

---

## Experiment Status

| Experiment | What | State |
|---|---|---|
| 0 | GMN properties on CNN graphs | **done** — [VERIFICATION_gmn_properties.md § CNN graphs](VERIFICATION_gmn_properties.md#on-cnn-inputs), all checks PASS |
| 1 | Reproduce accuracy prediction at w16 | **done** — [table above](#results), ScaleGMN $\tau = 0.9316$ |
| 2 | Widening sanity check, 2 families $\times$ 9 conditions | **done** 2026-09-12 — [tables above](#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check): on `uniform` all 6 dup-equiv/mp conditions are flat at every width (`max \|Δy\| ≤ 1.2e-7`) and the 3 baselines drift; on `general` only the 2 forward matrix-product conditions are flat |
| 3a | Zoo generation, muP CNN, splits | **done** 2026-08-22 00:41 — 5.9 h wall on five GPUs ($\approx 26$ GPU-hours), 675 GB; both arms' splits written, w16 muP16 hard-linked; quality checked, [muP16's norms flat and SP's $2.9\times$](#per-width-normalized-spectral-norms) |
| 3b | Size-gen training, 10 conditions | **done** 2026-08-25 04:37 — all 10 conditions, 3.2-day drain on five GPUs. Best OOD: conv matrix-product GMN on muP16, $\tau$ $0.9115$ OOD / $0.9006$ at w128 ($\Delta\tau = 0.005$ over $8\times$). `/tmp/predgen_sizegen_v1/SUMMARY.md` |
| 3c | This document | **done** — Experiment 1 / 2 / zoo-quality tables, all twelve Experiment 3 conditions' $\tau$ + $R^2$ tables, and (2026-09-12) all six mandatory diagnostics, the last being [diagnostic 4](#tau-within-hyperparameter-strata) |
| 3d | Zoo extension: w192/256/384/512, both arms | **DONE 2026-08-29 02:35** — all 8 arm-width jobs, **76.3 GPU-h** over 3.6 days serialized on one GPU, within $3.4\%$ of the cohort-count model's 79 h. Both `finalize()` split rewrites ran and `splits_md5_changed.txt` is **empty** — none of the eight pre-existing splits changed, so the w16–w128 checkpoints are valid at w512 too. [progress](#input-network-zoo-quality) |
| 3e | Forward re-score, 10 conditions to w16–w512 | **DONE — all ten scored**, inlined in each model's `#### Forward` table above. Eval-only from the w16–w128 checkpoints, so no training; each condition self-checks its w16–w128 $\tau$ against the original wandb eval run (`v1 agreement: exact`). Queue order was muP16 first (the controlled arm), `mpgmn` first within each arm. Prerequisite diagnostic re-run (09:28): muP16 valid over the full $32\times$ ladder, SP not past w128 (see the [SP caveat](#data)). `/tmp/predgen_sizegen_v2/SUMMARY.md` |
| 3f | Bidirectional arm, 10 conditions | **DONE 2026-08-28 14:05** — 5 trainings / 10 conditions in **75.9 GPU-h** over 44.1 h on three GPUs, $4.0\times$ cheaper than the forward arm's $\approx 300$ h and within $5\%$ of the projection; all ten `w16 identity` audits OK. Result: the direction is neutral on CNN graphs ($\Delta \in [-0.020, +0.039]$ OOD $\tau$, median $|\Delta| = 0.009$, 7 of 10 improving), so the forward-only scoping of the w192–w512 extension loses nothing; the FMNIST $-29.4$ pp collapse does not replicate. **⚠ Its four `symmetry: scale` conditions are invalid as of 2026-08-29** — trained without the reciprocal backward edge feature a bidirectional ScaleGMN is defined with, so they are not scale-equivariant; a retrain (`scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh`) was attempted and **dropped as out of scope** — NaNs by training step ~13 at every LR, the same elementwise-division instability the ScaleGMN paper (arXiv:2406.10685, Appendix A.2) reports and never resolved for positive-scale symmetry. The six `symmetry: permutation` rows stand, and their $\Delta$ alone still give the neutrality conclusion. **Extended to w512 2026-08-31** — the same six rows re-scored from their existing checkpoints (`scripts/run_cifar10_predgen_sizegen_v2_bidir.sh`, no retraining), `v1 agreement: OK` and `w16 identity: OK` for all six; the invalid ScaleGMN conditions were deliberately not extended. Each model's `#### Bidirectional` subsection above · `/tmp/predgen_sizegen_v1_bidir/SUMMARY.md`, `/tmp/predgen_sizegen_v2_bidir/SUMMARY.md` |
| 4 | Conv matrix-product GMN | **done** — layer, installer, 2 configs, and `scripts/check_matrix_product_gmn_cnn.py` (all 3 tests PASS) |
| 5 | Conv matrix-product **ScaleGMN** | **DONE 2026-08-29 07:29** — one training (13.7 h on GPU 0, 200 epochs at lr 1e-3, run `m8z4d8t9`) scored as 4 conditions: {SP, muP16} × {v1 w16–w128, v2 w16–w512}, all four `w16 identity` audits OK and every v2 row at w16–w128 reproducing its v1 row exactly. Best condition in the study on muP16 and the first whose OOD $\tau$ does not decay: OOD $\tau$ **0.9252** (w32–w128) vs the previous best $0.9115$, with `Δτ = −0.019` at $8\times$ and still `−0.006` at $32\times$ (w512 $\tau$ $0.9123$, $R^2$ $+0.913$). SP gains too — OOD $\tau$ $0.8937$ vs `mpgmn` SP's $0.8523$, $\Delta\tau$ $0.041$ vs $0.128$ — but decays by $32\times$ (w512 $\tau$ $0.7471$). In distribution it is indistinguishable from both references (w16 $\tau$ $0.9062$). [section](#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward) · `/tmp/mpsgmn_predgen/SUMMARY.md` |

**Gate V5 — cleared 2026-08-21.** Mass zoo generation was held for a human check of the muP CNN
implementation and the coordinate-check figures
(`results/figures/coord_check_cnn_{ours,ref}_{mup,sp}.png`); that review passed, generation ran
2026-08-21/22, and the zoo's measured norm statistics
([above](#per-width-normalized-spectral-norms)) are the downstream confirmation that the muP arm
behaves as the implementation claims: `‖w‖_Vn` flat to `W₁ ≤ 2.11` over a $32\times$ width range
(measured to w512 on 2026-08-29) where SP grows $7.7\times$. Verification results: V1 PASS;
base-width identity $0.00\mathrm{e}{+00}$; V2 (ours) muP drifts $\times 1.18$ across w16–w512 vs SP
$\times 45.2$ (the SP control's magnitude varies by a few tens of percent between runs — it is a
must-diverge control, not a measured constant); V2 (reference implementation) muP $\times 1.30$ vs SP
`convs.1` $\times 6.11$; V3 mean gap $\le 4.2\times 10^{-5}\cdot\mathrm{lr}$ over 24 rows (3 sampled
draws + one pinned $\lambda = 10^{-1}$ draw, $\times 3$ widths $\times 2$ arms), plus an
$\varepsilon = 10^{-3}$ row at $8.2\times 10^{-6}\cdot\mathrm{lr}$, with all three must-fail controls
firing — broken lr table at $1.8\cdot\mathrm{lr}$, double-counted `MuAdam` wd factor at
$8.2\times 10^{-2}\cdot\mathrm{lr}$, unscaled $\varepsilon$ at $3.5\times 10^{-2}\cdot\mathrm{lr}$;
V4 in the low $10^{-7}$–$10^{-6}$ range (tol $10^{-5}$; it moves by $\approx 2\times$ between runs
with the `vmap`'d convolution's reduction order). The **init table** is the one thing no numbered
check asserts — V1 checks `infshape`s, V3 copies the trainer's parameters into the reference before
stepping, and V2-ours gates at $t=20$; `scripts/summarize_zoo_cnn.py` covers it instead, by verifying
its own restated init formula against the trainer's realized per-parameter std. Three deviations
from the reference `mup` package are documented and deliberate: fixed optimizer/activation,
zero-initialized biases, and scaled Adam $\varepsilon$ — $\varepsilon_0/wm$ on every parameter with
an infinite dimension, since all of their gradients are $\Theta(1/c)$ in this parameterization while
`mup` leaves $\varepsilon$ fixed (Everett et al. 2024 §4.3; measured gradient-RMS ratios
$0.025$–$0.034$ over a $32\times$ width increase against the $0.031$ prediction, with the finite
output bias flat at $1.00$). At $\varepsilon_0 = 10^{-8}$ this changes nothing measurable at zoo
widths — the median $\sqrt{\hat v}$ is still $\approx 800\times$ above $\varepsilon_0$ at w512 — so
it is a correctness fix rather than a change to the numbers below.
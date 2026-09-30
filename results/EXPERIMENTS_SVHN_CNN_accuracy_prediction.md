# Experiments — SVHN-GS CNN Accuracy Prediction

The [CIFAR-10 CNN accuracy-prediction study](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md)
replicated on grayscale SVHN. Architecture, metanetwork task, roles, metrics, LR protocol, readout,
epoch budget and model-selection rule are identical — only the images the input CNNs were trained on
change — so every table here is a drop-in comparison against its CIFAR-10 counterpart.

## Contents
- [Task and data](#task-and-data) — the label distribution ties $18.4\%$ of all pairs, capping $\tau_b$ at $0.903$
- [Input-network zoo quality](#input-network-zoo-quality) — muP16's norms are flat, SP's grow $8.0\times$; the $\tau_b$ ceiling **moves with width** ($0.849$ at w16, $0.963$ at w384)
- [Experiment 1 — Reproduce accuracy prediction at the zoo's native width](#experiment-1--reproduce-accuracy-prediction-at-the-zoos-native-width-w16) (fixed width w16)
- [Experiment 2 — Size generalization on wider equivalent CNNs](#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check) (sanity check, 2 widening families $\times$ 9 conditions)
- [Experiment 3 — Size generalization on our multi-width zoo](#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512) (train w16, test w16–w512) — all seventeen conditions done on w16–w512
  - [$\tau$ within hyperparameter strata](#tau-within-hyperparameter-strata) (diagnostic 4)
- [Mandatory diagnostics](#mandatory-diagnostics)
- [Experiment status](#experiment-status)

---

## Task and Data

Input networks: the zoo CNN
`[1] → conv3×3/s2 → [c] → conv3×3/s2 → [c] → conv3×3/s2 → [c] → GAP → dense → [10]`, ReLU, with
$18c^2 + 21c + 10$ parameters. Metanetwork head: $\mathrm{sigmoid}(\mathrm{net}(\cdot))$ against the
CNN's test accuracy, MSE loss. Roles (`train` n=8000 seed 1000 w16 only; `val` n=1000 seed 2000 w16
only; `test` n=1000 per width seed $3000+w$; `paired` n=200 per width seed 9000), metric definitions
($\tau_b$ / $R^2$ / L1 / $R^2_{recal}$) and the "select on val $\tau$ at the train width only" rule
are unchanged from the
[CIFAR-10 study](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#task-and-data).

**SVHN-GS** = the 32×32 SVHN digit crops reduced to one channel by the ITU-R 601-2 luma weights,
73257 train / 26032 test images (`load_svhn_gray()` in `src/data/train_zoo_cnns.py`). Upstream treats
SVHN as a drop-in sibling of CIFAR-10 for this task — same `NFNZooDataset`, same
`layer_layout: [1, 16, 16, 16, 10]`, same `gresearch/smallcnnzoo-dataset` bucket.

Two differences from CIFAR-10 that the tables below cannot be read without:

1. **A collapsed CNN scores $0.196$, not $0.10$.** SVHN's test-set majority class is digit "1" at
   frequency $0.1959$, so the floor a degenerate input CNN lands on is the majority-class rate, well
   above the $1/10$ chance level CIFAR-10's balanced test set gives.
2. **$1.47\times$ the training images** (73257 vs 50000), so every zoo-generation cost scales by the
   same factor at fixed epoch count.

### Upstream zoo (Experiment 1's data)

`$ANYDIM_DATA_ROOT/svhn_zoo/` (`scripts/download_cnn_zoo.sh svhn`): 269892 metric rows, 29988 models
at the final checkpoint (step 86), of which **15122 are ReLU** — split 6151 train / 1466 val / 7505
test by upstream's hardcoded `svhn_split.csv` shuffle.

Test-accuracy distribution over those 15122 ReLU models:

| min | p10 | q25 | median | q75 | p90 | max | mean | frac $\le 0.11$ | frac $\le 0.196$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.101 | 0.196 | 0.196 | 0.270 | 0.457 | 0.596 | 0.781 | 0.334 | 0.0002 | **0.466** |

$46.6\%$ of the zoo sits at or below the majority-class rate $0.1959$ — a hard tie mass at the
collapsed-predictor accuracy. Almost nothing is at the $1/10$ level ($0.02\%$ below $0.11$), against
CIFAR-10's $22.7\%$ at or below its own chance floor: SVHN's failed draws collapse onto the majority
class rather than to uniform.

That tie mass bounds the metric itself. With $18.4\%$ of all $\binom{n}{2}$ pairs tied in $y$ and no
ties in $\hat y$, $\tau_b = (C-D)/\sqrt{(n_0-n_1)(n_0-n_2)}$ is maximized at
$\sqrt{(n_0-n_2)/n_0}$, so a **perfect** ranker reads

$$\tau_b^{\max} = 0.9031 \quad \text{(SVHN)} \qquad \text{against} \qquad 0.9883 \quad \text{(CIFAR-10)}.$$

Every $\tau$ in this document must be read against $0.903$, not against $1$ or against CIFAR-10's
numbers directly. Fractions of the ceiling are the comparable quantity. Both ceilings are computed
over each zoo's full ReLU population, not its test split — reconstructing upstream's exact test
subset from the hardcoded shuffle is not worth it, since the split is a contiguous half of that
shuffle and so carries the same tie mass.

---

## Input-network zoo quality

**Status: done 2026-09-03**, on the finished w16–w512 zoo (Stage 3a 2026-09-01 22:08, Stage 3d
2026-09-03 16:25). All numbers are read off the `meta_{role}.jsonl` records `generate_cnn_zoo.py`
writes — no model is loaded and no CNN is re-run — by

```bash
source .env && python scripts/summarize_cnn_zoo_quality.py --dataset svhn
```

on the `test` role (n = 1000 per width per arm), except `τ_HP`, which needs the `paired` role. The
floor is the majority-class rate $0.1959$, not $0.1$.

### Per-width accuracy distributions (labels)

| Width | arm | n | min | q25 | median | q75 | max | mean | frac ≤ 0.206 |
|---|---|---|---|---|---|---|---|---|---|
| w16 | SP ≡ muP16 | 1000 | 0.137 | 0.196 | 0.196 | 0.306 | 0.700 | 0.260 | **0.598** |
| w32 | SP | 1000 | 0.160 | 0.196 | 0.196 | 0.352 | 0.761 | 0.293 | 0.549 |
| w48 | SP | 1000 | 0.137 | 0.196 | 0.244 | 0.414 | 0.815 | 0.325 | 0.457 |
| w64 | SP | 1000 | 0.166 | 0.196 | 0.275 | 0.468 | 0.840 | 0.345 | 0.436 |
| w96 | SP | 1000 | 0.168 | 0.196 | 0.309 | 0.530 | 0.842 | 0.376 | 0.356 |
| w128 | SP | 1000 | 0.098 | 0.196 | 0.326 | 0.561 | 0.855 | 0.387 | 0.350 |
| w192 | SP | 1000 | 0.177 | 0.196 | 0.355 | 0.608 | 0.850 | 0.414 | 0.326 |
| w256 | SP | 1000 | 0.192 | 0.196 | 0.370 | 0.612 | 0.859 | 0.421 | 0.311 |
| w384 | SP | 1000 | 0.114 | 0.196 | 0.420 | 0.640 | 0.862 | 0.441 | 0.313 |
| w512 | SP | 1000 | 0.159 | 0.196 | 0.393 | 0.627 | 0.854 | 0.428 | 0.339 |
| w32 | muP16 | 1000 | 0.082 | 0.196 | 0.196 | 0.322 | 0.780 | 0.280 | 0.585 |
| w48 | muP16 | 1000 | 0.130 | 0.196 | 0.196 | 0.343 | 0.784 | 0.293 | 0.564 |
| w64 | muP16 | 1000 | 0.124 | 0.196 | 0.196 | 0.366 | 0.833 | 0.311 | 0.529 |
| w96 | muP16 | 1000 | 0.128 | 0.196 | 0.205 | 0.377 | 0.817 | 0.315 | 0.501 |
| w128 | muP16 | 1000 | 0.161 | 0.196 | 0.196 | 0.383 | 0.851 | 0.318 | 0.530 |
| w192 | muP16 | 1000 | 0.168 | 0.196 | 0.209 | 0.404 | 0.871 | 0.328 | 0.496 |
| w256 | muP16 | 1000 | 0.151 | 0.196 | 0.207 | 0.409 | 0.864 | 0.329 | 0.499 |
| w384 | muP16 | 1000 | 0.160 | 0.196 | 0.220 | 0.427 | 0.867 | 0.331 | 0.483 |
| w512 | muP16 | 1000 | 0.164 | 0.196 | 0.198 | 0.404 | 0.867 | 0.329 | 0.521 |

The **median w16 CNN is collapsed** ($0.196$), and $59.8\%$ of the training width's labels sit at or
below the floor, against CIFAR-10's $16.9\%$ at its own floor. Width buys resolution: SP's collapsed
fraction falls $0.598 \to 0.311$ by w256, muP16's only $0.598 \to 0.499$, so SP carries less tie mass
than muP16 at every width $> 16$. That does *not* make SP the easier target overall — it buys the
lower tie mass by moving the labels out of w16's support (next table), which is the trade the two
arms make against each other throughout this study.

Support overlap with the training width, same role:

| Width | arm | `W₁(y₁₆, y_w)` | frac inside w16's [q1, q99] | median shift |
|---|---|---|---|---|
| w32 | SP | 0.0327 | 0.928 | +0.0000 |
| w48 | SP | 0.0652 | 0.876 | +0.0480 |
| w64 | SP | 0.0852 | 0.846 | +0.0794 |
| w96 | SP | 0.1163 | 0.799 | +0.1129 |
| w128 | SP | 0.1268 | 0.761 | +0.1296 |
| w192 | SP | 0.1538 | 0.715 | +0.1589 |
| w256 | SP | 0.1607 | 0.704 | +0.1741 |
| w384 | SP | 0.1809 | **0.652** | +0.2238 |
| w512 | SP | 0.1678 | **0.700** | +0.1972 |
| w32 | muP16 | 0.0204 | 0.948 | +0.0000 |
| w48 | muP16 | 0.0336 | 0.911 | +0.0000 |
| w64 | muP16 | 0.0511 | 0.876 | +0.0002 |
| w96 | muP16 | 0.0548 | 0.880 | +0.0090 |
| w128 | muP16 | 0.0577 | 0.864 | +0.0002 |
| w192 | muP16 | 0.0686 | 0.850 | +0.0133 |
| w256 | muP16 | 0.0693 | 0.852 | +0.0107 |
| w384 | muP16 | 0.0713 | 0.855 | +0.0242 |
| w512 | muP16 | 0.0693 | 0.847 | +0.0024 |

On `W₁` and `frac inside` both arms track their CIFAR-10 counterparts to within $\approx 0.03$ at
every width, including SP's diagnostic-1 failure at w384 (`frac inside` $0.652$, against CIFAR-10's
$0.654$) and muP16's flat $0.847$–$0.880$ from w192 to w512 with `W₁ ≤ 0.071`. The median-shift
column does not transfer: SP's is larger than CIFAR-10's ($+0.224$ vs $+0.161$ at w384) and muP16's
is smaller ($\le +0.024$ over the whole ladder, against $+0.062$), both for the same reason — SVHN's
median sits on the collapsed floor at $0.196$, so it stays pinned under muP16 and jumps in one step
under SP once fewer than half the draws collapse.

### Per-width $\tau_b$ ceiling — SVHN only, and it moves with width

The $18.4\%$ tie mass that caps the *upstream* zoo at $\tau_b^{\max} = 0.9031$
([above](#upstream-zoo-experiment-1s-data)) is a property of the label distribution, so in our
multi-width zoo it is a **per-width, per-arm** quantity. Computed as
$\tau_b^{\max} = \sqrt{(n_0 - n_2)/n_0}$ on each `test` role ($n = 1000$, $n_2$ = tied $y$ pairs, no
ties in $\hat y$):

| Width | SP tie frac | SP $\tau_b^{\max}$ | muP16 tie frac | muP16 $\tau_b^{\max}$ |
|---|---|---|---|---|
| w16 | 0.2786 | **0.8494** | 0.2786 | **0.8494** |
| w32 | 0.2498 | 0.8661 | 0.2828 | 0.8469 |
| w48 | 0.1737 | 0.9090 | 0.2619 | 0.8591 |
| w64 | 0.1614 | 0.9158 | 0.2254 | 0.8801 |
| w96 | 0.1081 | 0.9444 | 0.2096 | 0.8891 |
| w128 | 0.0978 | 0.9498 | 0.2321 | 0.8763 |
| w192 | 0.0845 | 0.9568 | 0.1960 | 0.8966 |
| w256 | 0.0738 | 0.9624 | 0.1952 | 0.8971 |
| w384 | 0.0728 | **0.9629** | 0.1856 | 0.9025 |
| w512 | 0.0800 | 0.9592 | 0.2170 | 0.8849 |

The ceiling **rises** with width — most at the training width, $0.8494$, and $+0.113$ higher at w384
under SP. The same computation on CIFAR-10's `test` roles gives $0.9925$ (w16) to $0.9997$ (w512),
i.e. flat and $\approx 1$ at every width, so this is a hazard SVHN introduces and CIFAR-10 does not:
**a raw per-width $\tau$ curve is not comparable across widths in this study.** A condition holding
$\tau$ flat from w16 to w384 under SP is losing $12\%$ of the attainable ranking, and a condition
whose $\tau$ *rises* with width may be doing nothing but tracking the ceiling. Experiment 3's SP
columns must be read as $\tau / \tau_b^{\max}(w)$; the muP16 arm is nearly exempt ($0.847$–$0.903$,
a $0.056$ spread) because its collapsed fraction barely moves.

### Per-width normalized spectral norms

$\|w\|_{Vn} = \sum_l \sqrt{n_{l-1}/n_l}\cdot\|W^{(l)}\|_2$ on the **effective** weights, convs
flattened to `[c_out, c_in·k_h·k_w]`, plus `W₁(μ₁₆, μ_w)`.

| Width | SP mean `‖w‖_Vn` | SP `W₁(μ₁₆, μ_w)` | muP16 mean `‖w‖_Vn` | muP16 `W₁(μ₁₆, μ_w)` |
|---|---|---|---|---|
| w16 | 34.85 | 0 (by definition) | 34.85 | 0 (by definition) |
| w32 | 47.49 | 12.65 | 34.22 | 1.87 |
| w48 | 64.06 | 29.22 | 35.96 | 1.42 |
| w64 | 74.46 | 39.61 | 37.76 | 2.94 |
| w96 | 92.37 | 57.52 | 36.75 | 2.27 |
| w128 | 109.66 | 74.82 | 36.27 | 1.86 |
| w192 | 138.40 | 103.56 | 37.84 | 3.15 |
| w256 | 169.01 | 134.16 | 37.82 | 3.07 |
| w384 | 228.62 | 193.78 | 37.61 | 3.20 |
| w512 | 279.73 | 244.88 | 36.67 | 2.40 |

muP16's `W₁` is $1.4$–$3.2$ at every width and no larger at w512 than at w192, against SP's
$12.65 \to 244.88$ — $40\times$ smaller at w128, $102\times$ at w512. muP16's mean moves
$34.85 \to 36.67$ over $32\times$ width ($+5.2\%$); SP's grows $8.03\times$, against
$\sqrt{32} = 5.66$ predicted by $\Theta(\sqrt n)$. CIFAR-10's counterparts are $-1.8\%$ and
$7.7\times$. Per-layer means (conv1 / conv2 / conv3 / fc):

- SP: w16 `1.61 / 16.59 / 13.78 / 2.86` → w128 `1.18 / 56.49 / 40.74 / 11.26` → w512 `1.02 / 140.93 / 108.68 / 29.10`
- muP16: w16 `1.61 / 16.59 / 13.78 / 2.86` → w128 `1.74 / 18.27 / 13.01 / 3.24` → w512 `1.88 / 18.97 / 12.57 / 3.25`

The growth is in the two inner convs and the readout — the layers with two infinite dimensions or an
infinite fan-in — exactly as on CIFAR-10.

### Baseline: ranking via hyperparameters (`τ_HP`)

`τ_HP(16, w)` = Kendall $\tau$ between the w16 and w_w accuracies of the *same* hyperparameter draw,
on the width-independent [`paired` role](#task-and-data), $n = 200$, standard error $\approx \pm 0.05$. This
is the floor a result must clear before "it generalizes across width" is the right description.

| Width | SP `τ_HP` | muP16 `τ_HP` |
|---|---|---|
| w32 | 0.7075 | 0.7575 |
| w48 | 0.6875 | 0.7350 |
| w64 | 0.6321 | 0.6496 |
| w96 | 0.5966 | 0.6505 |
| w128 | 0.5913 | 0.6330 |
| w192 | 0.5234 | 0.6202 |
| w256 | 0.4873 | 0.6063 |
| w384 | 0.4179 | 0.6059 |
| w512 | 0.3812 | 0.6212 |

Two departures from CIFAR-10, both in the same direction:

1. **No crossover.** muP16's ceiling is above SP's at *every* width ($0.7575$ vs $0.7075$ at w32,
   $0.6212$ vs $0.3812$ at w512), where CIFAR-10 has SP at or above muP16 through w128 and below
   from w256 on. Each arm still gets compared to its own ceiling at its own width.
2. **muP16's ceiling stops falling.** It flattens at $0.606$–$0.621$ from w192 to w512 — a
   hyperparameter-only predictor keeps $\approx 61\%$ of the ranking out to $32\times$ — while SP's
   falls to $0.381$, below CIFAR-10's $0.457$. The w32 → w512 SP decline ($0.708 \to 0.381$) is
   $\approx 6.5$ standard errors.

Mean label by sampled-LR quartile (edges $5.0\mathrm{e}{-5}$ / $3.0\mathrm{e}{-4}$ /
$1.5\mathrm{e}{-3}$ / $8.5\mathrm{e}{-3}$ / $5.0\mathrm{e}{-2}$), `test` role: SP w16
$0.210 / 0.267 / 0.332 / 0.231$ → w512 $0.454 / 0.554 / 0.450 / 0.249$; muP16 w16 the same → w512
$0.214 / 0.292 / 0.448 / 0.365$. The two arms put their width gains in opposite quartiles. SP gains in
the *low*-LR quartiles ($+0.244$ and $+0.287$ on Q1/Q2) while its top quartile is pinned at
$0.22$–$0.27$ at every width: SP never makes a high-LR wide CNN trainable. muP16 gains in the *high*-LR
quartiles ($+0.116$ and $+0.134$ on Q3/Q4) with Q1 flat at $0.21$ — the expected signature of muP's LR
transfer. This is also why muP16 keeps *more* tie mass, not less: its wide models cluster in a
narrower accuracy band rather than spreading out the way SP's do.

---

## Experiment 1 — Reproduce Accuracy Prediction at the Zoo's Native Width (w16)

**TL;DR**: both variants reach $\tau \approx 0.866$, i.e. $\approx 96\%$ of SVHN's $\tau_b$ ceiling
$0.9031$, against CIFAR-10's $0.9316$ = $94\%$ of its $0.9883$ ceiling. The plumbing reproduces on
SVHN, and the plain GMN is marginally ahead of ScaleGMN here — the reverse of CIFAR-10's ordering.

**Dataset**: upstream small CNN zoo at the native width w16, split **6151 train / 1466 val / 7505
test**. 200 epochs, `--direction forward`, AdamW lr 1e-3 / wd 0.01, `WarmupLRScheduler`, batch 64 —
the only experiment that uses upstream's trainer (`src/scalegmn/predicting_generalization.py` via
`scripts/train_predgen.py`), upstream's data and upstream's splits.
**Configs**: `configs/svhn_predgen/{scalegmn,gmn}_predgen_sp.yml`
**Date**: 2026-09-01 (00:24–01:14, GPUs 0 and 1)

### Reproduction

```bash
source .env
bash scripts/download_cnn_zoo.sh svhn

screen -dmS svhn_predgen_sgmn bash -c 'source .env && CUDA_VISIBLE_DEVICES=0 \
  python scripts/train_predgen.py --conf configs/svhn_predgen/scalegmn_predgen_sp.yml --wandb True'
screen -dmS svhn_predgen_gmn bash -c 'source .env && CUDA_VISIBLE_DEVICES=1 \
  python scripts/train_predgen.py --conf configs/svhn_predgen/gmn_predgen_sp.yml --wandb True'
```

**Logs**: `/tmp/predgen_svhn_stage1/{scalegmn,gmn}.log`

### Results

| Variant | Best val τ | Test τ (best over epochs) | Test τ (final epoch) | Test R² | Test L1 | Frac. of ceiling | Run ID |
|---|---|---|---|---|---|---|---|
| ScaleGMN (forward, scale) | 0.8705 (ep 198) | 0.8655 | 0.8603 | 0.9922 | 0.0078 | 0.958 | [`6qz30wq7`](https://wandb.ai/yuxinma/svhn_predgen/runs/6qz30wq7) |
| Plain GMN (forward, permutation) | 0.8762 (ep 156) | **0.8672** | 0.8671 | 0.9929 | 0.0078 | **0.960** | [`ocygvhd1`](https://wandb.ai/yuxinma/svhn_predgen/runs/ocygvhd1) |

**PASS.** Unlike the CIFAR-10 table, the test $\tau$ *at the best-val-$\tau$ epoch* is not recoverable
offline — upstream's trainer logs test metrics only to wandb, not to the local log — so both the
running-max (`test/best_tau`) and final-epoch (`test/kendall_tau`) columns are given; they bracket it,
and for the plain GMN they nearly coincide ($0.8672$ vs $0.8671$).

The scale-equivariant inductive bias buys **nothing** at this fixed width on SVHN ($-0.002$ $\tau$,
$-0.006$ val $\tau$), against the $+0.005$ it buys on CIFAR-10. Both variants sit $\approx 0.037$
below the ceiling, so what neither reads is the ordering *within* the tie-free $53\%$ of the zoo.

---

## Experiment 2 — Size Generalization on Wider Equivalent CNNs (sanity check)

**Goal**: take the zoo's w16 test CNNs, widen them to w32, w48, w64, w96, w128 by a
*function-preserving* construction, and predict accuracy from the widened weights. The widened CNN
computes exactly the same function as its w16 source and carries that source's test accuracy as its
label, so every change in $\tau$ is the metanetwork failing to be invariant to the widening — never
distribution shift, never a label shift. Compare against
[Experiment 3](#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512),
where the wider CNNs are trained from random initialization, and with
[CIFAR-10 Experiment 2](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check).

Every width's labels are the **w16** labels, so the $\tau_b$ ceiling is $0.8494$ at every column
([above](#per-width-tau_b-ceiling--svhn-only-and-it-moves-with-width)) — the width-dependent ceiling
of Experiment 3 does not apply here, and a $\tau$ above it is a tie-breaking artefact of the
predictions, not a better ranking.

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
$\mathcal{N}(0, 1)$ probe): `general` gives $\max|\Delta\text{logits}|$ $2.3\times10^{-6}$ (w32) to
$4.5\times10^{-5}$ (w96), with row residual $\le 7.5\times10^{-7}$, column residual $4.92$–$12.5$ and
minimum non-uniformity $2.62$–$4.18$ — it genuinely violates the column condition and never
degenerates to the uniform block. `uniform` gives $\le 4.1\times10^{-7}$, nonzero only at $k = 3, 6$,
where $1/k_{l-1}$ is not representable in float32. Blocks are drawn in float64 and cast on write:
`general`'s are $O(1)$ draws whose row sums cancel to an $O(10^{-2})$ kernel, so a float32
construction would lose the row condition outright.

The evaluator runs in **full fp32, not TF32**. The `max |Δy|` floor is $\approx 10^{-7}$ on `uniform`
and larger on `general`, whose row residual is $\approx 10^{-4}$ *relative* on an $O(10^{-2})$ kernel;
TF32 would add $\approx 10^{-4}$ of its own noise on top. `duplication_equiv` and `readout` are read
back off each checkpoint rather than re-specified.

**Scripts**: `scripts/generate_duplicated_cnns.py --dataset svhn --family` (data),
`scripts/eval_sizegen_predgen_duplicated.py --dataset svhn` (eval),
`scripts/run_predgen_widen_families.sh` (orchestration).
**Configs**: `configs/svhn_predgen/{scalegmn,gmn,mpgmn,mpsgmn}_dup_w16.yml`
**Date**: 2026-09-12 (18:22–19:23 on GPUs 0/1/2: 5 trainings, then 4 eval jobs). Supersedes the
2026-09-01 forward-only `uniform` run of 4 conditions, whose numbers the forward `uniform` rows below
reproduce exactly from the same four checkpoints.

### Reproduction

```bash
source .env

# Data: the `general` tree (w32-w128); `uniform` is w{16..128}_zoo_dup/, already on disk
python scripts/generate_duplicated_cnns.py --dataset svhn --family general

# Phase 1 trains the 5 conditions the 4-condition run never covered (mp-GMN fw, mp-ScaleGMN fw,
# plain GMN bd x2, mp-GMN bd); phase 2 scores {general,uniform} x {forward,bidirectional}, each
# job with --model-type all. Resumable: re-running skips finished jobs.
screen -dmS widen_svhn bash -c 'bash scripts/run_predgen_widen_families.sh svhn "0 1 2"'
```

**Logs**: `/tmp/predgen_svhn_widen_families/` (`SUMMARY.md`, per-job logs,
`preds_{family}_{direction}.npz`; one worker per GPU draining a job queue).

Two families of numbers per width: against the **labels** ($\tau$ / $R^2$ / L1 / $R^2_{recal}$), and
against the **w16 predictions** (`τ(y₁₆, y_w)`, `max |Δy|`, `mean |Δy|`) — the sharp form, since
pairing is by index and every width's split lists the same base models in the same order. `max_drop`
is $\tau(\text{w16}) - \min_w \tau(w)$.

### `general` widening — the largest function-preserving family

Eval runs [`a7l69vx4`](https://wandb.ai/yuxinma/svhn_predgen/runs/a7l69vx4) (forward) and
[`p472p0tg`](https://wandb.ai/yuxinma/svhn_predgen/runs/p472p0tg) (bidirectional), extended
2026-09-15 to w192/w256/w384/w512 (superseding the earlier w16–w128 run of the same name).

#### ScaleGMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | +0.993 | −0.591 | −1.326 | −1.800 | −2.430 | −2.812 | −3.256 | −3.502 | −3.781 | −3.923 | 4.915 |
| ScaleGMN baseline, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| ScaleGMN dup-equiv, fw | +0.992 | −0.534 | −0.879 | −1.026 | −1.134 | −1.187 | −1.288 | −1.338 | −1.392 | −1.419 | 2.412 |
| ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **mp-ScaleGMN dup-equiv, fw** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.993 \to -2.812$ / $0.0077 \to 0.2951$ /
$+0.993 \to +0.455$; dup-equiv $+0.992 \to -1.187$ / $0.0079 \to 0.2127$ / $+0.992 \to +0.435$;
mp-ScaleGMN $+0.991$ / $0.0082$ / $+0.992$ at every width.

#### Plain GMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | +0.992 | −1.213 | −2.104 | −2.480 | −2.868 | −3.041 | −3.211 | −3.294 | −3.375 | −3.413 | 4.405 |
| GMN baseline, bd | +0.993 | −1.309 | −2.363 | −3.130 | −3.989 | −4.374 | −4.729 | −4.885 | −5.032 | −5.099 | 6.093 |
| GMN dup-equiv, fw | +0.992 | −3.486 | −3.842 | −3.861 | −3.843 | −3.834 | −3.820 | −3.810 | −3.800 | −3.791 | 4.853 |
| GMN dup-equiv, bd | +0.993 | −4.098 | −4.885 | −4.872 | −4.670 | −4.529 | −4.361 | −4.268 | −4.173 | −4.124 | 5.877 |
| **mp-GMN dup-equiv, fw** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **0.000** |
| mp-GMN dup-equiv, bd | +0.993 | −1.890 | −2.639 | −2.878 | −3.154 | −3.228 | −3.257 | −3.201 | −3.149 | −2.972 | 4.250 |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.992 \to -3.041$ / $0.0087 \to 0.3055$ /
$+0.993 \to +0.572$ (fw) and $+0.993 \to -4.374$ / $0.0077 \to 0.3546$ / $+0.994 \to +0.496$ (bd);
dup-equiv $+0.992 \to -3.834$ / $0.0083 \to 0.3319$ / $+0.992 \to +0.689$ (fw) and
$+0.993 \to -4.529$ / $0.0085 \to 0.3546$ / $+0.993 \to +0.328$ (bd); mp-GMN $+0.993$ / $0.0076$ /
$+0.993$ at every width (fw) and $+0.993 \to -3.228$ / $0.0074 \to 0.2966$ / $+0.993 \to +0.009$ (bd).

#### Agreement with the w16 predictions

`max |Δy|`:

| Condition | w32 | w48 | w64 | w96 | w128 |
|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 4.1e-01 | 4.5e-01 | 4.9e-01 | 5.1e-01 | 5.1e-01 |
| ScaleGMN dup-equiv, fw | 4.1e-01 | 4.5e-01 | 4.4e-01 | 4.4e-01 | 4.4e-01 |
| **mp-ScaleGMN dup-equiv, fw** | **6.7e-07** | **2.3e-06** | **4.3e-06** | **7.4e-06** | **1.3e-05** |
| GMN baseline, fw | 4.5e-01 | 5.0e-01 | 5.0e-01 | 5.1e-01 | 5.2e-01 |
| GMN baseline, bd | 4.8e-01 | 5.3e-01 | 5.3e-01 | 5.4e-01 | 5.4e-01 |
| GMN dup-equiv, fw | 5.1e-01 | 5.3e-01 | 5.3e-01 | 5.2e-01 | 5.2e-01 |
| GMN dup-equiv, bd | 5.6e-01 | 6.0e-01 | 5.9e-01 | 5.7e-01 | 5.7e-01 |
| **mp-GMN dup-equiv, fw** | **8.5e-07** | **3.5e-06** | **6.9e-06** | **1.6e-05** | **2.5e-05** |
| mp-GMN dup-equiv, bd | 5.8e-01 | 5.8e-01 | 5.8e-01 | 6.1e-01 | 5.9e-01 |

The two forward matrix-product rows are the `general` floor, not zero: the row condition itself holds
only to $\approx 10^{-4}$ relative in float32, giving `mean |Δy|` $2.6\times10^{-8} \to
6.3\times10^{-7}$ (mp-ScaleGMN) / $4.3\times10^{-8} \to 1.8\times10^{-6}$ (mp-GMN) and
$\tau(y_{16}, y_w) \ge 0.9999$ / $\ge 0.9994$. Every other row's $\tau(y_{16}, y_w)$ is within
$0.10$ of its label $\tau$, with `mean |Δy|` $1.7\times10^{-1}$–$3.7\times10^{-1}$.

### `uniform` (Kronecker) widening — the reference family

Eval runs [`v80bxj2q`](https://wandb.ai/yuxinma/svhn_predgen/runs/v80bxj2q) (forward) and
[`zmmacgo9`](https://wandb.ai/yuxinma/svhn_predgen/runs/zmmacgo9) (bidirectional), extended
2026-09-15/16 to w192/w256/w384/w512 (superseding the earlier w16–w128 run of the same name).

#### ScaleGMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | +0.993 | +0.977 | +0.946 | +0.907 | +0.815 | +0.723 | +0.558 | +0.421 | +0.220 | +0.090 | 0.902 |
| ScaleGMN baseline, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **ScaleGMN dup-equiv, fw** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **0.000** |
| ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |
| **mp-ScaleGMN dup-equiv, fw** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **+0.991** | **0.000** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | out of scope |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.993 \to +0.723$ / $0.0077 \to 0.0656$ /
$+0.993 \to +0.899$; dup-equiv $+0.992$ / $0.0079$ / $+0.992$ and mp-ScaleGMN $+0.991$ / $0.0082$ /
$+0.992$ at every width.

#### Plain GMN

**Per-width $R^2$** (recomputed from the saved predictions, `scripts/render_widen_r2_tables.py`):

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | +0.992 | +0.845 | +0.579 | +0.394 | +0.220 | +0.155 | +0.110 | +0.097 | +0.088 | +0.086 | 0.905 |
| GMN baseline, bd | +0.993 | +0.874 | +0.561 | +0.267 | −0.002 | −0.073 | −0.118 | −0.133 | −0.143 | −0.146 | 1.139 |
| **GMN dup-equiv, fw** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **+0.992** | **0.000** |
| **GMN dup-equiv, bd** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **0.000** |
| **mp-GMN dup-equiv, fw** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **0.000** |
| **mp-GMN dup-equiv, bd** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **+0.993** | **0.000** |

$R^2$ / L1 / $R^2_{recal}$, w16 $\to$ w128: baseline $+0.992 \to +0.155$ / $0.0087 \to 0.0942$ /
$+0.993 \to +0.596$ (fw) and $+0.993 \to -0.073$ / $0.0077 \to 0.1108$ / $+0.994 \to +0.538$ (bd);
dup-equiv $+0.992$ / $0.0083$ / $+0.992$ (fw) and $+0.993$ / $0.0085$ / $+0.993$ (bd), mp-GMN
$+0.993$ / $0.0076$ / $+0.993$ (fw) and $+0.993$ / $0.0074$ / $+0.993$ (bd), all flat.

#### Agreement with the w16 predictions

`max |Δy|`:

| Condition | w32 | w48 | w64 | w96 | w128 |
|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 6.5e-02 | 1.1e-01 | 1.4e-01 | 2.0e-01 | 2.5e-01 |
| **ScaleGMN dup-equiv, fw** | **1.2e-07** | **6.0e-08** | **1.2e-07** | **6.0e-08** | **6.0e-08** |
| **mp-ScaleGMN dup-equiv, fw** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **8.9e-08** |
| GMN baseline, fw | 2.4e-01 | 3.3e-01 | 3.7e-01 | 4.1e-01 | 4.3e-01 |
| GMN baseline, bd | 2.1e-01 | 3.3e-01 | 4.0e-01 | 4.2e-01 | 4.3e-01 |
| **GMN dup-equiv, fw** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **8.9e-08** | **1.2e-07** |
| **GMN dup-equiv, bd** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **1.2e-07** | **1.2e-07** |
| **mp-GMN dup-equiv, fw** | **1.2e-07** | **8.9e-08** | **1.2e-07** | **8.9e-08** | **8.9e-08** |
| **mp-GMN dup-equiv, bd** | **1.2e-07** | **1.2e-07** | **6.0e-08** | **1.2e-07** | **7.5e-08** |

All six invariant rows have $\tau(y_{16}, y_w) = 1.0000$ and `mean |Δy|`
$7.0\times10^{-9}$–$1.1\times10^{-8}$ at every width — the float32 floor, from the $\approx 10^{-7}$ relative
rounding at $k = 3, 6$ plus summation reordering, not zero. The baselines:
$\tau(y_{16}, y_w)$ $0.8511 \to 0.7660$ with `mean |Δy|` $1.6\text{e-}02 \to 6.6\text{e-}02$
(ScaleGMN fw), $0.8613 \to 0.7182$ / $4.4\text{e-}02 \to 9.8\text{e-}02$ (GMN fw),
$0.8234 \to 0.4476$ / $3.7\text{e-}02 \to 1.1\text{e-}01$ (GMN bd).

### Checkpoints

All 9 trained at w16 on `w16_zoo_dup/`, selected on val $\tau$ at w16, and scored on both families.

| Condition | Checkpoint | Training run | Best val $\tau$ (w16) |
|---|---|---|---|
| ScaleGMN baseline, fw | `svhn-predgen-dup-sgmn-base_bn3rdozo` | [`bn3rdozo`](https://wandb.ai/yuxinma/svhn_predgen/runs/bn3rdozo) | 0.8722 (ep 91) |
| ScaleGMN dup-equiv, fw | `svhn-predgen-dup-sgmn-deq_3njk2izb` | [`3njk2izb`](https://wandb.ai/yuxinma/svhn_predgen/runs/3njk2izb) | 0.8736 (ep 111) |
| mp-ScaleGMN dup-equiv, fw | `svhn-predgen-dup-mpsgmn-deq_yxqd9okd` | [`yxqd9okd`](https://wandb.ai/yuxinma/svhn_predgen/runs/yxqd9okd) | 0.8741 (ep 30) |
| GMN baseline, fw | `svhn-predgen-dup-gmn-base_1y7g4ymy` | [`1y7g4ymy`](https://wandb.ai/yuxinma/svhn_predgen/runs/1y7g4ymy) | 0.8759 (ep 54) |
| GMN baseline, bd | `svhn-predgen-dup-gmn-base-bidir_2lvoifts` | [`2lvoifts`](https://wandb.ai/yuxinma/svhn_predgen/runs/2lvoifts) | 0.8782 (ep 141) |
| GMN dup-equiv, fw | `svhn-predgen-dup-gmn-deq_i0s03dz1` | [`i0s03dz1`](https://wandb.ai/yuxinma/svhn_predgen/runs/i0s03dz1) | 0.8754 (ep 158) |
| GMN dup-equiv, bd | `svhn-predgen-dup-gmn-deq-bidir_i9tffn43` | [`i9tffn43`](https://wandb.ai/yuxinma/svhn_predgen/runs/i9tffn43) | 0.8765 (ep 199) |
| mp-GMN dup-equiv, fw | `svhn-predgen-dup-mpgmn-deq_jbz6y07q` | [`jbz6y07q`](https://wandb.ai/yuxinma/svhn_predgen/runs/jbz6y07q) | 0.8761 (ep 177) |
| mp-GMN dup-equiv, bd | `svhn-predgen-dup-mpgmn-deq-bidir_3nypxpe1` | [`3nypxpe1`](https://wandb.ai/yuxinma/svhn_predgen/runs/3nypxpe1) | 0.8743 (ep 170) |

Predictions: `/tmp/predgen_svhn_widen_families/preds_{general,uniform}_{forward,bidirectional}.npz`.

### Figures

`scripts/plot_widen_families.py --dataset svhn` (reads the tables above):

| Figure | Contents |
|---|---|
| `results/figures/widen_svhn_gmn.png` | GMN baseline / dup-equiv / mp-GMN, `general` \| `uniform` panels |
| `results/figures/widen_svhn_scalegmn.png` | ScaleGMN baseline / dup-equiv / mp-ScaleGMN, same panels |

Solid forward, dashed bidirectional; colour + marker carry the model, matching every other figure in
`results/figures/`.

---

## Experiment 3 — Size Generalization on Our Multi-Width Zoo (train w16, test w16–w512)

**Status: all ten forward conditions done 2026-09-05 14:02, re-scored to w16–w512 by 16:05.** The six
bidirectional conditions and the conv matrix-product ScaleGMN are still running.

**Goal**: train the metanetwork on w16 CNNs only and predict the test accuracy of w32–w512 CNNs it has
never seen, on a zoo generated at ten widths so width is a controlled variable.

**Setup** (identical to
[CIFAR-10 Experiment 3](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512)):
train on **w16**; muP16 uses **base width 16** = the train width, so w16 muP16 $\equiv$ w16 SP exactly
(hard-linked) and the two arms differ only at the OOD widths; test widths w16–w128 ($8\times$)
extended to w192–w512 ($32\times$) by re-scoring the same checkpoints; eval batch size scales as
$bs\cdot(16/w)^2$, flooring at 1 from w256 on. Baseline = `aggregator: add`, raw edge features, no
fan-in rescaling; dup-equiv = `aggregator: mean` + fan-in rescaled edges (fan-in factor $n_{in}$, not
$n_{in}k_hk_w$). d_hid=128, 4 GNN layers, batch 64 at w16, `LastLayerReadout` in the `full_graph`
slot, `edge_pos_embed: False`. 200 epochs, 7-value LR sweep (6.25e-5 to 4e-3, $\times 2$ grid, 8600
steps), selected on val $\tau$ at w16 only.

**Conditions**: ScaleGMN and plain GMN, each in {baseline, dup-equiv}, on each of {SP, muP16} = 8,
plus the conv matrix-product GMN (dup-equiv only) on {SP, muP16} = **10 forward conditions**; the conv
matrix-product ScaleGMN (`mpsgmn`, dup-equiv, forward only) on top; and **6 bidirectional conditions**
— plain GMN and conv matrix-product GMN only.

> **⚠ The four `symmetry: scale` bidirectional conditions are out of scope and will not be run.** A
> bidirectional ScaleGMN is defined with the reciprocal backward edge feature $1/W[u,v]$; on CNN
> graphs its float32 reciprocal overflows (the smallest nonzero $|W|$ in the CIFAR-10 zoo is
> $\approx 1.2\times10^{-15}$) and training NaNs by step $\approx 13$ at every LR. The cause is
> architectural, not dataset-specific — the ScaleGMN paper (Kalogeropoulos et al., NeurIPS'24,
> arXiv:2406.10685) Appendix A.2 defines the same construction, reports the same instability for
> positive-scale symmetry, and never resolves it. See
> [`src/models/bidir_reciprocal.py`](../src/models/bidir_reciprocal.py)
> and the same verdict on
> [CIFAR-10](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#model-and-training-setup).

### Data

- **SP**: `$ANYDIM_DATA_ROOT/svhn_cnn_zoo/w{16,32,48,64,96,128}_sp/` plus `w{192,256,384,512}_sp/`.
- **muP16**: `w{16,...,512}_mup16/` — muP with base width 16, implemented in reference coordinates
  with an explicit readout multiplier $1/wm$; the exported `.pth` holds the effective weights
  $W_{eff} = W/wm$ (`src/data/train_zoo_cnns.py`).

Splits: `svhn_predgen_splits.json` per width directory
(`scripts/generate_predgen_splits.py --dataset svhn`). Generation:
`scripts/generate_cnn_zoo.py --dataset svhn` (w16–w128), then the w192–w512 extension. The tree is
disjoint from CIFAR-10's `cnn_zoo/`, so neither study can overwrite the other.

**Cost, measured on Stage 3a's own logs** (11 (arm, width) jobs, 3 GPUs, 2026-09-01 08:46–22:08):
**30.9 GPU-h** (gpu0 13.35h: sp w128 4.32h + sp w64 1.83h + sp w16 7.20h; gpu1 8.55h: mup16 w128
4.33h + mup16 w64 1.87h + sp w48 1.45h + sp w32 0.90h; gpu2 9.03h: sp w96 3.33h + mup16 w96 3.37h +
mup16 w48 1.43h + mup16 w32 0.90h) — close to the $1.47\times$-scaled CIFAR-10 projection of
$\approx$28 GPU-h this section used to carry. Disk: **5.7 GB** measured (w16–w128, SP + muP16; w16
muP16 hard-linked so free), not the 675 GB originally recorded for CIFAR-10's w16–w128 tree: that
figure predates the `export_state_dicts()` clone fix, when each saved `.pth` was a *view* into its
cohort's batch storage and `torch.save` wrote the whole storage. Post-fix widths measure exactly
$4\cdot(18c^2+21c+10)$ bytes per model.

**Stage 3d cost** (8 (arm, width) jobs, 216 cohorts, 3 GPUs, 2026-09-01 22:08 – 2026-09-03 16:25):
**124.3 GPU-h** over a 42.3 h wall-clock span, i.e. the 3 workers were busy essentially throughout —
sp w512 34.48h, mup16 w512 24.80h, mup16 w384 16.60h, sp w384 16.30h, sp w256 9.39h, mup16 w256
9.38h, mup16 w192 6.68h, sp w192 6.67h. The two arms cost the same at equal width apart from sp w512,
whose cohorts 2–4 ran $\approx 4480$ s against cohort 1's $2252$ s under contention from other users'
jobs on the same GPU; the per-cohort times are otherwise stable to $\approx 1\%$ within a job. Disk:
**83 GB** measured (w192 3.0 GB, w256 5.4 GB, w384 12 GB, w512 22 GB, each per arm), matching
$4\cdot(18c^2+21c+10)$ bytes $\times$ 1200 models (`test` 1000 + `paired` 200) per width — Stage 3d
fits only those two roles, since w16 is the only training width. Total SVHN zoo: **89 GB**.

### Reproduction

```bash
source .env

# Stage 3a (w16-w128) then, gated on its DONE sentinel, the v2 extension (w192-w512).
# Both runners use GPUS=${1:-...}, so the GPU list must ALWAYS be passed explicitly.
bash scripts/run_svhn_zoo_gen.sh "0 1 2"
bash scripts/run_svhn_zoo_gen_v2.sh "0 1 2"

# 10 conditions, forward: sweep -> 200-epoch training -> eval-only re-score, one worker per GPU
bash scripts/run_svhn_predgen_sizegen_v1.sh "<gpus>"
bash scripts/run_svhn_predgen_sizegen_v2.sh "<gpus>"

# 6 bidirectional conditions: 3 trainings -> score under muP16 -> score under SP -> w16-identity audit
bash scripts/run_svhn_predgen_sizegen_v1_bidir.sh "<gpus>"
bash scripts/run_svhn_predgen_sizegen_v2_bidir.sh "<gpus>"

# Conv matrix-product ScaleGMN: one training, scored on both width ranges and both arms
bash scripts/run_svhn_predgen_mpsgmn_sizegen.sh "<gpus>"

# zoo quality: the three tables this experiment cannot be read without
python scripts/summarize_cnn_zoo_quality.py --dataset svhn
```

**GPU budget**: at most 4 concurrent, always leaving $\ge 2$ vacant for other users. Stage 3a runs 3
workers on GPUs 0/1/2.

**Configs**: `configs/svhn_predgen/{scalegmn,gmn,mpgmn,mpsgmn}_sizegen_{sp,mup16}_{v1,v2}.yml` —
byte-identical to their `configs/cifar10_predgen/` counterparts apart from data paths, the wandb
project (`svhn_predgen`) and run-name/tag prefixes, so the two studies are comparable
apples-to-apples. `mpgmn_*` and `mpsgmn_*` must be run with `--duplication-equiv True`.
**Logs**: `/tmp/predgen_svhn_zoo/`, `/tmp/predgen_svhn_zoo_v2/`, `/tmp/predgen_svhn_sizegen_v1/`,
`/tmp/predgen_svhn_sizegen_v2/`, `/tmp/predgen_svhn_sizegen_v1_bidir/`,
`/tmp/predgen_svhn_sizegen_v2_bidir/`, `/tmp/mpsgmn_predgen_svhn/` — each an incremental `SUMMARY.md`
plus per-run sweep / full / eval logs. Queue files are created only if missing, so re-running with a
different GPU list **adds** workers rather than duplicating work.

The two `_v2*` re-score runners must not be started until every training has finished: their workers
pop a condition, and if its v1 checkpoint is not on disk yet they log `no v1 checkpoint` and drop the
job rather than requeueing it. Chain them after the trainings, never alongside.

Progress is pushed to Slack by `scripts/watch_experiments.sh` (running in `screen -S notify`), which
polls each queue directory and posts per-condition completions. Only the SVHN **zoo** and **Stage 2
dup** queues are registered there; the five size-gen queues below are not, so they report through
their own `SUMMARY.md` files only.

### Results

All **ten forward conditions** trained (2026-09-04 00:03 → 2026-09-05 14:02, `/tmp/predgen_svhn_sizegen_v1/`)
and re-scored to w16–w512 from the same checkpoints (2026-09-05 14:43 → 16:05,
`/tmp/predgen_svhn_sizegen_v2/`), `v1 agreement: OK` for all ten — each re-score reproduces its
w16–w128 $\tau$ exactly. The **conv matrix-product ScaleGMN** adds an eleventh forward cell on the same
w16–w512 ladder (2026-09-05 18:22 → 2026-09-06 03:26, `/tmp/mpsgmn_predgen_svhn/`), and the **six
bidirectional conditions** — `symmetry: permutation` only, w16–w128 — landed 2026-09-05 06:38 →
2026-09-06 00:26 (`/tmp/predgen_svhn_sizegen_v1_bidir/`) and sit in each model's own
`#### Bidirectional` subsection.

Unlike the CIFAR-10 study, each arm was trained separately rather than trained once under muP16 and
re-scored: the two arms' sweeps returned identical val $\tau$ at all 7 LRs and identical best val
$\tau$ (e.g. ScaleGMN dup-equiv $0.8026$ for both `d4ou10sf` and `pp15v7b7`), which is the
base-width identity showing up as a coincidence rather than as an audited assertion.

At a glance, ordered by mean OOD $\tau/\tau_b^{\max}$ — the width-comparable quantity here, since
the ceiling moves from $0.8494$ at w16 to $0.9629$ at w384
([above](#per-width-tau_b-ceiling--svhn-only-and-it-moves-with-width)):

| Condition | w16 $\tau$ | w512 $\tau$ | OOD mean $\tau$ | OOD mean $\tau/\tau_b^{\max}$ | $\Delta(\tau/\tau_b^{\max})$, w16→w512 | w512 $R^2$ | mean $\tau - \tau_{HP}$ |
|---|---|---|---|---|---|---|---|
| Conv matrix-product ScaleGMN, muP16 | 0.7788 | 0.8383 | 0.8332 | **0.9452** | **−0.0304** | **+0.913** | +0.180 |
| ScaleGMN, muP16, baseline | 0.7875 | 0.8276 | 0.8255 | 0.9367 | −0.0081 | +0.256 | +0.172 |
| Conv matrix-product GMN, muP16 | 0.7738 | 0.8146 | 0.8170 | 0.9270 | −0.0095 | +0.813 | +0.164 |
| ScaleGMN, muP16, dup-equiv | 0.8039 | 0.7779 | 0.8057 | 0.9144 | +0.0674 | +0.785 | +0.153 |
| Conv matrix-product ScaleGMN, SP | 0.7788 | 0.7752 | **0.8383** | 0.8964 | +0.1087 | +0.797 | **+0.280** |
| ScaleGMN, SP, baseline | 0.7875 | 0.6639 | 0.8076 | 0.8644 | +0.2350 | −0.200 | +0.249 |
| Plain GMN, muP16, dup-equiv | 0.7946 | 0.6863 | 0.7527 | 0.8549 | +0.1599 | +0.685 | +0.100 |
| Conv matrix-product GMN, SP | 0.7738 | 0.6516 | 0.7951 | 0.8513 | +0.2317 | +0.578 | +0.237 |
| ScaleGMN, SP, dup-equiv | 0.8039 | 0.4956 | 0.7326 | 0.7860 | +0.4298 | +0.375 | +0.174 |
| Plain GMN, muP16, baseline | 0.7877 | 0.6544 | 0.6879 | 0.7813 | +0.1879 | −0.362 | +0.035 |
| Plain GMN, SP, dup-equiv | 0.7946 | 0.0798 | 0.4958 | 0.5379 | +0.8523 | −0.187 | −0.063 |
| Plain GMN, SP, baseline | 0.7877 | 0.1424 | 0.4236 | 0.4605 | +0.7789 | −1.108 | −0.135 |

$\Delta$ is positive when $\tau$ fell. The last column averages $\tau - \tau_{HP}(16, w)$ over
w32–w512 against each arm's own [hyperparameter-only ceiling](#baseline-ranking-via-hyperparameters-τ_hp);
only the two plain-GMN SP conditions are negative on average, and they cross below $\tau_{HP}$ at w96
(baseline) and w128 (dup-equiv). Plain GMN muP16 baseline sits below its ceiling at w32 and w48
($-0.025$ worst) before recovering — no other condition ever drops below.

**The conv matrix-product ScaleGMN on muP16 is best on every column that is width-comparable**:
normalized OOD $\tau$ $0.9452$, the flattest ladder in the study ($\Delta = -0.0304$ — it ranks
*better* at $32\times$ than in distribution), $R^2 = +0.913$ and L1 $= 0.0263$ at w512. No other
condition is simultaneously best on ranking and on calibration: ScaleGMN muP16 baseline is second on
normalized $\tau$ but falls to $R^2 = +0.256$, and the conv matrix-product GMN on muP16 holds
$R^2 = +0.813$ at $0.9270$ normalized $\tau$.

Three conditions end $32\times$ out *above* their in-distribution normalized $\tau$ — conv
matrix-product ScaleGMN muP16 ($-0.0304$), conv matrix-product GMN muP16 ($-0.0095$) and ScaleGMN
muP16 baseline ($-0.0081$). All three are muP16; no SP condition is among them.

**One ordering inverts relative to CIFAR-10:** ScaleGMN's **baseline beats its dup-equiv variant** in
both arms on OOD $\tau$ ($0.8255$ vs $0.8057$ muP16, $0.8076$ vs $0.7326$ SP), where CIFAR-10 has
dup-equiv ahead in both ($+0.012$ muP16, $+0.169$ SP).

The raw OOD-$\tau$ column is topped by the conv matrix-product ScaleGMN's **SP** arm ($0.8383$ against
its muP16 arm's $0.8332$), and that ordering reverses once normalized ($0.8964$ vs $0.9452$): the raw
column credits SP for the ceiling's own rise. Same for the $\tau - \tau_{HP}$ column, where SP wins by
construction — its ceiling is the one that collapses to $0.3812$.

#### Normalized $\tau/\tau_b^{\max}$, per width

Every $\tau$ divided by the largest $\tau_b$ that width's own labels admit
([table](#per-width-tau_b-ceiling--svhn-only-and-it-moves-with-width)); rendered by
`python scripts/render_predgen_v2_results.py --dataset svhn --stdout`.

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 |
|---|---|---|---|---|---|---|---|---|---|---|
| Conv matrix-product ScaleGMN, muP16 | 0.9169 | 0.9303 | 0.9395 | 0.9466 | 0.9394 | 0.9481 | 0.9560 | 0.9472 | 0.9521 | 0.9474 |
| ScaleGMN, muP16, baseline | 0.9272 | 0.9455 | 0.9383 | 0.9292 | 0.9427 | 0.9331 | 0.9425 | 0.9304 | 0.9330 | 0.9353 |
| Conv matrix-product GMN, muP16 | 0.9110 | 0.9327 | 0.9201 | 0.9428 | 0.9377 | 0.9207 | 0.9323 | 0.9227 | 0.9131 | 0.9206 |
| ScaleGMN, muP16, dup-equiv | 0.9465 | 0.9390 | 0.9395 | 0.9327 | 0.9233 | 0.9181 | 0.9199 | 0.9019 | 0.8764 | 0.8791 |
| Conv matrix-product ScaleGMN, SP | 0.9169 | 0.9337 | 0.9395 | 0.9365 | 0.9276 | 0.9010 | 0.8962 | 0.8762 | 0.8486 | 0.8082 |
| ScaleGMN, SP, baseline | 0.9272 | 0.9385 | 0.9472 | 0.9316 | 0.9204 | 0.8891 | 0.8652 | 0.8317 | 0.7636 | 0.6922 |
| Plain GMN, muP16, dup-equiv | 0.9355 | 0.9286 | 0.9205 | 0.9022 | 0.8824 | 0.8655 | 0.8362 | 0.8022 | 0.7806 | 0.7756 |
| Conv matrix-product GMN, SP | 0.9110 | 0.9446 | 0.9396 | 0.9373 | 0.9013 | 0.8707 | 0.8307 | 0.8037 | 0.7549 | 0.6793 |
| ScaleGMN, SP, dup-equiv | 0.9465 | 0.9267 | 0.9434 | 0.9238 | 0.8778 | 0.8367 | 0.7652 | 0.6985 | 0.5849 | 0.5167 |
| Plain GMN, muP16, baseline | 0.9274 | 0.8777 | 0.8261 | 0.8134 | 0.7781 | 0.7744 | 0.7482 | 0.7410 | 0.7329 | 0.7395 |
| Plain GMN, SP, dup-equiv | 0.9355 | 0.9256 | 0.8929 | 0.8135 | 0.6764 | 0.5524 | 0.4156 | 0.3145 | 0.1668 | 0.0832 |
| Plain GMN, SP, baseline | 0.9274 | 0.8640 | 0.7809 | 0.7114 | 0.5625 | 0.4346 | 0.3110 | 0.2057 | 0.1485 | 0.1256 |

Three conditions hold over the whole ladder, all on muP16: the conv matrix-product ScaleGMN, which
*rises* from $0.9169$ at w16 to a $0.947$–$0.956$ plateau from w128 on; ScaleGMN baseline
($0.927$–$0.946$, no trend); and the conv matrix-product GMN ($0.911$–$0.943$). Reading the raw $\tau$ column instead
would credit the SP conditions for the ceiling's own rise: ScaleGMN SP baseline's raw $\tau$ *goes up*
from $0.7875$ at w16 to $0.8692$ at w96 while its normalized $\tau$ is already falling
($0.9272 \to 0.9204$).

Only $\tau$ carries a label-tie ceiling. $R^2$ is divided by *this width's* label variance, which is
also width-dependent ($\mathrm{sd}(y)$ w16 → w512: muP16 $0.101 \to 0.190$, SP $0.101 \to 0.219$), so
it is not width-comparable either; L1 is normalized by nothing and is the one raw cross-width number.

### Plain GMN Results (`symmetry: permutation`, forward and bidirectional)

#### Forward

**LR sweep** (val $\tau$ at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- baseline, both arms: **0.7972** / 0.7967 / 0.7954 / 0.7951 / 0.7952 / 0.7904 / 0.7925 → **6.25e-5**
- dup-equiv, both arms: **0.7932** / 0.7917 / 0.7923 / 0.7919 / 0.7927 / 0.7881 / 0.7885 → **6.25e-5**

The grid is flat to $\pm 0.007$ $\tau$ over a $64\times$ LR range and peaks at the smallest LR in both
variants, so no interior optimum was found.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 0.7877 | **0.7946** | 0.7877 | **0.7946** |
| w32 | 0.7483 | **0.8017** | 0.7433 | 0.7864 |
| w48 | 0.7099 | **0.8117** | 0.7097 | 0.7908 |
| w64 | 0.6515 | 0.7450 | 0.7159 | **0.7940** |
| w96 | 0.5312 | 0.6388 | 0.6918 | **0.7845** |
| w128 | 0.4128 | 0.5247 | 0.6786 | **0.7584** |
| w192 | 0.2976 | 0.3976 | 0.6709 | **0.7498** |
| w256 | 0.1980 | 0.3027 | 0.6648 | **0.7197** |
| w384 | 0.1209 | 0.1606 | 0.6614 | **0.7045** |
| w512 | 0.1424 | 0.0798 | 0.6544 | **0.6863** |
| **OOD mean** (w32–w512) | 0.4236 | 0.4958 | 0.6879 | **0.7527** |

Both SP columns fall below $\tau_{HP}$ — baseline from w96 ($0.5312$ against $0.5966$), dup-equiv from
w128 ($0.5247$ against $0.5913$) — and reach $0.14$ / $0.08$ at w512, indistinguishable from random
ranking. muP16 dup-equiv is the row-best at every width from w64 on, but clears its own ceiling by
only $+0.065$ at w512 ($0.6863$ against $\tau_{HP} = 0.6212$), and muP16 baseline by $+0.033$: under
muP16 the plain GMN is barely doing better at $32\times$ than a predictor that reads no weights at
all. Dup-equivariance buys $+0.072$ OOD mean under SP and $+0.065$ under muP16 — unlike CIFAR-10,
where SP dup-equiv is *worse* than SP baseline ($0.4714$ vs $0.5326$).

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.986 / 0.0066 / +0.987 | +0.983 / 0.0083 / +0.986 | +0.986 / 0.0066 / +0.987 | +0.983 / 0.0083 / +0.986 |
| w32 | +0.640 / 0.0521 / +0.939 | +0.963 / 0.0143 / +0.975 | +0.593 / 0.0493 / +0.930 | +0.956 / 0.0153 / +0.985 |
| w48 | +0.195 / 0.0970 / +0.809 | +0.908 / 0.0293 / +0.931 | +0.180 / 0.0778 / +0.830 | +0.927 / 0.0228 / +0.980 |
| w64 | −0.152 / 0.1279 / +0.602 | +0.804 / 0.0501 / +0.824 | −0.024 / 0.0996 / +0.723 | +0.901 / 0.0305 / +0.970 |
| w96 | −0.668 / 0.1714 / +0.309 | +0.609 / 0.0780 / +0.620 | −0.243 / 0.1105 / +0.641 | +0.841 / 0.0387 / +0.928 |
| w128 | −0.804 / 0.1862 / +0.164 | +0.433 / 0.1060 / +0.455 | −0.265 / 0.1149 / +0.605 | +0.822 / 0.0437 / +0.912 |
| w192 | −1.016 / 0.2157 / +0.092 | +0.267 / 0.1318 / +0.306 | −0.334 / 0.1262 / +0.593 | +0.774 / 0.0529 / +0.864 |
| w256 | −1.115 / 0.2234 / +0.056 | +0.125 / 0.1481 / +0.211 | −0.338 / 0.1274 / +0.573 | +0.734 / 0.0592 / +0.819 |
| w384 | −1.208 / 0.2444 / +0.012 | −0.029 / 0.1757 / +0.101 | −0.393 / 0.1306 / +0.553 | +0.694 / 0.0636 / +0.767 |
| w512 | −1.108 / 0.2315 / +0.016 | −0.187 / 0.1864 / +0.058 | −0.362 / 0.1285 / +0.494 | +0.685 / 0.0669 / +0.742 |

SP baseline bottoms out at $R^2 = -1.208$ (w384) with `R²_recal` $+0.012$ — the ordering itself is
gone, not merely rescaled — against CIFAR-10's $-3.983$; the difference is the denominator, since
SVHN's w512 label sd is $0.219$ against CIFAR-10's narrower spread. muP16 baseline's $R^2$ is negative
from w64 on but stalls at $\approx -0.35$, and muP16 dup-equiv is the only plain-GMN column that never
goes negative ($+0.685$ at w512).

| Condition | Checkpoint | Training run | LR | Best val τ (w16) | Eval run (w16–w128) | Eval run (w16–w512) |
|---|---|---|---|---|---|---|
| SP baseline | `svhn-predgen-sizegen-v1-gmn-sp-base_d91htiq1` | [`d91htiq1`](https://wandb.ai/yuxinma/svhn_predgen/runs/d91htiq1) | 6.25e-5 | 0.7974 (ep 100) | [`608zu1tn`](https://wandb.ai/yuxinma/svhn_predgen/runs/608zu1tn) | [`337b9rqp`](https://wandb.ai/yuxinma/svhn_predgen/runs/337b9rqp) |
| SP dup-equiv | `svhn-predgen-sizegen-v1-gmn-sp-deq_i6eyyh3u` | [`i6eyyh3u`](https://wandb.ai/yuxinma/svhn_predgen/runs/i6eyyh3u) | 6.25e-5 | 0.7932 (ep 52) | [`hii4gtl5`](https://wandb.ai/yuxinma/svhn_predgen/runs/hii4gtl5) | [`b9bcyy1c`](https://wandb.ai/yuxinma/svhn_predgen/runs/b9bcyy1c) |
| muP16 baseline | `svhn-predgen-sizegen-v1-gmn-mup16-base_9nzt62dc` | [`9nzt62dc`](https://wandb.ai/yuxinma/svhn_predgen/runs/9nzt62dc) | 6.25e-5 | 0.7974 | [`gxnbd336`](https://wandb.ai/yuxinma/svhn_predgen/runs/gxnbd336) | [`4thnlqp3`](https://wandb.ai/yuxinma/svhn_predgen/runs/4thnlqp3) |
| muP16 dup-equiv | `svhn-predgen-sizegen-v1-gmn-mup16-deq_kvxbx8zf` | [`kvxbx8zf`](https://wandb.ai/yuxinma/svhn_predgen/runs/kvxbx8zf) | 6.25e-5 | 0.7932 | [`fac2gj6m`](https://wandb.ai/yuxinma/svhn_predgen/runs/fac2gj6m) | [`gumu12sv`](https://wandb.ai/yuxinma/svhn_predgen/runs/gumu12sv) |

Predictions: `/tmp/predgen_svhn_sizegen_v{1,2}/preds_gmn-{sp,mup16}-{base,deq}.npz`.

#### Bidirectional

**LR sweep** (val $\tau$ at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- baseline, both arms: 0.7926 / 0.7923 / **0.7939** / 0.7934 / 0.7930 / 0.7870 / 0.7865 → **2.5e-4**
- dup-equiv, both arms: 0.7987 / **0.7989** / 0.7926 / 0.7924 / 0.7923 / 0.7898 / 0.7871 → **1.25e-4**

Each variant trains **once**, under the muP16 config, and is scored under both arms (base width =
train width, so w16 is byte-identical across arms); `w16 identity` audits OK for all four rows. Both
checkpoints ran all 200 epochs. Re-scored to w512 on 2026-09-12 from the same checkpoints
(`scripts/run_svhn_predgen_sizegen_v2_bidir.sh`, nothing retrained), `v1 agreement: OK` and
`w16 identity: OK` on all four rows.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | **0.7910** | 0.7851 | **0.7910** | 0.7851 |
| w32 | 0.7444 | **0.7913** | 0.7145 | 0.7745 |
| w48 | 0.7024 | **0.7772** | 0.6136 | 0.7658 |
| w64 | 0.6152 | 0.7065 | 0.5604 | **0.7539** |
| w96 | 0.3782 | 0.6138 | 0.4656 | **0.7297** |
| w128 | 0.1797 | 0.5082 | 0.4359 | **0.7070** |
| w192 | 0.0474 | 0.4126 | 0.4279 | **0.6946** |
| w256 | $-0.0241$ | 0.3343 | 0.4061 | **0.6669** |
| w384 | $-0.1127$ | 0.2355 | 0.3742 | **0.6412** |
| w512 | $-0.1500$ | 0.1718 | 0.3753 | **0.6281** |
| **OOD mean** (w32–w512) | 0.2645 | 0.5057 | 0.4859 | **0.7069** |

**Bidirectionality costs $\tau$ in five of the six conditions**, against the matched w32–w512 forward
means: plain GMN SP baseline $-0.159$ ($0.2645$ vs $0.4236$), muP16 baseline $-0.202$ ($0.4859$ vs
$0.6879$), muP16 dup-equiv $-0.046$, and SP dup-equiv is the lone gain at $+0.010$ ($0.5057$ vs
$0.4958$). On CIFAR-10 the same six matched comparisons run $-0.020$ to $+0.018$, three up and three
down — so this is an SVHN effect, not a property of the direction, and extending to $32\times$ widens
the SVHN penalty rather than closing it.

SP baseline is the only condition in the study to go **negative**: $-0.0241$ at w256 falling to
$-0.1500$ at w512, i.e. its ranking is weakly anti-correlated with the truth, where the forward arm
merely decays to $+0.1424$. Both SP columns fall below $\tau_{HP}$ (baseline from w96, $0.3782$ against
$0.5966$; dup-equiv at w128, $0.5082$ against $0.5913$) and muP16 baseline is below its own ceiling at
every OOD width ($-0.197$ at w128, $-0.246$ at w512). muP16 dup-equiv is the only row still clearing
$\tau_{HP}$ at w512 ($0.6281$ against $0.6212$), and by $+0.007$.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.985 / 0.0066 / +0.986 | +0.975 / 0.0106 / +0.982 | +0.985 / 0.0066 / +0.986 | +0.975 / 0.0106 / +0.982 |
| w32 | +0.681 / 0.0482 / +0.936 | +0.935 / 0.0218 / +0.944 | +0.586 / 0.0490 / +0.919 | +0.955 / 0.0156 / +0.962 |
| w48 | +0.225 / 0.0946 / +0.784 | +0.847 / 0.0419 / +0.869 | +0.073 / 0.0816 / +0.734 | +0.919 / 0.0240 / +0.936 |
| w64 | −0.220 / 0.1310 / +0.578 | +0.721 / 0.0652 / +0.750 | −0.214 / 0.1071 / +0.458 | +0.871 / 0.0353 / +0.891 |
| w96 | −0.793 / 0.1771 / +0.254 | +0.559 / 0.0900 / +0.590 | −0.398 / 0.1165 / +0.362 | +0.785 / 0.0465 / +0.798 |
| w128 | −0.888 / 0.1907 / +0.119 | +0.402 / 0.1171 / +0.446 | −0.396 / 0.1201 / +0.307 | +0.755 / 0.0520 / +0.766 |
| w192 | −1.064 / 0.2186 / +0.039 | +0.290 / 0.1379 / +0.329 | −0.454 / 0.1314 / +0.285 | +0.692 / 0.0629 / +0.699 |
| w256 | −1.149 / 0.2257 / +0.007 | +0.183 / 0.1514 / +0.247 | −0.440 / 0.1320 / +0.245 | +0.635 / 0.0713 / +0.638 |
| w384 | −1.238 / 0.2465 / +0.009 | +0.088 / 0.1729 / +0.156 | −0.487 / 0.1348 / +0.287 | +0.579 / 0.0782 / +0.584 |
| w512 | −1.133 / 0.2333 / +0.011 | −0.033 / 0.1820 / +0.102 | −0.451 / 0.1328 / +0.251 | +0.545 / 0.0838 / +0.554 |

muP16 dup-equiv is again the only plain-GMN column that stays positive at every width ($+0.545$ at
w512, within $0.140$ of Forward's $+0.685$), and both baselines cross zero at w64. SP baseline's
`R²_recal` reaches $+0.011$ at w512 against Forward's $+0.016$ — the ordering is gone in both
directions, not merely mis-scaled — and SP dup-equiv joins it below zero on $R^2$ at w512 ($-0.033$).
muP16 baseline is the one column whose `R²_recal` stays well above its $R^2$ at every width
($+0.251$ against $-0.451$ at w512): its ranking survives better than its calibration.

**Normalized $\tau/\tau_b^{\max}$, w16 → w512:**

| Condition | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | OOD mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SP baseline | 0.9313 | 0.8595 | 0.7727 | 0.6718 | 0.4005 | 0.1892 | 0.0495 | $-0.0250$ | $-0.1170$ | $-0.1564$ | 0.2939 |
| SP dup-equiv | 0.9243 | 0.9136 | 0.8550 | 0.7715 | 0.6499 | 0.5350 | 0.4312 | 0.3474 | 0.2446 | 0.1791 | 0.5475 |
| muP16 baseline | 0.9313 | 0.8437 | 0.7142 | 0.6367 | 0.5237 | 0.4974 | 0.4772 | 0.4527 | 0.4146 | 0.4241 | 0.5538 |
| muP16 dup-equiv | 0.9243 | 0.9146 | 0.8914 | 0.8566 | 0.8208 | 0.8068 | 0.7747 | 0.7434 | 0.7105 | 0.7098 | **0.8032** |

Normalization reorders the two SP columns against the two muP16 ones: on raw OOD $\tau$ muP16 baseline
($0.4859$) sits below SP dup-equiv ($0.5057$), but normalized it is above ($0.5538$ against $0.5475$),
because the SP labels' tie mass thins with width while muP16's does not. muP16 dup-equiv holds
$0.71$–$0.92$ over the whole $32\times$ ladder.

| Condition | Checkpoint | Training run | LR | Best val τ (w16) | Eval run (w16–w128) | Eval run (w16–w512) |
|---|---|---|---|---|---|---|
| SP baseline | `svhn-predgen-sizegen-v1-bidir-gmn-base-bidir_dcbl1fhp` | [`dcbl1fhp`](https://wandb.ai/yuxinma/svhn_predgen/runs/dcbl1fhp) | 2.5e-4 | 0.7954 | [`70fzv56b`](https://wandb.ai/yuxinma/svhn_predgen/runs/70fzv56b) | [`kbrco3x6`](https://wandb.ai/yuxinma/svhn_predgen/runs/kbrco3x6) |
| muP16 baseline | `svhn-predgen-sizegen-v1-bidir-gmn-base-bidir_dcbl1fhp` | [`dcbl1fhp`](https://wandb.ai/yuxinma/svhn_predgen/runs/dcbl1fhp) | 2.5e-4 | 0.7954 | [`q720tbc5`](https://wandb.ai/yuxinma/svhn_predgen/runs/q720tbc5) | [`iofkn08d`](https://wandb.ai/yuxinma/svhn_predgen/runs/iofkn08d) |
| SP dup-equiv | `svhn-predgen-sizegen-v1-bidir-gmn-deq-bidir_sqno0nqf` | [`sqno0nqf`](https://wandb.ai/yuxinma/svhn_predgen/runs/sqno0nqf) | 1.25e-4 | 0.7989 | [`egzigdnc`](https://wandb.ai/yuxinma/svhn_predgen/runs/egzigdnc) | [`wcuy45e6`](https://wandb.ai/yuxinma/svhn_predgen/runs/wcuy45e6) |
| muP16 dup-equiv | `svhn-predgen-sizegen-v1-bidir-gmn-deq-bidir_sqno0nqf` | [`sqno0nqf`](https://wandb.ai/yuxinma/svhn_predgen/runs/sqno0nqf) | 1.25e-4 | 0.7989 | [`tifd44ib`](https://wandb.ai/yuxinma/svhn_predgen/runs/tifd44ib) | [`q9agnwmw`](https://wandb.ai/yuxinma/svhn_predgen/runs/q9agnwmw) |

Predictions: `/tmp/predgen_svhn_sizegen_v{1,2}_bidir/preds_gmn-{base,deq}-bidir_{sp,mup16}.npz`.

### ScaleGMN Results (`symmetry: scale`, forward and bidirectional)

#### Forward

**LR sweep** (val $\tau$ at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- baseline, both arms: 0.7937 / 0.7964 / 0.7965 / **0.8006** / 0.7997 / 0.8000 / 0.7948 → **5e-4**
- dup-equiv, both arms: 0.7880 / 0.7896 / 0.7968 / 0.7908 / 0.7960 / **0.7997** / 0.7971 → **2e-3**

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 0.7875 | **0.8039** | 0.7875 | **0.8039** |
| w32 | **0.8129** | 0.8026 | 0.8007 | 0.7952 |
| w48 | **0.8610** | 0.8576 | 0.8061 | 0.8071 |
| w64 | **0.8531** | 0.8460 | 0.8178 | 0.8209 |
| w96 | **0.8692** | 0.8290 | 0.8381 | 0.8209 |
| w128 | **0.8445** | 0.7947 | 0.8177 | 0.8045 |
| w192 | 0.8278 | 0.7321 | **0.8451** | 0.8248 |
| w256 | 0.8004 | 0.6722 | **0.8347** | 0.8091 |
| w384 | 0.7353 | 0.5632 | **0.8420** | 0.7909 |
| w512 | 0.6639 | 0.4956 | **0.8276** | 0.7779 |
| **OOD mean** (w32–w512) | 0.8076 | 0.7326 | **0.8255** | 0.8057 |

Every cell clears its arm's $\tau_{HP}$, by $+0.038$ at worst (muP16 dup-equiv, w32). muP16 baseline
holds $\tau \ge 0.8007$ over the whole $32\times$ ladder and is the only condition in the study whose
$\Delta(\tau/\tau_b^{\max})$ is flat to $0.008$; **the baseline outranks dup-equiv in both arms**
($+0.020$ OOD mean under muP16, $+0.075$ under SP), the reverse of CIFAR-10, where dup-equiv leads by
$0.012$ / $0.169$. SP's raw curve peaks at w96 ($0.8692$, above its own w16 number) purely because the
ceiling rises there; normalized, it is monotone from w48 on.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP baseline | SP dup-equiv | muP16 baseline | muP16 dup-equiv |
|---|---|---|---|---|
| w16 (IN) | +0.987 / 0.0060 / +0.987 | +0.986 / 0.0064 / +0.986 | +0.987 / 0.0060 / +0.987 | +0.986 / 0.0064 / +0.986 |
| w32 | +0.932 / 0.0207 / +0.986 | +0.961 / 0.0136 / +0.983 | +0.941 / 0.0171 / +0.989 | +0.966 / 0.0115 / +0.985 |
| w48 | +0.853 / 0.0397 / +0.983 | +0.927 / 0.0247 / +0.981 | +0.882 / 0.0288 / +0.988 | +0.944 / 0.0175 / +0.985 |
| w64 | +0.775 / 0.0561 / +0.972 | +0.888 / 0.0351 / +0.971 | +0.822 / 0.0415 / +0.983 | +0.913 / 0.0254 / +0.978 |
| w96 | +0.619 / 0.0843 / +0.957 | +0.826 / 0.0523 / +0.938 | +0.725 / 0.0535 / +0.971 | +0.879 / 0.0303 / +0.965 |
| w128 | +0.491 / 0.1025 / +0.925 | +0.758 / 0.0646 / +0.891 | +0.673 / 0.0607 / +0.964 | +0.865 / 0.0333 / +0.962 |
| w192 | +0.297 / 0.1331 / +0.903 | +0.653 / 0.0885 / +0.779 | +0.554 / 0.0768 / +0.929 | +0.838 / 0.0392 / +0.950 |
| w256 | +0.142 / 0.1502 / +0.850 | +0.589 / 0.1004 / +0.686 | +0.464 / 0.0849 / +0.914 | +0.818 / 0.0426 / +0.935 |
| w384 | −0.097 / 0.1819 / +0.712 | +0.442 / 0.1294 / +0.505 | +0.286 / 0.0980 / +0.840 | +0.776 / 0.0476 / +0.897 |
| w512 | −0.200 / 0.1862 / +0.556 | +0.375 / 0.1369 / +0.406 | +0.256 / 0.0999 / +0.813 | +0.785 / 0.0489 / +0.894 |

The $\tau$ and $R^2$ orderings disagree, and this is where the baseline's OOD $\tau$ lead is paid for:
muP16 baseline's $R^2$ falls $+0.987 \to +0.256$ while `R²_recal` stays $+0.813$, so it needs a
per-width affine recalibration that dup-equiv does not ($R^2$ $+0.785$, `R²_recal − R²` $= 0.109$
against the baseline's $0.557$). SP baseline crosses zero by w384; SP dup-equiv holds $+0.375$ at w512
with $\tau$ down to $0.4956$ — the inverse pattern, calibrated but no longer ranking.

| Condition | Checkpoint | Training run | LR | Best val τ (w16) | Eval run (w16–w128) | Eval run (w16–w512) |
|---|---|---|---|---|---|---|
| SP baseline | `svhn-predgen-sizegen-v1-sgmn-sp-base_jdn3hwyz` | [`jdn3hwyz`](https://wandb.ai/yuxinma/svhn_predgen/runs/jdn3hwyz) | 5e-4 | 0.8040 (ep 165) | [`hndl5188`](https://wandb.ai/yuxinma/svhn_predgen/runs/hndl5188) | [`iyh9reia`](https://wandb.ai/yuxinma/svhn_predgen/runs/iyh9reia) |
| SP dup-equiv | `svhn-predgen-sizegen-v1-sgmn-sp-deq_d4ou10sf` | [`d4ou10sf`](https://wandb.ai/yuxinma/svhn_predgen/runs/d4ou10sf) | 2e-3 | 0.8026 (ep 180) | [`8ik3ccp5`](https://wandb.ai/yuxinma/svhn_predgen/runs/8ik3ccp5) | [`0p0jjgxm`](https://wandb.ai/yuxinma/svhn_predgen/runs/0p0jjgxm) |
| muP16 baseline | `svhn-predgen-sizegen-v1-sgmn-mup16-base_oa73am61` | [`oa73am61`](https://wandb.ai/yuxinma/svhn_predgen/runs/oa73am61) | 5e-4 | 0.8040 | [`covvnfjc`](https://wandb.ai/yuxinma/svhn_predgen/runs/covvnfjc) | [`m4i5u5pk`](https://wandb.ai/yuxinma/svhn_predgen/runs/m4i5u5pk) |
| muP16 dup-equiv | `svhn-predgen-sizegen-v1-sgmn-mup16-deq_pp15v7b7` | [`pp15v7b7`](https://wandb.ai/yuxinma/svhn_predgen/runs/pp15v7b7) | 2e-3 | 0.8026 | [`66juh2o2`](https://wandb.ai/yuxinma/svhn_predgen/runs/66juh2o2) | [`lb0t044v`](https://wandb.ai/yuxinma/svhn_predgen/runs/lb0t044v) |

Predictions: `/tmp/predgen_svhn_sizegen_v{1,2}/preds_sgmn-{sp,mup16}-{base,deq}.npz`.

#### Bidirectional

**Out of scope** — see the ⚠ warning in
[the setup above](#experiment-3--size-generalization-on-our-multi-width-zoo-train-w16-test-w16w512).
No run exists and none will.

### Conv Matrix-Product GMN Results (`message_fn_type: matrix_product_conv`, forward and bidirectional)

Dup-equiv only — the MSG function is a sum of matrix products only on top of fan-in rescaling + mean
aggregation, so there is no baseline variant. Layer, installer and property checks are the CIFAR-10
study's ([its section](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#conv-matrix-product-gmn-results-message_fn_type-matrix_product_conv-forward-and-bidirectional)),
unchanged.

#### Forward

**LR sweep** (val $\tau$, both arms): 0.7660 / 0.7659 / 0.7682 / 0.7838 / 0.7833 / **0.7854** /
0.7794 → **2e-3**.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | 0.7738 | 0.7738 |
| w32 | **0.8181** | 0.7899 |
| w48 | **0.8541** | 0.7905 |
| w64 | **0.8583** | 0.8298 |
| w96 | **0.8512** | 0.8337 |
| w128 | **0.8270** | 0.8068 |
| w192 | 0.7948 | **0.8359** |
| w256 | 0.7735 | **0.8278** |
| w384 | 0.7269 | **0.8240** |
| w512 | 0.6516 | **0.8146** |
| **OOD mean** (w32–w512) | 0.7951 | **0.8170** |

muP16's $\tau$ *rises* $0.7738$ at w16 → $0.8146$ at w512 and its normalized $\tau$ is flat
($0.911 \to 0.921$, $\Delta = -0.0095$, the flattest in the study), against SP's $0.911 \to 0.679$.
The arms cross at w192: SP leads through w128 and loses $0.163$ from there to w512. Both clear
$\tau_{HP}$ at every width, SP by $+0.237$ on average and muP16 by $+0.164$.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | +0.984 / 0.0074 / +0.984 | +0.984 / 0.0074 / +0.984 |
| w32 | +0.980 / 0.0105 / +0.985 | +0.977 / 0.0103 / +0.986 |
| w48 | +0.961 / 0.0181 / +0.980 | +0.963 / 0.0150 / +0.988 |
| w64 | +0.950 / 0.0235 / +0.978 | +0.944 / 0.0209 / +0.985 |
| w96 | +0.906 / 0.0335 / +0.943 | +0.915 / 0.0254 / +0.973 |
| w128 | +0.873 / 0.0420 / +0.927 | +0.905 / 0.0275 / +0.972 |
| w192 | +0.810 / 0.0568 / +0.882 | +0.886 / 0.0324 / +0.967 |
| w256 | +0.772 / 0.0638 / +0.858 | +0.875 / 0.0342 / +0.964 |
| w384 | +0.700 / 0.0808 / +0.823 | +0.811 / 0.0398 / +0.913 |
| w512 | +0.578 / 0.0915 / +0.717 | +0.813 / 0.0400 / +0.916 |

muP16 holds $R^2 \ge +0.811$ and `R²_recal − R²` $\le 0.105$ at every width, and L1 stays at
$0.0400$ accuracy units at w512 — the smallest of any condition, $2.5\times$ better than the ScaleGMN
muP16 baseline's $0.0999$. This is the only SVHN condition that never gives up either metric. SP holds
$R^2 \ge +0.578$, i.e. it is the best SP column on $R^2$ too, but its $\tau$ decays.

| Condition | Checkpoint | Training run | LR | Best val τ (w16) | Eval run (w16–w128) | Eval run (w16–w512) |
|---|---|---|---|---|---|---|
| SP | `svhn-predgen-sizegen-v1-mpgmn-sp-deq_ilb86d48` | [`ilb86d48`](https://wandb.ai/yuxinma/svhn_predgen/runs/ilb86d48) | 2e-3 | 0.7934 (ep 93) | [`5unw733h`](https://wandb.ai/yuxinma/svhn_predgen/runs/5unw733h) | [`ezv0lphf`](https://wandb.ai/yuxinma/svhn_predgen/runs/ezv0lphf) |
| muP16 | `svhn-predgen-sizegen-v1-mpgmn-mup16-deq_kgf5fl3f` | [`kgf5fl3f`](https://wandb.ai/yuxinma/svhn_predgen/runs/kgf5fl3f) | 2e-3 | 0.7934 | [`z04p62gs`](https://wandb.ai/yuxinma/svhn_predgen/runs/z04p62gs) | [`0h96aopj`](https://wandb.ai/yuxinma/svhn_predgen/runs/0h96aopj) |

Predictions: `/tmp/predgen_svhn_sizegen_v{1,2}/preds_mpgmn-{sp,mup16}-deq.npz`.

#### Bidirectional

**LR sweep** (val $\tau$, both arms): 0.7749 / 0.7779 / 0.7836 / **0.7911** / 0.7841 / 0.7857 /
0.7790 → **5e-4**. One training under the muP16 config, scored under both arms, `w16 identity: OK` for
both rows, all 200 epochs run. Re-scored to w512 on 2026-09-12 from the same checkpoint, `v1
agreement: OK` on both rows.

**Per-width test Kendall $\tau$, w16 → w512** (row-best in bold):

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | 0.7657 | 0.7657 |
| w32 | **0.8061** | 0.7882 |
| w48 | **0.8477** | 0.7904 |
| w64 | **0.8457** | 0.8119 |
| w96 | **0.8388** | 0.8310 |
| w128 | 0.7992 | **0.8127** |
| w192 | 0.7601 | **0.8343** |
| w256 | 0.7127 | **0.8215** |
| w384 | 0.6510 | **0.8269** |
| w512 | 0.6316 | **0.8101** |
| **OOD mean** (w32–w512) | 0.7659 | **0.8141** |

This is the **least damaged by bidirectionality** of the six over the full ladder: $-0.003$ on muP16
($0.8141$ against Forward's $0.8170$) and $-0.029$ on SP ($0.7659$ against $0.7951$), where the plain
GMN loses up to $0.202$. Both arms clear $\tau_{HP}$ at every width, SP by $+0.207$ on average and
muP16 by $+0.176$. muP16 is above its in-distribution $\tau$ at every OOD width and peaks at w192
($0.8343$), not at the train width; SP is above it through w192 and falls below from w256.

Normalized, muP16 is flat ($0.9015 \to 0.9155$, $\Delta = -0.0140$, OOD mean $0.9237$) while SP falls
($0.9015 \to 0.6585$, OOD mean $0.8206$). The arms cross at w128 on raw $\tau$ and at w96 once
normalized — the same crossover the forward arm shows at w192, moved earlier by the direction.

**$R^2$ / L1 / $R^2_{recal}$**, same row order:

| Width | SP | muP16 |
|---|---|---|
| w16 (IN) | +0.983 / 0.0077 / +0.984 | +0.983 / 0.0077 / +0.984 |
| w32 | +0.977 / 0.0113 / +0.981 | +0.978 / 0.0099 / +0.986 |
| w48 | +0.957 / 0.0196 / +0.974 | +0.963 / 0.0145 / +0.986 |
| w64 | +0.927 / 0.0280 / +0.958 | +0.943 / 0.0205 / +0.984 |
| w96 | +0.884 / 0.0406 / +0.933 | +0.907 / 0.0254 / +0.969 |
| w128 | +0.813 / 0.0559 / +0.882 | +0.890 / 0.0284 / +0.965 |
| w192 | +0.735 / 0.0753 / +0.835 | +0.867 / 0.0337 / +0.960 |
| w256 | +0.652 / 0.0879 / +0.760 | +0.843 / 0.0363 / +0.946 |
| w384 | +0.557 / 0.1102 / +0.688 | +0.807 / 0.0393 / +0.921 |
| w512 | +0.506 / 0.1172 / +0.634 | +0.799 / 0.0405 / +0.918 |

muP16 holds $R^2 \ge +0.799$ and `R²_recal − R²` $\le 0.119$ across the whole ladder — within $0.014$
of Forward's muP16 column at every width, and its L1 stays at $0.0405$ accuracy units at w512. This is
the only bidirectional condition here whose calibration is intact at $32\times$; SP keeps $R^2$
positive throughout ($+0.506$ at w512) but at $2.9\times$ the L1.

| Condition | Checkpoint | Training run | LR | Best val τ (w16) | Eval run (w16–w128) | Eval run (w16–w512) |
|---|---|---|---|---|---|---|
| SP | `svhn-predgen-sizegen-v1-bidir-mpgmn-deq-bidir_bfeiloq3` | [`bfeiloq3`](https://wandb.ai/yuxinma/svhn_predgen/runs/bfeiloq3) | 5e-4 | 0.7911 | [`wlvcg8w4`](https://wandb.ai/yuxinma/svhn_predgen/runs/wlvcg8w4) | [`1dyjbm0o`](https://wandb.ai/yuxinma/svhn_predgen/runs/1dyjbm0o) |
| muP16 | `svhn-predgen-sizegen-v1-bidir-mpgmn-deq-bidir_bfeiloq3` | [`bfeiloq3`](https://wandb.ai/yuxinma/svhn_predgen/runs/bfeiloq3) | 5e-4 | 0.7911 | [`pwe2z3aq`](https://wandb.ai/yuxinma/svhn_predgen/runs/pwe2z3aq) | [`8rqxs8ox`](https://wandb.ai/yuxinma/svhn_predgen/runs/8rqxs8ox) |

Predictions: `/tmp/predgen_svhn_sizegen_v{1,2}_bidir/preds_mpgmn-deq-bidir_{sp,mup16}.npz`.

### Conv Matrix-Product ScaleGMN Results (`message_fn_type: matrix_product_conv_scale`, forward)

The bilinear-per-offset MSG constraint on **ScaleGMN's** scale-equivariant node states and node update
instead of the plain GMN's concat-MLP. Dup-equiv only and forward only; identical LR grid, epochs,
splits, readout (`LastLayerReadout`) and selection metric (val $\tau$ at w16) to the conv matrix-product
GMN above, so these are drop-in columns in the same per-width tables. Layer, installer and property
checks are the CIFAR-10 study's
([its section](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward)),
unchanged.

**Date**: 2026-09-05 18:22 → 2026-09-06 03:26, 9.1 h on one GPU
**Configs**: `configs/svhn_predgen/mpsgmn_sizegen_{sp,mup16}_{v1,v2}.yml`, with
`--duplication-equiv True --direction forward`
**Logs**: `/tmp/mpsgmn_predgen_svhn/`

One training under the muP16 v1 config, scored four ways — {SP, muP16} × {v1 w16–w128, v2 w16–w512} —
`w16 identity: OK` on all four, and each v2 row at w16–w128 reproduces its v1 row to the last digit in
both arms.

**LR sweep** (val $\tau$ at w16, 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
0.7798 / 0.7811 / 0.7825 / 0.7800 / 0.7879 / 0.7739 / **0.7929** → **4e-3**. Then 200 epochs at 4e-3,
best val $\tau$ **0.7929**, all 200 run. The grid peaks at its largest value, so no interior optimum
was found — the same boundary hit the plain GMN's forward sweep has at the *smallest* LR. In
distribution nothing separates this model from its two references on the same grid (conv matrix-product
GMN $0.7854$, dup-equiv ScaleGMN $0.7997$); the whole difference is OOD.

| Width | muP16 τ | muP16 R² | muP16 L1 | muP16 R²_recal | SP τ | SP R² | SP L1 | SP R²_recal |
|---|---|---|---|---|---|---|---|---|
| w16 (IN) | 0.7788 | +0.977 | 0.0088 | +0.979 | 0.7788 | +0.977 | 0.0088 | +0.979 |
| w32 | 0.7878 | +0.979 | 0.0095 | +0.981 | 0.8087 | +0.980 | 0.0108 | +0.982 |
| w48 | 0.8071 | +0.975 | 0.0118 | +0.985 | 0.8540 | +0.970 | 0.0160 | +0.980 |
| w64 | 0.8331 | +0.963 | 0.0167 | +0.983 | 0.8576 | +0.960 | 0.0208 | +0.976 |
| w96 | 0.8352 | +0.951 | 0.0185 | +0.979 | **0.8760** | +0.942 | 0.0267 | +0.966 |
| w128 | 0.8308 | +0.946 | 0.0199 | +0.979 | 0.8558 | +0.912 | 0.0340 | +0.944 |
| w192 | 0.8572 | +0.936 | 0.0228 | +0.978 | 0.8575 | +0.900 | 0.0402 | +0.944 |
| w256 | 0.8497 | +0.931 | 0.0238 | +0.975 | 0.8432 | +0.880 | 0.0437 | +0.925 |
| w384 | 0.8592 | +0.918 | 0.0252 | +0.967 | 0.8171 | +0.859 | 0.0512 | +0.914 |
| w512 | 0.8383 | +0.913 | 0.0263 | +0.965 | 0.7752 | +0.797 | 0.0562 | +0.850 |
| **OOD mean w32–w128 (v1)** | 0.8188 | | | | **0.8504** | | | |
| **OOD mean w32–w512 (v2)** | 0.8332 | | | | **0.8383** | | | |

**Normalized $\tau/\tau_b^{\max}$:**

| Arm | w16 | w32 | w48 | w64 | w96 | w128 | w192 | w256 | w384 | w512 | OOD mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| muP16 | 0.9169 | 0.9303 | 0.9395 | 0.9466 | 0.9394 | 0.9481 | 0.9560 | 0.9472 | 0.9521 | 0.9474 | **0.9452** |
| SP | 0.9169 | 0.9337 | 0.9395 | 0.9365 | 0.9276 | 0.9010 | 0.8962 | 0.8762 | 0.8486 | 0.8082 | 0.8964 |

On muP16 this is the **best condition in the study**: normalized OOD $\tau$ $0.9452$ against the
previous best $0.9367$ (ScaleGMN muP16 baseline), $\Delta(\tau/\tau_b^{\max}) = -0.0304$ — it ranks
better at $32\times$ than at the width it trained on, and its worst normalized $\tau$ over the whole
ladder is the in-distribution one. $R^2$ holds $+0.913$ at w512 with `R²_recal − R²` $\le 0.052$
everywhere, so no per-width recalibration is needed; against $\tau_{HP}(16, 512) = 0.6212$ it clears
the hyperparameter-only ceiling by $0.217$. SP raw $\tau$ peaks at w96 ($0.8760$, the largest single
$\tau$ in the study) but normalized it is monotone from w48, ending at $0.8082$.

The same model on CIFAR-10 reaches $0.9219$ OOD $\tau$ (w32–w512) on muP16 and is also best there, so
it ranks first on both zoos. Its margin over the best non-`mpsgmn` condition is far narrower here:
$+0.008$ raw ($0.8332$ against ScaleGMN muP16 baseline's $0.8255$) against $+0.025$ on CIFAR-10
($0.9219$ against the conv matrix-product GMN's $0.8971$), where the near-unit ceiling makes raw and
normalized $\tau$ interchangeable.

| Condition | Eval run | `w16 identity` |
|---|---|---|
| muP16, v1 (w16–w128) | [`zv856hlb`](https://wandb.ai/yuxinma/svhn_predgen/runs/zv856hlb) | OK |
| SP, v1 (w16–w128) | [`x0wntryu`](https://wandb.ai/yuxinma/svhn_predgen/runs/x0wntryu) | OK |
| muP16, v2 (w16–w512) | [`aglwcbas`](https://wandb.ai/yuxinma/svhn_predgen/runs/aglwcbas) | OK |
| SP, v2 (w16–w512) | [`7lla3u6s`](https://wandb.ai/yuxinma/svhn_predgen/runs/7lla3u6s) | OK |

Training run [`190tbetr`](https://wandb.ai/yuxinma/svhn_predgen/runs/190tbetr), checkpoint
`svhn-predgen-sizegen-mpsgmn_190tbetr`. Predictions:
`/tmp/mpsgmn_predgen_svhn/preds_mpsgmn-deq_{v1,v2}_{sp,mup16}.npz`.

### $\tau$ within Hyperparameter Strata

Diagnostic 4, computed 2026-09-12 by `scripts/summarize_predgen_hp_strata.py --dataset svhn` from the
`preds_*.npz` dumps of the runs already recorded above (no new training, no new wandb runs). Method,
stratifications and the positional-join audit are
[CIFAR-10's](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#tau-within-hyperparameter-strata),
unchanged: `lr` = $10$ learning-rate deciles ($100$ CNNs each, $9.9\%$ of pairs), `hp` = exact
$(\text{dropout}, \text{epochs}, \text{train\_frac})$ crossed with LR terciles ($108$ strata,
$\approx 9$ CNNs each, $0.92\%$ of pairs). Read against
[`τ_HP`](#baseline-ranking-via-hyperparameters-τ_hp), which on this zoo bottoms out at $0.3812$ (SP)
and $0.6212$ (muP16).

Pooling is $\sum_s (C_s - D_s)$ over $\sum_s \sqrt{(n_{0s} - n_{1s})(n_{0s} - n_{2s})}$, so the tie
mass that gives this study its [moving $\tau_b$ ceiling](#per-width-tau_b-ceiling--svhn-only-and-it-moves-with-width)
is handled per stratum exactly as the unstratified $\tau_b$ handles it.

OOD columns are means over w32–w512, except `mpsgmn-deq-v1` (w32–w128). The four `symmetry: scale`
bidirectional conditions are [out of scope](#experiment-status) and not reported.

| Condition | Arm | $\tau$ (w16) | OOD $\tau$ | OOD $\tau$ (`lr`) | $\Delta$ | OOD $\tau$ (`hp`) | $\Delta$ |
|---|---|---|---|---|---|---|---|
| gmn-base | mup16 | $+0.7877$ | $+0.6879$ | $+0.5742$ | $-0.1137$ | $+0.4930$ | $-0.1949$ |
| gmn-base | sp | $+0.7877$ | $+0.4236$ | $+0.4055$ | $-0.0182$ | $+0.2214$ | $-0.2022$ |
| gmn-deq | mup16 | $+0.7946$ | $+0.7527$ | $+0.7198$ | $-0.0329$ | $+0.7241$ | $-0.0286$ |
| gmn-deq | sp | $+0.7946$ | $+0.4958$ | $+0.6201$ | $+0.1243$ | $+0.5194$ | $+0.0236$ |
| sgmn-base | mup16 | $+0.7875$ | $+0.8255$ | $+0.7906$ | $-0.0349$ | $+0.7783$ | $-0.0472$ |
| sgmn-base | sp | $+0.7875$ | $+0.8076$ | $+0.7905$ | $-0.0171$ | $+0.7113$ | $-0.0963$ |
| sgmn-deq | mup16 | $+0.8039$ | $+0.8057$ | $+0.7728$ | $-0.0329$ | $+0.7616$ | $-0.0441$ |
| sgmn-deq | sp | $+0.8039$ | $+0.7326$ | $+0.7766$ | $+0.0440$ | $+0.6733$ | $-0.0593$ |
| mpgmn-deq | mup16 | $+0.7738$ | $+0.8170$ | $+0.7832$ | $-0.0338$ | $+0.7644$ | $-0.0526$ |
| mpgmn-deq | sp | $+0.7738$ | $+0.7951$ | $+0.7705$ | $-0.0246$ | $+0.6876$ | $-0.1074$ |
| gmn-base-bidir | mup16 | $+0.7910$ | $+0.4860$ | $+0.4508$ | $-0.0351$ | $+0.3587$ | $-0.1273$ |
| gmn-base-bidir | sp | $+0.7910$ | $+0.2645$ | $+0.4570$ | $+0.1925$ | $+0.2677$ | $+0.0032$ |
| gmn-deq-bidir | mup16 | $+0.7851$ | $+0.7069$ | $+0.6815$ | $-0.0253$ | $+0.6996$ | $-0.0073$ |
| gmn-deq-bidir | sp | $+0.7851$ | $+0.5057$ | $+0.5817$ | $+0.0760$ | $+0.5140$ | $+0.0083$ |
| mpgmn-deq-bidir | mup16 | $+0.7657$ | $+0.8141$ | $+0.7775$ | $-0.0366$ | $+0.7543$ | $-0.0598$ |
| mpgmn-deq-bidir | sp | $+0.7657$ | $+0.7659$ | $+0.7387$ | $-0.0272$ | $+0.6042$ | $-0.1616$ |
| mpsgmn-deq-v1 | mup16 | $+0.7788$ | $+0.8188$ | $+0.7854$ | $-0.0335$ | $+0.7707$ | $-0.0481$ |
| mpsgmn-deq-v1 | sp | $+0.7788$ | $+0.8504$ | $+0.8230$ | $-0.0275$ | $+0.7728$ | $-0.0776$ |
| mpsgmn-deq-v2 | mup16 | $+0.7788$ | $+0.8332$ | $+0.7977$ | $-0.0355$ | $+0.7833$ | $-0.0498$ |
| mpsgmn-deq-v2 | sp | $+0.7788$ | $+0.8383$ | $+0.8043$ | $-0.0340$ | $+0.7407$ | $-0.0976$ |

No condition depends on the hyperparameter shortcut. Fourteen of twenty lose under `lr`, but only
`gmn-base — mup16` loses more than $0.05$ ($-0.1137$), and it is the weakest muP16 condition to begin
with. The two matrix-product families and both ScaleGMN variants land in a tight $-0.017$ to $-0.037$
band on both arms, so their OOD $\tau$ is not recipe-reading.

Four SP conditions **gain**, largest `gmn-base-bidir` $+0.1925$ and `gmn-deq` $+0.1243$ — the same
pattern CIFAR-10 shows and for the same reason: within a fixed LR band the ordering survives, while
across bands the predictions stop reproducing the accuracy gaps. `gmn-base-bidir — sp` is the sharpest
case, since its unstratified OOD $\tau$ is $+0.2645$ with per-width values that go
[negative from w256](#bidirectional), yet its within-decile $\tau$ is $+0.4570$; the anti-correlation
is a between-recipe artifact, not a reversed ranking inside any recipe.

`gmn-base` is the exception on the SP arm: $-0.0182$ under `lr` and $-0.2022$ under `hp`, the largest
`hp` loss in the study, ending at $+0.2214$. Together with `gmn-base — mup16`'s $-0.1949$ this makes
plain GMN baseline the one architecture whose OOD ranking is substantially recipe-carried on SVHN.

Under `hp` the picture is *not* uniformly worse than under `lr`, unlike CIFAR-10: three conditions
improve slightly (`gmn-deq — sp` $+0.0236$, `gmn-deq-bidir — sp` $+0.0083$, `gmn-base-bidir — sp`
$+0.0032$) and `gmn-deq-bidir — mup16` is flat ($-0.0073$), because on this zoo holding `epochs` and
`train_frac` fixed removes variation that was hurting those columns. The study's best conditions are
barely touched: `mpsgmn-deq-v2 — mup16` keeps $+0.7833$ of $+0.8332$ ($94\%$) with the recipe held
exactly fixed and $\approx 9$ comparable CNNs per stratum, and `mpgmn-deq — mup16` $+0.7644$ of
$+0.8170$.

### Figures

`python scripts/plot_sizegen_results.py` reads the per-width tables above and writes
`results/figures/sizegen_svhn_{tau,r2}_{mup,sp}_{gmn,scalegmn}_{fw,bd}.png` — 12 figures,
the ScaleGMN-bidirectional pair excluded as out of scope. Three series per panel (baseline /
dup-equiv / matrix-product) with the SP or muP16 $\tau_{HP}$ ceiling as the dotted grey reference on
the $\tau$ panels and $R^2 = 0$ on the $R^2$ panels. The two `scalegmn_fw` panels carry the conv
matrix-product ScaleGMN as their third series; the `gmn_fw` panels carry the conv matrix-product GMN.
All twelve panels now span w16–w512 — the four bidirectional ones were on a shorter x-axis until the
w512 re-score landed 2026-09-12.

---

## Mandatory Diagnostics

Same list as [CIFAR-10's](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md#mandatory-diagnostics) — a
per-width $\tau$ table alone does not identify what OOD size generalization means, because the width
axis moves the CNNs' accuracies, their weight norms and the hyperparameter→accuracy map at once.

1. **Per-width accuracy distributions** — quantiles and support overlap with w16. **Done 2026-09-03**
   for w16–w512: the median w16 CNN is collapsed and `frac inside w16's support` falls to $0.652$ at
   w384 (SP) against muP16's flat $0.847$–$0.880$, so diagnostic 1 fires for SP past w128 exactly as
   on CIFAR-10. Table: [Input-network zoo quality](#per-width-accuracy-distributions-labels).
2. **Baseline: ranking via hyperparameters** — `τ_HP(16, w)` on the width-independent `paired` role.
   **Done 2026-09-03** for w32–w512: $0.708 \to 0.381$ (SP) and $0.758 \to 0.621$ (muP16), with
   muP16 above SP at every width — no crossover, unlike CIFAR-10.
   Table: [Input-network zoo quality](#baseline-ranking-via-hyperparameters-τ_hp).
3. **Per-width $\tau$ / $R^2$ / L1 / $R^2_{recal}$** for every condition. **Done for all seventeen on
   w16–w512** — ten forward and the conv matrix-product ScaleGMN (2026-09-05/06), six bidirectional
   (2026-09-06 to w128, extended 2026-09-12) — inlined in the per-model tables above. `R²_recal` vs `R²` separates a
   collapsed ordering from a drifted scale, and the two disagree sharply here: ScaleGMN muP16 baseline
   is second on OOD $\tau$ but $R^2 = +0.256$ at w512 with `R²_recal` $= +0.813$, so its scale drifted
   while its ordering held; plain GMN SP baseline has both gone ($R^2 = -1.108$, `R²_recal` $= +0.016$,
   $\tau = 0.1424$). The conv matrix-product ScaleGMN on muP16 is the only condition where neither
   moves (`R²_recal − R²` $\le 0.052$ at every width).
4. **$\tau$ within fixed-LR HP strata** — $\tau$ computed inside groups that share the sampled
   hyperparameters, which removes the HP signal and leaves only what the weights carry. **Done
   2026-09-12 for all 20 in-scope conditions**, on w32–w512 in both directions: no condition depends
   on the shortcut. Both matrix-product families and both ScaleGMN variants lose only $0.017$–$0.037$
   under LR-decile strata, and the four SP conditions that fall below `τ_HP` **gain** (up to
   $+0.193$), so their unstratified collapse is a compressed prediction range across recipes rather
   than lost within-recipe ordering. The exception is plain GMN baseline, which loses $0.19$–$0.20$ on
   both arms under the exact-recipe strata — the one architecture here whose OOD ranking is
   substantially recipe-carried. Table:
   [$\tau$ within hyperparameter strata](#tau-within-hyperparameter-strata).
5. **Per-width normalized spectral norms** $\|w\|_{Vn}$ and `W₁(μ₁₆, μ_w)`. **Done 2026-09-03** for
   w16–w512: SP grows $8.03\times$ over $32\times$ width (against $\sqrt{32} = 5.66$), muP16 holds to
   $+5.2\%$ with `W₁ ≤ 3.2`.
   Table: [Input-network zoo quality](#per-width-normalized-spectral-norms).
6. **wandb run IDs for every training and eval run reported.** Done for Experiments 1 and 2; for all
   ten Experiment 3 forward conditions, each with its training run, its w16–w128 eval run and its
   w16–w512 re-score eval run, the last audited against the first by `v1 agreement: OK`; for the six
   bidirectional conditions (3 training runs, 12 eval runs — 6 to w128 and 6 to w512, `w16 identity:
   OK` and `v1 agreement: OK` on all six); and for the conv matrix-product ScaleGMN (1 training run,
   4 eval runs).
7. **SVHN-only: the $\tau_b$ ceiling.** $\tau_b^{\max} = 0.9031$ from the $18.4\%$ tied label pairs
   ([above](#upstream-zoo-experiment-1s-data)), so cross-dataset $\tau$ comparisons must be made as
   fractions of each dataset's ceiling. **Recomputed per width 2026-09-03**, and it is *not* constant:
   $0.8494$ at the training width rising to $0.9629$ at w384 under SP, against $0.9925$–$0.9997$ (flat)
   on CIFAR-10. Table: [Input-network zoo quality](#per-width-tau_b-ceiling--svhn-only-and-it-moves-with-width).
   **Applied to every condition 2026-09-05/06, and to the bidirectional arm's full w16–w512 ladder
   2026-09-12**:
   [Normalized $\tau/\tau_b^{\max}$, per width](#normalized-tautau_bmax-per-width) for the eleven
   forward cells, and inline in each `#### Bidirectional` subsection. It changes conclusions rather
   than decorating them — ScaleGMN SP baseline's raw $\tau$ rises from $0.7875$ (w16) to $0.8692$ (w96)
   while its normalized $\tau$ is already falling, and the study's best condition swaps arm under
   normalization: the conv matrix-product ScaleGMN's SP arm leads on raw OOD $\tau$ ($0.8383$ vs
   $0.8332$) and its muP16 arm leads by $0.049$ once normalized.

---

## Experiment Status

| Experiment | What | State |
|---|---|---|
| 0 | GMN properties on CNN graphs | **done** (architecture-only, dataset-independent) — [VERIFICATION_gmn_properties.md § CNN graphs](VERIFICATION_gmn_properties.md#on-cnn-inputs). `check_mup_cnn.py --only identity` and `check_cnn_widening_equiv.py` to be re-run on real SVHN w16 weights — Stage 3a landed 2026-09-01, not yet re-run |
| 1 | Reproduce accuracy prediction at w16 | **done** 2026-09-01 — [table above](#results): plain GMN $\tau = 0.8672$, ScaleGMN $0.8655$, both $\approx 96\%$ of the $0.9031$ ceiling |
| 2 | Widening sanity check, 2 families $\times$ 9 conditions | **done** 2026-09-12 19:23 — [tables above](#experiment-2--size-generalization-on-wider-equivalent-cnns-sanity-check): on `uniform` all 6 dup-equiv/mp conditions are flat at every width (`max \|Δy\| ≤ 1.2e-7`) and the 3 baselines drift $-0.066$ to $-0.403$; on `general` only the 2 forward matrix-product conditions are flat |
| 3a | Zoo generation w16–w128, muP CNN, splits | **done** 2026-09-01 22:08 — 11 (arm, width) jobs on 3 workers (GPUs 0/1/2); 30.9 GPU-h, 5.7 GB measured (see [Data](#data) above). `DONE` sentinel chained straight into Stage 3d. `/tmp/predgen_svhn_zoo/` |
| 3b | Size-gen training, 10 forward conditions | **done** 2026-09-05 14:02 (launched 2026-09-04 00:03) — started on one worker on GPU 2, the only free GPU at the time, grown to 3 as GPUs freed via `scripts/add_workers_when_free_pool.sh`. All ten checkpoints at 200 epochs; [results above](#results-1). `/tmp/predgen_svhn_sizegen_v1/` |
| 3c | This document | **done for everything measured** — Experiment 1 / 2 tables, the full [zoo-quality section](#input-network-zoo-quality) (w16–w512, 2026-09-03), the per-width $\tau_b$-ceiling finding, and every condition's $\tau$ / $R^2$ / L1 / $R^2_{recal}$ and normalized-$\tau$ tables: all seventeen conditions on w16–w512, and (2026-09-12) all seven mandatory diagnostics, the last being [diagnostic 4](#tau-within-hyperparameter-strata) |
| 3d | Zoo extension w192–w512, both arms | **done** 2026-09-03 16:25 — 8 (arm, width) jobs / 216 cohorts on 3 workers (GPUs 0/1/2); 124.3 GPU-h, 83 GB measured (see [Data](#data) above). All 10 widths' split JSONs rewritten with `disjointness: OK`; `finalize()`'s md5 check reports **no change** to any pre-existing w16–w128 split. `/tmp/predgen_svhn_zoo_v2/` |
| 3e | Forward re-score to w16–w512 | **done** 2026-09-05 16:05 — all ten v1 checkpoints re-scored on the w16–w512 ladder, `v1 agreement: OK` for all ten (w16–w128 $\tau$ reproduces exactly). Nothing retrained. `/tmp/predgen_svhn_sizegen_v2/` |
| 3f | Bidirectional arm, 6 conditions | **done** 2026-09-06 00:26 (launched 2026-09-05 06:38, 17.8 h) — 3 workers, 3 trainings for 6 conditions (each trained once under muP16, scored under both arms), `w16 identity: OK` ×6, all at 200 epochs. The 4 `symmetry: scale` conditions are **out of scope** — see the ⚠ warning above. **Extended to w512 2026-09-12** — the same six rows re-scored from their existing checkpoints (`scripts/run_svhn_predgen_sizegen_v2_bidir.sh`, no retraining), 34 min wall on three GPUs, `v1 agreement: OK` and `w16 identity: OK` for all six. Result: bidirectionality costs $\tau$ in five of six over the full w32–w512 ladder ($-0.202$ to $+0.010$), a wider penalty than at $8\times$, and plain GMN SP baseline goes **negative** past w256 ($-0.1500$ at w512). `/tmp/predgen_svhn_sizegen_v{1,2}_bidir/` |
| 5 | Conv matrix-product ScaleGMN | **done** 2026-09-06 03:26 (started 2026-09-05 18:22, 9.1 h on one GPU) — one training at lr 4e-3, scored 4 ways ({SP, muP16} × {v1 w16–w128, v2 w16–w512}), `w16 identity: OK` ×4 and each v2 row reproducing its v1 row exactly. **Best condition in the study on muP16** — [section above](#conv-matrix-product-scalegmn-results-message_fn_type-matrix_product_conv_scale-forward). `/tmp/mpsgmn_predgen_svhn/` |

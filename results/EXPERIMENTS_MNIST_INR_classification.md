# Experiments — MNIST INR Classification

## Contents
- [Norm statistics verification](#norm-statistics-verification-section-3-theory)
- [INR reconstruction quality](#inr-reconstruction-quality)
- [Experiment 1 — Reproduce ScaleGMN Table 1](#experiment-1--reproduce-scalegmn-table-1-mnist-inr-classification) (fixed width w32)
- [Experiment 2 — Size generalization on wider equivalent networks](#experiment-2--size-generalization-on-wider-equivalent-networks-sanity-check)
  (sanity check; `general` vs `uniform` widening $\times$ 11 conditions)
- [Experiment 3 — Size generalization](#experiment-3--size-generalization-train-w24-test-w24w1024)
  (train w24, test w24–w1024) — the main size-generalization result
    - [Plain GMN results](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional)
    - [Scale GMN results](#scalegmn-results-symmetry-scale-forward-and-bidirectional)
    - [Matrix Product GMN results](#matrix-product-gmn-results-message_fn_type-matrix_product-forward--bidirectional)
    - [Matrix Produdct ScaleGMN results](#matrix-product-scalegmn-results-message_fn_type-matrix_product_scale-forward)
- [Why ScaleGMN generalizes relatively well without duplication equivalence](#why-scalegmn-baseline-generalizes-relatively-well-without-duplication-equivariance)
- [Historical notes (deprecated)](#historical-notes-deprecated), incl.
  [size-gen v2](#size-generalization-v2-deprecated--train-w16-test-w24w96) (train w16, test w24–w96)

---

## Norm Statistics Verification (Section 3 theory)

**Goal**: Numerically verify the theoretical predictions about $\|w\|_{V_n}$ scaling for INR weights:
- Standard parameterization (SP) at initialization: $\|w\|_{V_n} = \Theta(\sqrt{n})$, dominated by last layer.
- SP after training the INRs until convergence: not the same pattern as initialization.
- muP at initialization and after training the INRs (`MuBatchINRFitter` in `src/data/fit_inrs.py`).
  - the same training dynamics across widths
  - very similar norms across widths

Measurements use the per-width-LR SP fits (`w{N}_sp/`) and muP with **base_width=24**
(`w{N}_mup24/`), widths w24–w256. Earlier w16-inclusive / base_width=32 measurements are in
[Historical notes](#historical-notes-deprecated).

**Date**: 2026-08-18

### SP at initialization (Kaiming uniform, `[2, w, w, 1]`, 500 samples)

**Script**: `scripts/verify_norm_stats.py --widths 24 32 48 64 80 96 128 192 256 --num-samples 500`

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 24 | 4.53 ± 0.26 | 0.63 | 1.07 | 2.83 | 0.624 |
| 32 | 4.95 ± 0.26 | 0.63 | 1.09 | 3.24 | 0.654 |
| 48 | 5.72 ± 0.25 | 0.62 | 1.10 | 3.99 | 0.698 |
| 64 | 6.33 ± 0.27 | 0.61 | 1.11 | 4.61 | 0.728 |
| 80 | 6.89 ± 0.26 | 0.61 | 1.12 | 5.16 | 0.749 |
| 96 | 7.41 ± 0.26 | 0.61 | 1.13 | 5.68 | 0.766 |
| 128 | 8.26 ± 0.25 | 0.61 | 1.13 | 6.52 | 0.790 |
| 192 | 9.74 ± 0.26 | 0.60 | 1.14 | 8.00 | 0.822 |
| 256 | 10.97 ± 0.26 | 0.60 | 1.14 | 9.23 | 0.842 |

### SP trained ReLU INRs (per-width fitting LR, 500 samples per width)

**Script**: `scripts/verify_norm_stats.py` (reads the fitted INRs in `w{N}_sp/`)

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 24 | 43.26 ± 4.64 | 2.47 | 13.62 | 27.17 | 0.628 |
| 32 | 46.22 ± 4.83 | 2.27 | 14.14 | 29.81 | 0.645 |
| 48 | 54.38 ± 6.56 | 2.07 | 15.10 | 37.21 | 0.684 |
| 64 | 55.60 ± 6.19 | 1.92 | 14.77 | 38.91 | 0.700 |
| 80 | 58.90 ± 6.71 | 1.84 | 15.07 | 41.99 | 0.713 |
| 96 | 62.83 ± 7.50 | 1.77 | 15.25 | 45.81 | 0.729 |
| 128 | 64.12 ± 7.09 | 1.69 | 14.78 | 47.65 | 0.743 |
| 192 | 66.11 ± 5.52 | 1.60 | 14.30 | 50.21 | 0.760 |
| 256 | 72.18 ± 6.37 | 1.56 | 14.27 | 56.35 | 0.781 |

### muP at initialization (base_width=24, Kaiming He init + $\sqrt{w_m}$ output scaling, 500 samples)

**Script**: `scripts/verify_norm_stats_mup.py --base-width 24 --widths 24 ... 256`

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 24 | 11.04 ± 1.03 | 1.56 | 2.67 | 6.81 | 0.617 |
| 32 | 11.11 ± 0.88 | 1.55 | 2.70 | 6.87 | 0.618 |
| 48 | 11.11 ± 0.73 | 1.52 | 2.73 | 6.85 | 0.617 |
| 64 | 11.18 ± 0.63 | 1.51 | 2.75 | 6.91 | 0.619 |
| 80 | 11.17 ± 0.55 | 1.51 | 2.76 | 6.91 | 0.618 |
| 96 | 11.17 ± 0.53 | 1.49 | 2.77 | 6.91 | 0.618 |
| 128 | 11.18 ± 0.43 | 1.48 | 2.78 | 6.92 | 0.619 |
| 192 | 11.19 ± 0.36 | 1.47 | 2.79 | 6.92 | 0.619 |
| 256 | 11.17 ± 0.29 | 1.47 | 2.80 | 6.90 | 0.618 |

### muP trained ReLU INRs (base_width=24, 1000 steps, lr=0.01, 500 samples per width)

**Script**: `scripts/verify_norm_stats_mup.py --base-width 24 --trained-from-disk mup24`
(reads the fitted INRs in `w{N}_mup24/`)

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 24 | 43.36 ± 4.41 | 2.47 | 13.69 | 27.20 | 0.627 |
| 32 | 42.83 ± 4.11 | 2.48 | 13.56 | 26.78 | 0.625 |
| 48 | 42.46 ± 4.01 | 2.50 | 13.29 | 26.67 | 0.628 |
| 64 | 42.11 ± 3.63 | 2.50 | 13.15 | 26.46 | 0.628 |
| 80 | 41.98 ± 3.52 | 2.52 | 12.97 | 26.49 | 0.631 |
| 96 | 41.93 ± 3.55 | 2.52 | 12.84 | 26.57 | 0.634 |
| 128 | 41.77 ± 3.35 | 2.53 | 12.61 | 26.63 | 0.638 |
| 192 | 41.53 ± 3.12 | 2.53 | 12.19 | 26.81 | 0.646 |
| 256 | 41.25 ± 3.02 | 2.54 | 11.89 | 26.83 | 0.650 |

---

## INR Reconstruction Quality

Reconstruction MSE/PSNR between fitted INRs and original MNIST images, evaluated on 1000 random
training samples per width. Architecture `[2, w, w, 1]`, 1000 steps of Adam.
"Fail%" = fraction with $\text{MSE} > 0.01$ (essentially unconverged fits).
Both tables below use **w24 as the base/training width**; w16-based fits and the
`base_width=32` muP variant are in [Historical notes](#historical-notes-deprecated).

### Standard-Param ReLU INRs (per-width fitting LR)

The `w{N}_sp/` datasets on disk were fitted with a per-width LR sweep (`scripts/sweep_inr_lr.py`,
500 samples, 1000 steps, sweep batch size matched to the fitting run). One rule is applied
identically at every width: **minimize failure rate ($\text{MSE} > 0.01$), ties broken on higher mean PSNR**
(literally `sort(key=(fail_rate, -psnr_mean))`) — a failed fit is a qualitatively broken input the
metanetwork still has to classify, which matters more downstream than PSNR differences among fits
that converged. (`--max-fail-rate` offers a PSNR-first alternative; it was never used for any dataset
here.) Candidate grids were [0.002, 0.005, 0.01, 0.02] for w24–w256 and
[0.0005, 0.001, 0.002, 0.005] for w384–w1024.

| Width | LR | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|---|
| 24 | 0.01 | 30.2 dB | 30.6 dB | 3.4 | 0.7% |
| 32 | 0.01 | 31.9 dB | 32.4 dB | 3.3 | 0.3% |
| 48 | 0.02 | 34.5 dB | 34.7 dB | 3.3 | 0.2% |
| 64 | 0.01 | 34.8 dB | 35.0 dB | 3.2 | 0.2% |
| 80 | 0.01 | 35.4 dB | 36.0 dB | 3.5 | 0.4% |
| 96 | 0.01 | 35.9 dB | 36.1 dB | 3.4 | 0.5% |
| 128 | 0.005 | 35.0 dB | 35.4 dB | 3.4 | 0.4% |
| 192 | 0.002 | 32.6 dB | 33.0 dB | 3.2 | 0.7% |
| 256 | 0.002 | 33.4 dB | 33.9 dB | 3.4 | 1.1% |
| 384 | 0.001 | 31.1 dB | 31.5 dB | 3.1 | 1.1% |
| 512 | 0.001 | 31.7 dB | 32.1 dB | 3.3 | 1.0% |
| 768 | 0.0005 | 29.3 dB | 29.9 dB | 3.2 | 1.0% |
| 1024 | 0.0005 | 29.5 dB | 30.5 dB | 4.3 | 3.2% |

Five widths selected a value at a grid boundary (w48, w192, w256, w768, w1024) and were re-swept past
it; every original selection survived, so **all 13 LRs are interior optima of the union grid**
[1.25e-4 … 0.08] and no width is under-tuned or needed a refit (failure rate per candidate, selected
value in bold):

- w48 @ 0.01 / 0.02 / 0.04 / 0.08 → 0.2% / **0.0%** / 2.2% / 74.2%
- w192 @ 2.5e-4 / 5e-4 / 1e-3 / 2e-3 → 79.8% / 24.0% / 0.2% / **0.0%**
- w256 @ 2.5e-4 / 5e-4 / 1e-3 / 2e-3 → 69.4% / 10.6% / 0.4% / **0.2%**
- w768 @ 1.25e-4 / 2.5e-4 / 5e-4 → 69.6% / 12.8% / **1.0%**
- w1024 @ 1.25e-4 / 2.5e-4 / 5e-4 → 61.4% / 5.2% / **2.6%**

(Sweep numbers are on 500 freshly-fitted samples and differ slightly from the
`verify_inr_quality.py` numbers in the table above, which score the actual on-disk INRs from a
different random init — e.g. 0.0% vs 0.7% at w192, well inside the ±0.45pp sampling noise.)

### muP ReLU INRs (base_width=24)

Used in the size generalization experiment. Same Kaiming He init as SP, muP output-layer scaling
with base_width=24, **fixed lr=0.01 at every width — no per-width sweep**. This is the point of muP:
`get_mup_optimizer` already rescales per-layer LRs by `base_width/width`, so the *base* LR transfers
across widths and the optimum is stable rather than shrinking as it does under SP. We therefore reuse
the w24 (= base width) value of 0.01 at every width from w24 to w1024. The differing criteria are the
point of the comparison: holding the base LR fixed is what makes muP the controlled arm — sweeping it
per width would assume away the property being tested.

| Width | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|
| 24 | 30.2 dB | 30.5 dB | 3.24 | 0.3% |
| 32 | 31.3 dB | 31.6 dB | 3.23 | 0.1% |
| 48 | 32.2 dB | 32.5 dB | 2.93 | 0.3% |
| 64 | 32.8 dB | 33.0 dB | 2.87 | 0.0% |
| 80 | 33.0 dB | 33.3 dB | 2.84 | 0.0% |
| 96 | 33.0 dB | 33.4 dB | 2.90 | 0.3% |
| 128 | 33.2 dB | 33.5 dB | 2.92 | 0.0% |
| 192 | 33.1 dB | 33.4 dB | 2.78 | 0.1% |
| 256 | 33.0 dB | 33.3 dB | 2.77 | 0.2% |
| 384 | 32.8 dB | 33.1 dB | 2.7 | 0.0% |
| 512 | 32.5 dB | 32.8 dB | 2.6 | 0.0% |
| 768 | 32.8 dB | 33.1 dB | 2.7 | 0.0% |
| 1024 | 32.5 dB | 33.0 dB | 2.7 | 0.0% |

---

## Experiment 1 — Reproduce ScaleGMN Table 1 (MNIST-INR classification)

**TL;DR**: ReLU INRs with `symmetry: scale` achieve **97.5% test accuracy** against the paper's
96.8% on DWS SIREN INRs.

### Reproduction

```bash
source .env

# ScaleGMN (forward + bidirectional)
screen -dmS scalegmn_fw bash -c 'source .env && CUDA_VISIBLE_DEVICES=0 \
  python scripts/train_scalegmn.py --conf configs/mnist_cls/scalegmn_reproduce_sp.yml --wandb True'
screen -dmS scalegmn_bidir bash -c 'source .env && CUDA_VISIBLE_DEVICES=1 \
  python scripts/train_scalegmn.py --conf configs/mnist_cls/scalegmn_reproduce_sp_bidir.yml --wandb True'

# Plain GMN (forward + bidirectional)
screen -dmS gmn_fw bash -c 'source .env && CUDA_VISIBLE_DEVICES=2 \
  python scripts/train_scalegmn.py --conf configs/mnist_cls/gmn_reproduce_sp.yml --wandb True'
screen -dmS gmn_bidir bash -c 'source .env && CUDA_VISIBLE_DEVICES=3 \
  python scripts/train_scalegmn.py --conf configs/mnist_cls/gmn_reproduce_sp_bidir.yml --wandb True'
```

### Reference — DWS SIREN INRs (from paper)

**Dataset**: DWS MNIST-INRs downloaded from Dropbox. Phase-canonicalized. Split: 55k/5k/10k.
**Date**: 2026-05-12

| Variant | Best test acc | Run ID |
|---|---|---|
| ScaleGMN (forward, sign) | 0.9679 | [`w1vs7otg`](https://wandb.ai/yuxinma/mnist_cls/runs/w1vs7otg) |
| ScaleGMN-B (bidirectional, sign) | 0.9644 | [`2qxitoil`](https://wandb.ai/yuxinma/mnist_cls/runs/2qxitoil) |

### ReLU INRs (our data)

**Dataset**: Standard-param ReLU INRs (`scripts/generate_inrs.py --init-type sp`).
Architecture `[2, 32, 32, 1]`, split: 55k/5k/10k. PSNR=32.1 dB, 0.1% failure rate.

**Config**: `configs/mnist_cls/scalegmn_reproduce_sp.yml` (`symmetry: scale`)
**Date**: 2026-05-18

| Variant | Best val acc | Best test acc | Run ID |
|---|---|---|---|
| ScaleGMN (forward, scale) | 0.9752 | 0.9752 | [`lyfxngw0`](https://wandb.ai/yuxinma/mnist_cls/runs/lyfxngw0) |
| ScaleGMN-B (bidirectional, scale) | 0.9716 | 0.9710 | [`86h7jk36`](https://wandb.ai/yuxinma/mnist_cls/runs/86h7jk36) |

The ScaleGMN-B row carries the reciprocal backward edge feature a bidirectional scale model is
defined with (`configs/mnist_cls/scalegmn_reproduce_sp_bidir.yml`, `reciprocal: True`; LR $10^{-3}$,
200 epochs, patience $=50$, no early stop). **Date**: 2026-09-02. The superseded
[`1hzzwv2s`](https://wandb.ai/yuxinma/mnist_cls/runs/1hzzwv2s) run (0.9746 / 0.9734) omitted it and
is not scale-equivariant — see the
[Experiment 3 ScaleGMN block](#scalegmn-results-symmetry-scale-forward-and-bidirectional) for the
full statement.

### Plain GMN (no scale equivariance)

**Dataset**: Same SP ReLU INRs as above (w32, 55k/5k/10k).
**Config**: `configs/mnist_cls/gmn_reproduce_sp.yml` (`symmetry: permutation`)
**Date**: 2026-06-25

| Variant | Best val acc | Best test acc | Run ID |
|---|---|---|---|
| GMN (forward, permutation) | 0.9594 | 0.9606 | [`34n125s5`](https://wandb.ai/yuxinma/mnist_cls/runs/34n125s5) |
| GMN-B (bidirectional, permutation) | 0.9604 | 0.9598 | [`x0g99ffx`](https://wandb.ai/yuxinma/mnist_cls/runs/x0g99ffx) |

---

## Experiment 2 — Size generalization on wider equivalent networks (sanity check)

**Goal**: Take the w24 test INRs, widen them to w48, w96, w144, w192, w240 by a *function-preserving*
construction, and measure size generalization. Because the widened INR computes exactly the same
function as its w24 source, every accuracy change is the metanetwork failing to be invariant to the
widening — never distribution shift in the underlying images. Compare against
[Experiment 3](#experiment-3--size-generalization-train-w24-test-w24w1024), where the wider inputs are
fitted from random initialization.

Two widening families are scored, both defined in
[VERIFICATION_gmn_properties.md](VERIFICATION_gmn_properties.md) § Equivalence classes. Writing the
widened weight blockwise, $W^{(l)}_{\mathrm{up}}[i k_l + a,\, j k_{l-1} + b] = B^{(l)}_{ij}[a, b]$ with
$b^{(l)}_{\mathrm{up}} = b^{(l)} \otimes \mathbf{1}_{k_l}$ and $k_0 = k_L = 1$:

| Family | $B^{(l)}_{ij}$ | Conditions |
|---|---|---|
| `general` | independent random block per $(i,j)$, **row** condition only | $B^{(l)}_{ij}\mathbf{1}_{k_{l-1}} = W^{(l)}_{ij}\mathbf{1}_{k_l}$ |
| `uniform` | $W^{(l)}_{ij}\,\mathbf{1}\mathbf{1}^{\mathsf{T}}/k_{l-1}$ (Kronecker) | row **and** $\mathbf{1}_{k_l}^{\mathsf{T}}B^{(l)}_{ij} = (k_l/k_{l-1})W^{(l)}_{ij}\mathbf{1}_{k_{l-1}}^{\mathsf{T}}$ |

The row condition alone preserves the function, so `general` is the **largest** function-preserving
widening and `uniform` is a measure-zero special case of it. `uniform` is the reference: it is
the family the dup-equiv modifications (fan-in rescaling + mean aggregation + `LayerWiseMeanReadout`)
are built for, so reading a `general` number against it separates "not invariant to this family" from
"cannot handle a wide graph at all".

**11 conditions per parameterization arm**: ScaleGMN and plain GMN each contribute
$\{\text{baseline}, \text{dup-equiv}\} \times \{\text{forward}, \text{bidirectional}\}$; mp-GMN
contributes dup-equiv $\times$ both directions; mp-ScaleGMN contributes dup-equiv forward. The
matrix-product models have no baseline variant (the MSG constraint is defined on top of fan-in
rescaling + mean aggregation), and **mp-ScaleGMN is forward-only in every experiment**, so its
bidirectional cell is out of scope rather than missing. All 22 checkpoints are the Experiment 3 ones;
bidirectional ScaleGMN uses the `-recip` retrains.

**No SP / muP distinction here**, so only the muP24 arm is tabulated. The parameterization only affects
how INRs of *different* widths are fitted, and no wide INR is ever fitted in this experiment: at
`width == base_width` muP reduces exactly to SP ($w_m = 1$, so both the per-layer LR rescaling and the
$\sqrt{w_m}$ output scaling are the identity), and both w24 draws were in fact fitted at lr $=0.01$ to
the same PSNR mean of $30.2$ dB. The SP arm is therefore a **replicate**, differing only in the INR
fitting RNG and in the metanetwork checkpoint trained on that draw: it was scored in the same batch
([`svu1p79c`](https://wandb.ai/yuxinma/mnist_cls/runs/svu1p79c),
[`ldu38w77`](https://wandb.ai/yuxinma/mnist_cls/runs/ldu38w77) for `general` forward/bidirectional;
[`vvrkcngc`](https://wandb.ai/yuxinma/mnist_cls/runs/vvrkcngc),
[`9bok7al9`](https://wandb.ai/yuxinma/mnist_cls/runs/9bok7al9) for `uniform`) and agrees to $0.0$pp on
every invariant cell and to $\le 8.6$pp elsewhere.

**Function preservation, measured**: `generate_duplicated_inrs.py --verify 25` reports, over w48–w240
and both arms, $\max|\Delta\text{pixels}| \le 3.2\times10^{-6}$ and
$\max|\Delta\text{logits}| \le 2.5\times10^{-5}$ against the w24 source — the same order as the
`uniform` control ($5.9\times10^{-7}$ / $5.1\times10^{-6}$), i.e. pure float32 storage rounding. Blocks
are drawn in float64 and cast on write; the float32 cancellation problem that forces float64 for *conv*
general widenings does not arise for MLP INRs. Row residual $\le 4.8\times10^{-7}$; column residual
$4.05$–$14.0$ and minimum non-uniformity $2.2$–$3.8$ confirm `general` genuinely violates the column
condition and never degenerates to the uniform block.

**Scripts**: `scripts/generate_duplicated_inrs.py --family` (data),
`scripts/eval_sizegen_duplicated.py` (eval), `scripts/run_mnist_cls_widen_families_eval.sh`
(orchestration).

**Date**: 2026-09-12 (families $\times$ 11 conditions; supersedes the 2026-08-18 forward-only
uniform-Kronecker run). The w16-base run is in
[Historical notes](#deprecated-experiment-2-w16-base-size-gen-v2-checkpoints).

### Reproduction

```bash
source .env

# Data: widen the w24 test INRs under each family (~1.0 GB per family per arm)
for INIT in mup24 sp; do for FAM in general uniform; do
  python scripts/generate_duplicated_inrs.py --init-type $INIT --base-width 24 \
      --target-widths 48 96 144 192 240 --family $FAM --seed 0 --verify 25
done; done

# Eval: 8 jobs = {general,uniform} x {mup24,sp} x {forward,bidirectional}, one worker per GPU
screen -dmS widen_eval bash -c 'bash scripts/run_mnist_cls_widen_families_eval.sh "0 1"'
```

Eval batch size is scaled as `bs · (24/w)²` to keep memory flat out to w240.

### `general` widening — the largest function-preserving family

Eval runs [`nti3p8bn`](https://wandb.ai/yuxinma/mnist_cls/runs/nti3p8bn) (forward) and
[`0c7ajfoy`](https://wandb.ai/yuxinma/mnist_cls/runs/0c7ajfoy) (bidirectional), extended
2026-09-15 to w384/w768 (superseding the earlier w48–w240 run of the same name).

#### ScaleGMN

| Condition | w24 | w48 | w96 | w144 | w192 | w240 | w384 | w768 | max_drop |
|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 95.5% | 86.4% | 51.1% | 41.5% | 36.4% | 33.9% | 28.1% | 24.6% | 70.8% |
| ScaleGMN baseline, bd | 95.8% | 79.5% | 53.1% | 44.5% | 39.5% | 35.4% | 25.1% | 15.6% | 80.2% |
| ScaleGMN dup-equiv, fw | 96.4% | 90.9% | 51.9% | 33.3% | 26.2% | 23.2% | 20.7% | 19.4% | 77.0% |
| ScaleGMN dup-equiv, bd | 96.6% | 87.2% | 46.1% | 32.6% | 26.9% | 24.4% | 22.4% | 21.5% | 75.1% |
| **mp-ScaleGMN dup-equiv, fw** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **0.0%** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | out of scope |

#### Plain GMN

| Condition | w24 | w48 | w96 | w144 | w192 | w240 | w384 | w768 | max_drop |
|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | 93.0% | 78.0% | 53.0% | 48.1% | 46.3% | 45.5% | 44.1% | 43.3% | 49.7% |
| GMN baseline, bd | 93.7% | 73.2% | 31.7% | 25.6% | 23.6% | 22.1% | 18.3% | 13.8% | 79.9% |
| GMN dup-equiv, fw | 94.5% | 45.2% | 18.4% | 14.9% | 14.0% | 13.9% | 13.8% | 14.4% | 80.8% |
| GMN dup-equiv, bd | 93.8% | 34.0% | 10.2% | 10.0% | 10.0% | 10.0% | 10.0% | 9.9% | 84.0% |
| **mp-GMN dup-equiv, fw** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **0.0%** |
| mp-GMN dup-equiv, bd | 95.0% | 91.5% | 56.6% | 37.0% | 27.2% | 22.4% | 17.4% | 14.6% | 80.3% |

### `uniform` (Kronecker) widening — the reference family

Eval runs [`kcsmnh2w`](https://wandb.ai/yuxinma/mnist_cls/runs/kcsmnh2w) (forward) and
[`7u700r73`](https://wandb.ai/yuxinma/mnist_cls/runs/7u700r73) (bidirectional), extended
2026-09-15 to w384/w768 (superseding the earlier w48–w240 run of the same name).

#### ScaleGMN

| Condition | w24 | w48 | w96 | w144 | w192 | w240 | w384 | w768 | max_drop |
|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 95.5% | 95.4% | 94.0% | 92.8% | 92.0% | 90.8% | 89.1% | 86.0% | 9.5% |
| ScaleGMN baseline, bd | 95.8% | 93.8% | 86.9% | 69.8% | 46.9% | 29.3% | 14.6% | 10.8% | 84.9% |
| **ScaleGMN dup-equiv, fw** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **0.0%** |
| **ScaleGMN dup-equiv, bd** | **96.6%** | **96.6%** | **96.6%** | **96.6%** | **96.6%** | **96.6%** | **96.6%** | **96.6%** | **0.0%** |
| **mp-ScaleGMN dup-equiv, fw** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **96.4%** | **0.0%** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | out of scope |

#### Plain GMN

| Condition | w24 | w48 | w96 | w144 | w192 | w240 | w384 | w768 | max_drop |
|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | 93.0% | 78.5% | 43.4% | 34.8% | 32.2% | 30.6% | 29.6% | 28.8% | 64.2% |
| GMN baseline, bd | 93.7% | 55.9% | 19.6% | 17.4% | 16.9% | 16.4% | 16.0% | 15.4% | 78.3% |
| **GMN dup-equiv, fw** | **94.5%** | **94.5%** | **94.5%** | **94.5%** | **94.5%** | **94.5%** | **94.5%** | **94.5%** | **0.0%** |
| **GMN dup-equiv, bd** | **93.8%** | **93.8%** | **93.8%** | **93.8%** | **93.8%** | **93.8%** | **93.8%** | **93.8%** | **0.0%** |
| **mp-GMN dup-equiv, fw** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **0.0%** |
| **mp-GMN dup-equiv, bd** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **95.0%** | **0.0%** |

The forward rows reproduce the 2026-08-18 run they supersede (`letjkx8z`, `jy3vx4sc`) to within
$0.1$pp — the only differences are ScaleGMN baseline w192 ($92.0$% vs $92.1$%) and the resulting
max_drop ($4.6$% vs $4.7$%), $2/2000$ samples, TF32 kernel-selection noise under
`set_float32_matmul_precision("high")`.

### Figures

`scripts/plot_widen_families.py` (reads the tables above):

| Figure | Contents |
|---|---|
| `results/figures/widen_gmn.png` | GMN baseline / dup-equiv / mp-GMN, `general` \| `uniform` panels |
| `results/figures/widen_scalegmn.png` | ScaleGMN baseline / dup-equiv / mp-ScaleGMN, same panels |

Solid forward, dashed bidirectional; colour + marker carry the model, matching every other figure in
`results/figures/`.

---

## Experiment 3 — Size Generalization (train w24, test w24–w1024)

**Goal**: Measure how well a metanetwork trained at a single INR width transfers to unseen, wider
INRs, and how much of that transfer is attributable to (i) the parameterization used to fit the
INRs and (ii) the modifications of the metanetwork.

**Setup**:
- Train on **w24**: a sufficiently small width whose INR fits are of good quality, so that OOD degradation cannot be blamed on a poorly-fitted training set.
- muP uses **`base_width=24`** = the training width, so that w24 muP $\equiv$ w24 SP exactly and the
  two data conditions differ only at the OOD widths.
- **Every test set uses disjoint images**, both from each other and from train/val, so no image is
  ever seen at two widths. Architecture `[2, w, w, 1]` throughout.

### Data

- SP ReLU: `w{N}_sp/` (Kaiming He init, per-width swept fitting LR, 1000 steps)
- muP24: `w{N}_mup24/` (base_width=24, Kaiming He init, muP optimizer, fixed lr=0.01, 1000 steps)

The test-width range was built in two stages, which is only a data-provenance detail — all
w24 → w1024 numbers below come from one evaluation protocol:

1. **w24–w256** (`mnist_sizegen_splits.json`, seed 123): shuffle all 70,000 MNIST images and
   partition into 51k train + 1k val at w24, plus 2k disjoint test images per width. This consumed
   the entire dataset.
2. **w384–w1024** (`mnist_sizegen_splits_v4.json`, seed 20260819, `scripts/extend_sizegen_splits.py`):
   because stage 1 left no unused images, the four new widths were made room for by pooling the nine
   2k test sets (18,000 images) and re-allocating **1k per width across all 13 widths**. Train and
   val are copied **byte-identical**, so the checkpoints trained in stage 1 remain valid — none of
   their training images appears in any test set. The script asserts strict disjointness and that
   both parameterizations agree on the train/val assignment.

Because train/val never changed, **no model was retrained** to add w384–w1024: the same checkpoints
were re-scored with `train_sizegen.py --eval-only-ckpt`.

**Only the evaluated INRs are fitted at w384–w1024.** A full 70k-image dataset at w1024 would be
~280 GB to support a 1k-image test set, so `generate_inrs.py --only-from-split` fits exactly the 1000
INRs each width's test set names — 15.8 GB for all 8 new directories. The split JSON is written
*first* and acts as the fitting manifest. Filenames are unaffected: `canonical_file_ids()` derives
them from a full pass over the dataset in its fixed order, so
`{prefix}_png_{train,test}/{digit}/{index:06d}.pth` denotes the same source image whether that width
was fitted in full or as a subset (verified by rendering existing INRs against the mapping).

> **Caveat for reading the SP columns.** Even at its selected LR, SP fit quality declines past w256
> (33.4 dB → 29.3/29.5 dB at w768/w1024) and w1024 fails on 3.2% of fits. Part of any SP accuracy
> drop at the largest widths is the INRs getting worse, not the metanetwork failing to generalize.
> muP24 is flat at 32.5–32.8 dB with **0.0% failures** at every width past w256, so it is the
> controlled comparison.

### Model and training setup

**Model variants**: ScaleGMN (`symmetry: scale`) and plain GMN (`symmetry: permutation`), each in
{baseline, dup-equiv}.
Baseline = `aggregator: add`, raw edge weights, sum-pooling readout.
Dup-equiv = `aggregator: mean`, fan-in rescaled edges ($d_{l-1} \cdot W^{(l)}$), `LayerWiseMeanReadout`.
All variants: d_hid=128, 4 GNN layers, `readout_range: full_graph`.
Both message-passing directions were run: the forward variant (`ScaleGMN_GNN_fw`) in
[ScaleGMN (forward)](#scalegmn-results-symmetry-scale-forward-and-bidirectional) /
[Plain GMN (forward)](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional), and the bidirectional
variant (`ScaleGMN_GNN_bidir`) in
[ScaleGMN (bidirectional)](#scalegmn-results-symmetry-scale-forward-and-bidirectional) /
[Plain GMN (bidirectional)](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional) — same eight
conditions, same protocol, reported as separate subsections rather than extra columns.

The **matrix-product GMN** (`mpgmn`, dup-equiv only) was added to this experiment in both
directions: $\{\text{SP}, \text{muP24}\} \times \{\text{forward}, \text{bidirectional}\}$, same protocol and grid — see
[Matrix-Product GMN Results](#matrix-product-gmn-results-message_fn_type-matrix_product-forward--bidirectional)
below.

**Training**: batch_size=64, AdamW, warmup 1000 steps, 200 epochs, patience=50.
Eval batch size scales as `bs * (train_w² / test_w²)` to prevent OOM at large widths (batch 1 by
w768/w1024).

**Metanetwork LR protocol**: a 7-value sweep (6.25e-5 to 4e-3, $\times 2$ grid, 8600 steps) per condition,
selected on **best val accuracy at w24**, then full training at that LR. Note this is selected
in-distribution only — no OOD width influences model selection, which is what keeps the OOD numbers
honest.

**Date**: muP conditions 2026-07-10, SP conditions 2026-07-11 (after the SP INR refit), matrix-product
GMN 2026-08-19, w384–w1024 evaluation 2026-08-19, bidirectional plain GMN 2026-08-26,
**bidirectional ScaleGMN 2026-09-01** (all ten conditions of this experiment are measured in both
directions — see [Forward vs bidirectional](#forward-vs-bidirectional--all-ten-conditions))

### Reproduction

```bash
source .env

# Data + training for w24-w256:
bash scripts/run_mnist_cls_sizegen_v3.sh          # full pipeline
bash scripts/run_mnist_cls_sizegen_v3_final.sh    # training only (data already generated)
bash scripts/run_mnist_cls_mpgmn_sizegen_v3.sh    # matrix-product GMN, 4 conditions

# Matrix-product ScaleGMN, 2 conditions (sweep -> full training -> v4 re-score each).
# An empty GPU list seeds the queue without starting a worker, for when the GPU budget is
# full; re-run it with a GPU to add a worker.
bash scripts/run_mnist_cls_mpsgmn_sizegen_v3.sh ""

# Bidirectional ScaleGMN + plain GMN (8 conditions; sweep -> full training -> v4 re-score
# per condition, one worker per GPU, resumable — re-run it to add a worker):
bash scripts/run_mnist_cls_sizegen_v3_bidir.sh "0 2"

# Extending to w384-w1024 (splits FIRST — they are the fitting manifest):
python scripts/extend_sizegen_splits.py --dry-run
python scripts/extend_sizegen_splits.py
bash scripts/run_mnist_cls_sizegen_v4_data.sh 0   # SP sweep + fit the 1k test INRs per width
bash scripts/run_mnist_cls_sizegen_v4_eval.sh "0 2"   # re-score checkpoints, no retraining
```

**Configs**: `configs/mnist_cls/{scalegmn,gmn,mpgmn}_sizegen_{sp,mup24}_v3.yml` for training;
the `_v4.yml` variants are identical except for the four extra test widths and the
`mnist_sizegen_splits_v4.json` filename.
**Logs**: `/tmp/sizegen_v3_final/` (muP), `/tmp/mpgmn_sizegen_v3/` (matrix-product),
`/tmp/mpsgmn_sizegen_v3/` (matrix-product ScaleGMN),
`/tmp/sizegen_v3_bidir/` (bidirectional ScaleGMN + plain GMN),
`/tmp/sizegen_v4_eval/` + `/tmp/sizegen_v4_data/` (large widths). The four SP conditions were run
separately via `scripts/run_sp_v3_training.sh` after the INR refit; those logs
(`/tmp/sp_sizegen_v3_training/`) have been cleared, so their training-time numbers were recovered
from the local wandb run directories in `src/scalegmn/wandb/`.

### Plain GMN Results (`symmetry: permutation`, forward and bidirectional)

#### Forward

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.8550 / **0.8790** / **0.8790** / 0.8720 / 0.8710 / 0.8520 / 0.8390 → best **1.25e-4**
  (tied with 2.5e-4; argmax picked the smaller)
- SP dup-equiv: 0.8440 / 0.8710 / 0.8660 / **0.8800** / 0.8730 / 0.8510 / 0.8220 → best **5e-4**
- muP baseline: 0.8330 / 0.8530 / **0.8870** / 0.8510 / 0.8590 / 0.8450 / 0.8170 → best **2.5e-4**
- muP dup-equiv: 0.8440 / 0.8570 / 0.8840 / **0.8860** / 0.8770 / 0.8560 / 0.8110 → best **5e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring of the checkpoints below,
1k disjoint test images per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP24 baseline | muP24 dup-equiv |
|---|---|---|---|---|
| w24 (IN) | 94.3% | **95.8%** | 95.0% | 94.1% |
| w32 | 94.4% | **95.8%** | 93.8% | 95.7% |
| w48 | 82.9% | 95.1% | 83.3% | **96.5%** |
| w64 | 45.0% | 95.3% | 67.9% | **96.2%** |
| w80 | 20.6% | 93.4% | 50.8% | **97.6%** |
| w96 | 12.8% | 92.6% | 43.1% | **96.8%** |
| w128 | 12.0% | 92.3% | 33.2% | **95.7%** |
| w192 | 11.9% | 82.2% | 28.6% | **93.3%** |
| w256 | 10.7% | 68.3% | 26.9% | **90.4%** |
| w384 | 11.7% | 44.8% | 26.4% | **83.5%** |
| w512 | 10.6% | 22.4% | 28.2% | **65.8%** |
| w768 | 11.5% | 12.3% | 30.0% | **46.1%** |
| w1024 | 11.6% | 14.2% | 30.0% | **36.8%** |
| **OOD mean** (w32–w1024) | 28.0% | 67.4% | 45.2% | **82.9%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-v3-gmn-sp-base-full_lqrhopx1` | [`jzdkivti`](https://wandb.ai/yuxinma/mnist_cls/runs/jzdkivti) |
| SP dup-equiv | `sizegen-v3-gmn-sp-deq-full_c9y36coc` | [`f5hwcx9a`](https://wandb.ai/yuxinma/mnist_cls/runs/f5hwcx9a) |
| muP24 baseline | `sizegen-v3-gmn-mup-base_to0udreq` | [`giwdrtrh`](https://wandb.ai/yuxinma/mnist_cls/runs/giwdrtrh) |
| muP24 dup-equiv | `sizegen-v3-gmn-mup-deq_ikmnmpj4` | [`cqguwbwe`](https://wandb.ai/yuxinma/mnist_cls/runs/cqguwbwe) |

**Training runs** (200 epochs, patience=50; SP baseline, SP dup-equiv, and muP baseline all
early-stopped, muP dup-equiv ran to 200):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`lqrhopx1`](https://wandb.ai/yuxinma/mnist_cls/runs/lqrhopx1) | 1.25e-4 | 179 | 95.1% |
| SP dup-equiv | [`c9y36coc`](https://wandb.ai/yuxinma/mnist_cls/runs/c9y36coc) | 5e-4 | 140 | 95.4% |
| muP baseline | [`to0udreq`](https://wandb.ai/yuxinma/mnist_cls/runs/to0udreq) | 2.5e-4 | 133 | 96.4% |
| muP dup-equiv | [`ikmnmpj4`](https://wandb.ai/yuxinma/mnist_cls/runs/ikmnmpj4) | 5e-4 | 200 | 96.6% |

#### Bidirectional

Same protocol as the bidirectional ScaleGMN subsection above, with `symmetry: permutation`
(`GNN_layer`, concat+MLP messages). Run from the same queue as the forward block above, so the
sweep grid, epochs, patience and split are identical.

**Date**: 2026-08-19 – 2026-08-26

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.8500 / 0.8680 / 0.8740 / **0.8800** / 0.8780 / 0.8610 / 0.8350 → best **5e-4**
- SP dup-equiv: 0.8360 / 0.8560 / 0.8710 / **0.8960** / 0.8930 / 0.8760 / 0.8700 → best **5e-4**
- muP baseline: 0.8420 / 0.8750 / **0.8940** / 0.8710 / 0.8900 / 0.8630 / 0.8640 → best **2.5e-4**
- muP dup-equiv: 0.8240 / 0.8600 / 0.8710 / **0.8790** / 0.8730 / 0.8780 / 0.8320 → best **5e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring, 1k disjoint test images
per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP24 baseline | muP24 dup-equiv |
|---|---|---|---|---|
| w24 (IN) | 95.3% | 94.2% | 94.9% | **95.4%** |
| w32 | 90.2% | 94.7% | 92.2% | **96.5%** |
| w48 | 44.1% | 90.2% | 60.4% | **96.3%** |
| w64 | 12.2% | 91.6% | 29.7% | **97.1%** |
| w80 | 11.2% | 88.2% | 20.5% | **96.2%** |
| w96 | 11.2% | 86.8% | 17.7% | **96.9%** |
| w128 | 12.0% | 75.2% | 20.4% | **95.3%** |
| w192 | 11.9% | 41.4% | 19.6% | **92.2%** |
| w256 | 10.7% | 33.6% | 18.1% | **86.5%** |
| w384 | 11.7% | 30.5% | 21.5% | **68.2%** |
| w512 | 11.3% | 25.2% | 18.6% | **56.1%** |
| w768 | 15.8% | 13.2% | 18.9% | **37.5%** |
| w1024 | 16.2% | 13.3% | 18.9% | **28.0%** |
| **OOD mean** (w32–w1024) | 21.5% | 57.0% | 29.7% | **78.9%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-v3-gmn-sp-base-bidir_yu8rk2nv` | [`7t57tr34`](https://wandb.ai/yuxinma/mnist_cls/runs/7t57tr34) |
| SP dup-equiv | `sizegen-v3-gmn-sp-deq-bidir_tuiq98sc` | [`x3bg6qkc`](https://wandb.ai/yuxinma/mnist_cls/runs/x3bg6qkc) |
| muP24 baseline | `sizegen-v3-gmn-mup24-base-bidir_45215zqm` | [`6u5zibo2`](https://wandb.ai/yuxinma/mnist_cls/runs/6u5zibo2) |
| muP24 dup-equiv | `sizegen-v3-gmn-mup24-deq-bidir_u8oc73z3` | [`aioa4anr`](https://wandb.ai/yuxinma/mnist_cls/runs/aioa4anr) |

**Training runs** (200 epochs, patience=50):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`yu8rk2nv`](https://wandb.ai/yuxinma/mnist_cls/runs/yu8rk2nv) | 5e-4 | 200 (no early stop) | 95.4% |
| SP dup-equiv | [`tuiq98sc`](https://wandb.ai/yuxinma/mnist_cls/runs/tuiq98sc) | 5e-4 | 106 (early-stopped) | 94.5% |
| muP baseline | [`45215zqm`](https://wandb.ai/yuxinma/mnist_cls/runs/45215zqm) | 2.5e-4 | 125 (early-stopped) | 95.4% |
| muP dup-equiv | [`u8oc73z3`](https://wandb.ai/yuxinma/mnist_cls/runs/u8oc73z3) | 5e-4 | 157 (early-stopped) | 95.2% |

### ScaleGMN Results (`symmetry: scale`, forward and bidirectional)

#### Forward

**LR sweep** (7 LRs from 6.25e-5 to 4e-3, $\times 2$ grid, 8600 steps each; val acc at
6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.9520 / 0.9540 / **0.9590** / 0.9470 / 0.9570 / 0.9550 / 0.9510 → best **2.5e-4**
- SP dup-equiv: 0.9500 / 0.9520 / **0.9560** / 0.9480 / 0.9480 / 0.9450 / 0.9380 → best **2.5e-4**
- muP baseline: 0.9600 / 0.9640 / 0.9650 / **0.9660** / 0.9600 / 0.9520 / 0.9410 → best **5e-4**
- muP dup-equiv: 0.9570 / 0.9560 / 0.9540 / **0.9650** / 0.9560 / 0.9470 / 0.9420 → best **5e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring of the checkpoints below,
1k disjoint test images per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP24 baseline | muP24 dup-equiv |
|---|---|---|---|---|
| w24 (IN) | 97.1% | 96.9% | 96.8% | **97.5%** |
| w32 | 97.2% | 95.8% | 97.5% | **98.1%** |
| w48 | 97.3% | 96.1% | 97.9% | **98.3%** |
| w64 | 97.0% | 96.5% | 95.7% | **97.5%** |
| w80 | 96.5% | 95.9% | 96.1% | **98.1%** |
| w96 | 97.3% | 95.3% | 95.7% | **98.0%** |
| w128 | 96.6% | 93.2% | 94.7% | **98.2%** |
| w192 | 88.8% | 86.1% | 94.4% | **98.8%** |
| w256 | 77.3% | 81.9% | 92.6% | **98.2%** |
| w384 | 51.8% | 65.0% | 87.2% | **98.8%** |
| w512 | 42.7% | 57.4% | 84.0% | **97.7%** |
| w768 | 22.5% | 40.5% | 76.7% | **97.2%** |
| w1024 | 19.1% | 36.0% | 71.3% | **95.2%** |
| **OOD mean** (w32–w1024) | 73.7% | 78.3% | 90.3% | **97.8%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-v3-sgmn-sp-base-full_f8f8r0zp` | [`ui6ff92f`](https://wandb.ai/yuxinma/mnist_cls/runs/ui6ff92f) |
| SP dup-equiv | `sizegen-v3-sgmn-sp-deq-full_bngbeln2` | [`ngyv74ch`](https://wandb.ai/yuxinma/mnist_cls/runs/ngyv74ch) |
| muP24 baseline | `sizegen-v3-sgmn-mup-base_2ao0wbnm` | [`bngypo1g`](https://wandb.ai/yuxinma/mnist_cls/runs/bngypo1g) |
| muP24 dup-equiv | `sizegen-v3-sgmn-mup-deq_vgr92oh9` | [`ljootl07`](https://wandb.ai/yuxinma/mnist_cls/runs/ljootl07) |

**Training runs** (200 epochs, patience=50; all four runs early-stopped):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`f8f8r0zp`](https://wandb.ai/yuxinma/mnist_cls/runs/f8f8r0zp) | 2.5e-4 | 140 | 97.3% |
| SP dup-equiv | [`bngbeln2`](https://wandb.ai/yuxinma/mnist_cls/runs/bngbeln2) | 2.5e-4 | 93 | 96.8% |
| muP baseline | [`2ao0wbnm`](https://wandb.ai/yuxinma/mnist_cls/runs/2ao0wbnm) | 5e-4 | 74 | 97.6% |
| muP dup-equiv | [`vgr92oh9`](https://wandb.ai/yuxinma/mnist_cls/runs/vgr92oh9) | 5e-4 | 110 | 98.1% |

#### Bidirectional

Same eight-condition protocol as the forward runs, with `direction: bidirectional`
(`ScaleGMN_GNN_bidir`): each layer aggregates forward messages (l−1 → l) and backward messages
(l+1 → l) and updates via `update_node_feats_fn(cat(x, fw_aggr, bw_aggr))`, and the readout uses
both first- and last-layer nodes. Everything else is unchanged: d_hid=128, 4 layers,
`readout_range: full_graph`, batch_size=64, AdamW, warmup 1000 steps, 200 epochs, patience=50,
the same 7-value LR sweep selected on w24 val accuracy, and the same v3 training / v4 eval-only
re-scoring split.

**Date**: 2026-08-30 – 2026-09-01
**Launch**: `bash scripts/run_mnist_cls_scalegmn_bidir_recip.sh "7"` (one worker per GPU draining a
5-job queue: ScaleGMN × {SP, muP24} × {baseline, dup-equiv}, plus the Experiment 1 rerun)
**Configs**: `configs/mnist_cls/scalegmn_sizegen_{sp,mup24}_v3.yml` with `--direction bidirectional`
(`--bidir-reciprocal` defaults to `True`, installing `src/models/bidir_reciprocal.py`; matching
`_v4.yml` for the w24–w1024 re-score)
**Logs**: `/tmp/sizegen_v3_bidir_recip/` (incremental `SUMMARY.md`, per-run sweep / full / eval logs)

**LR sweep** (7 LRs from 6.25e-5 to 4e-3, $\times 2$ grid, 8600 steps each; val acc at
6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: **0.9570** / 0.9520 / 0.9480 / 0.9490 / 0.9470 / 0.9450 / 0.9370 → best **6.25e-5**
- SP dup-equiv: **0.9530** / 0.9530 / 0.9530 / 0.9510 / 0.9460 / 0.9420 / 0.9420 → best **6.25e-5**
- muP baseline: 0.9510 / **0.9640** / 0.9550 / 0.9610 / 0.9580 / 0.9440 / 0.9440 → best **1.25e-4**
- muP dup-equiv: 0.9580 / **0.9620** / 0.9590 / 0.9600 / 0.9560 / 0.9580 / 0.9340 → best **1.25e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring, 1k disjoint test images
per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP24 baseline | muP24 dup-equiv |
|---|---|---|---|---|
| w24 (IN) | 96.1% | 96.4% | 96.4% | **97.4%** |
| w32 | 96.8% | 97.2% | 96.6% | **97.5%** |
| w48 | 95.1% | 95.7% | 97.2% | **98.1%** |
| w64 | 96.4% | 97.2% | 95.2% | **97.4%** |
| w80 | 95.8% | 96.2% | 93.9% | **97.7%** |
| w96 | 93.9% | 96.3% | 92.3% | **98.3%** |
| w128 | 91.5% | 93.2% | 88.4% | **98.4%** |
| w192 | 64.3% | 86.1% | 71.6% | **98.8%** |
| w256 | 43.7% | 80.4% | 42.8% | **98.6%** |
| w384 | 17.2% | 67.9% | 18.3% | **98.5%** |
| w512 | 13.2% | 59.4% | 14.5% | **97.1%** |
| w768 | 13.4% | 42.0% | 11.0% | **96.5%** |
| w1024 | 12.7% | 37.6% | 11.0% | **94.6%** |
| **OOD mean** (w32–w1024) | 61.2% | 79.1% | 61.1% | **97.6%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-v3-recip-sgmn-sp-base-bidir-recip_7ubil4t3` | [`rd32tbtp`](https://wandb.ai/yuxinma/mnist_cls/runs/rd32tbtp) |
| SP dup-equiv | `sizegen-v3-recip-sgmn-sp-deq-bidir-recip_j73y79y1` | [`7dugztra`](https://wandb.ai/yuxinma/mnist_cls/runs/7dugztra) |
| muP24 baseline | `sizegen-v3-recip-sgmn-mup24-base-bidir-recip_qxx0ybx2` | [`p8kk9rqf`](https://wandb.ai/yuxinma/mnist_cls/runs/p8kk9rqf) |
| muP24 dup-equiv | `sizegen-v3-recip-sgmn-mup24-deq-bidir-recip_dsxv4qxi` | [`teoq1m1w`](https://wandb.ai/yuxinma/mnist_cls/runs/teoq1m1w) |

**Training runs** (200 epochs, patience=50):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`7ubil4t3`](https://wandb.ai/yuxinma/mnist_cls/runs/7ubil4t3) | 6.25e-5 | 123 | 97.1% |
| SP dup-equiv | [`j73y79y1`](https://wandb.ai/yuxinma/mnist_cls/runs/j73y79y1) | 6.25e-5 | 104 | 96.8% |
| muP baseline | [`qxx0ybx2`](https://wandb.ai/yuxinma/mnist_cls/runs/qxx0ybx2) | 1.25e-4 | 117 | 97.3% |
| muP dup-equiv | [`dsxv4qxi`](https://wandb.ai/yuxinma/mnist_cls/runs/dsxv4qxi) | 1.25e-4 | 192 | 97.5% |

### Matrix-Product GMN Results (`message_fn_type: matrix_product`, forward + bidirectional)

Same protocol and LR grid as above. The matrix-product GMN is **dup-equiv only** — its MSG
function (`edge_attr * A(x_j)` with the raw scalar weight) only becomes the matrix product
$Z^{(l)} = W^{(l)} H^{(l-1)} A$ on top of fan-in rescaling + mean aggregation + layer-wise mean readout,
so there is no baseline variant. Both message-passing directions were run.

**Date**: 2026-08-19
**Configs**: `configs/mnist_cls/mpgmn_sizegen_{sp,mup24}_v3.yml`, with
`--duplication-equiv True --direction {forward,bidirectional}`
**Launch**: `bash scripts/run_mnist_cls_mpgmn_sizegen_v3.sh` (one worker per GPU draining a job
queue; incremental summary + per-run logs in `/tmp/mpgmn_sizegen_v3/`)

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP forward: 0.9080 / 0.9240 / **0.9280** / 0.9200 / 0.9200 / 0.9040 / 0.8940 → best **2.5e-4**
- muP forward: 0.9310 / 0.9340 / **0.9360** / 0.9250 / 0.9250 / 0.9240 / 0.9100 → best **2.5e-4**
- SP bidirectional: 0.9060 / 0.9190 / 0.9210 / 0.9230 / **0.9270** / 0.8890 / 0.8880 → best **1e-3**
- muP bidirectional: 0.9010 / 0.9180 / **0.9260** / 0.9250 / 0.9120 / 0.9120 / 0.8890 → best **2.5e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring of the checkpoints below,
1k disjoint test images per width; row-best in bold):

| Width | SP forward | muP24 forward | SP bidir | muP24 bidir |
|---|---|---|---|---|
| w24 (IN) | **96.3%** | 95.6% | 95.1% | 95.3% |
| w32 | 97.1% | 96.4% | 95.9% | **97.2%** |
| w48 | 97.0% | 97.6% | 96.4% | **98.4%** |
| w64 | 96.6% | **97.5%** | 96.7% | 97.4% |
| w80 | 97.0% | 96.9% | 95.5% | **97.3%** |
| w96 | 96.1% | 97.5% | 94.6% | **98.2%** |
| w128 | 95.2% | 97.7% | 94.3% | **97.9%** |
| w192 | 94.3% | 97.6% | 91.7% | **98.3%** |
| w256 | 92.2% | 97.7% | 86.7% | **98.3%** |
| w384 | 88.4% | 97.9% | 76.8% | **99.1%** |
| w512 | 83.6% | 98.0% | 72.3% | **98.2%** |
| w768 | 80.0% | 97.9% | 58.9% | **98.7%** |
| w1024 | 68.1% | 97.7% | 53.2% | **97.8%** |
| **OOD mean** (w32–w1024) | 90.5% | 97.5% | 84.4% | **98.1%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP forward | `sizegen-v3-mpgmn-sp-fw_u1ox1bd0` | [`sddd8l6u`](https://wandb.ai/yuxinma/mnist_cls/runs/sddd8l6u) |
| muP24 forward | `sizegen-v3-mpgmn-mup24-fw_ktv38f0r` | [`nj5fxup5`](https://wandb.ai/yuxinma/mnist_cls/runs/nj5fxup5) |
| SP bidirectional | `sizegen-v3-mpgmn-sp-bidir_it3vpqm0` | [`hd39pbds`](https://wandb.ai/yuxinma/mnist_cls/runs/hd39pbds) |
| muP24 bidirectional | `sizegen-v3-mpgmn-mup24-bidir_aid3oqcf` | [`i2gdj0b0`](https://wandb.ai/yuxinma/mnist_cls/runs/i2gdj0b0) |

Training runs behind the checkpoints (LR from the sweep above): SP forward
[`u1ox1bd0`](https://wandb.ai/yuxinma/mnist_cls/runs/u1ox1bd0) @ 2.5e-4, muP24 forward
[`ktv38f0r`](https://wandb.ai/yuxinma/mnist_cls/runs/ktv38f0r) @ 2.5e-4, SP bidirectional
[`it3vpqm0`](https://wandb.ai/yuxinma/mnist_cls/runs/it3vpqm0) @ 1e-3, muP24 bidirectional
[`aid3oqcf`](https://wandb.ai/yuxinma/mnist_cls/runs/aid3oqcf) @ 2.5e-4.
All four finished 2026-08-19.

### Matrix-Product ScaleGMN Results (`message_fn_type: matrix_product_scale`, forward)

The same MSG constraint on top of **ScaleGMN's** scale-equivariant node states and node update
instead of the plain GMN's concat-MLP — i.e. the ScaleGMN message $\Psi_m(w_e(e) \odot w_v(h))$ restricted
to $w_e = \text{id}$, $w_v = \text{id}$, $\Psi_m(x) = xA$, with the update still ScaleGMN's
`EquivariantNet` on `cat(H^(l), Z^(l))` (`paper/main.tex` § "Forward matrix-product ScaleGMN"). Dup-equiv only, for the
same reason as `mpgmn`, and forward only by scope. Protocol, LR grid, epochs, patience and split
files are identical to the `mpgmn` block above, so these are drop-in columns in the same table.

Properties verified before queueing — see
[VERIFICATION_gmn_properties.md § Verification of the matrix-product models](VERIFICATION_gmn_properties.md#verification-of-the-matrix-product-models):
aggregate identity 1.9e-6, forward equivalence on all five widening families (~8e-7), the
duplication ablation's PASS/FAIL/FAIL pattern, and spectral continuity 5.39e-3 → 3.37e-4 over
n = 16 → 256 (vs plain ScaleGMN flat at ratio 0.98).

**Date**: 2026-08-28 (queued 00:46, both conditions done 20:40)
**Configs**: `configs/mnist_cls/mpsgmn_sizegen_{sp,mup24}_v3.yml` for training,
`..._v4.yml` for the w24–w1024 re-score, with `--duplication-equiv True --direction forward`
**Launch**: `bash scripts/run_mnist_cls_mpsgmn_sizegen_v3.sh ""` seeds the queue (2 conditions:
SP, muP24); workers are armed per GPU via `add_worker_when_free.sh`
**Logs**: `/tmp/mpsgmn_sizegen_v3/` (incremental `SUMMARY.md`, per-run sweep / full / eval logs)

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP: 0.9440 / **0.9570** / 0.9560 / 0.9550 / 0.9500 / 0.9460 / 0.9320 → best **1.25e-4**
- muP24: 0.9530 / 0.9640 / 0.9600 / **0.9650** / 0.9480 / 0.9490 / 0.9450 → best **5e-4**

**Per-width test accuracy, w24 → w1024** (`--eval-only-ckpt` re-scoring on the v4 splits, 1k
disjoint test images per width; row-best in bold):

| Width | SP | muP24 |
|---|---|---|
| w24 (IN) | **97.2%** | 96.6% |
| w32 | 97.4% | **97.9%** |
| w48 | 96.9% | **98.2%** |
| w64 | 97.2% | **97.7%** |
| w80 | **98.0%** | 97.3% |
| w96 | 97.7% | **98.2%** |
| w128 | 97.8% | **98.3%** |
| w192 | 97.3% | **99.0%** |
| w256 | 96.0% | **98.5%** |
| w384 | 95.5% | **99.1%** |
| w512 | 92.7% | **98.8%** |
| w768 | 89.0% | **98.9%** |
| w1024 | 84.1% | **98.8%** |
| **OOD mean** (w32–w1024) | 95.0% | **98.4%** |

| Condition | Training run | LR | Epochs | Best val | Checkpoint | Eval run |
|---|---|---|---|---|---|---|
| SP | [`w6a7k0zd`](https://wandb.ai/yuxinma/mnist_cls/runs/w6a7k0zd) | 1.25e-4 | 134 | 97.4% | `sizegen-v3-mpsgmn-sp_w6a7k0zd` | [`rw310twd`](https://wandb.ai/yuxinma/mnist_cls/runs/rw310twd) |
| muP24 | [`g451pl6g`](https://wandb.ai/yuxinma/mnist_cls/runs/g451pl6g) | 5e-4 | 129 | 98.0% | `sizegen-v3-mpsgmn-mup24_g451pl6g` | [`g6xjn66a`](https://wandb.ai/yuxinma/mnist_cls/runs/g6xjn66a) |

### Forward vs bidirectional — all ten conditions

$\Delta$ = bidirectional − forward, OOD mean over w32–w1024; better direction in bold. The four
ScaleGMN rows are the reciprocal-fix runs from the
[ScaleGMN block](#scalegmn-results-symmetry-scale-forward-and-bidirectional); the six
`symmetry: permutation` rows need no reciprocal and are unchanged.

| Condition | Forward | Bidirectional | $\Delta$ |
|---|---|---|---|
| ScaleGMN, SP, dup-equiv | 78.3% | **79.1%** | **+0.8pp** |
| Matrix-product GMN, muP24 | 97.5% | **98.1%** | **+0.6pp** |
| ScaleGMN, muP24, dup-equiv | **97.8%** | 97.6% | −0.2pp |
| Plain GMN, muP24, dup-equiv | **82.9%** | 78.9% | −4.0pp |
| Matrix-product GMN, SP | **90.5%** | 84.4% | −6.1pp |
| Plain GMN, SP, baseline | **28.0%** | 21.5% | −6.5pp |
| Plain GMN, SP, dup-equiv | **67.4%** | 57.0% | −10.4pp |
| ScaleGMN, SP, baseline | **73.7%** | 61.2% | −12.5pp |
| Plain GMN, muP24, baseline | **45.2%** | 29.7% | −15.5pp |
| ScaleGMN, muP24, baseline | **90.3%** | 61.1% | −29.2pp |

1. Bidirectional loses in eight of ten conditions. The three at or above parity are all dup-equiv or
   matrix-product; the three largest penalties ($-12.5$ to $-29.2$pp) are baselines, and the fourth
   largest ($-10.4$pp) is instead Plain GMN, SP, dup-equiv.
2. Within each family, both directions and both arms order dup-equiv $>$ baseline.

---

## Why ScaleGMN baseline generalizes relatively well without duplication equivariance

The ScaleGMN baseline (sum aggregation, no fan-in rescaling, sum-pooling readout) reaches 90.3% muP
OOD mean over w32–w1024 in Experiment 3 and a 4.6% drop on uniformly widened INRs in Experiment 2
(forward), against 45.2% and 62.5% for the plain GMN baseline. The mechanism is the
**`InvariantLayer` with `symmetry: scale`** inside `EquivariantNet`
(`src/scalegmn/src/scalegmn/layers.py:203-206`):

```python
norms = torch.norm(x, p=2, dim=-1, keepdim=True)
r = x / (norms + eps)
```

Every GNN layer in ScaleGMN routes node/edge features through `EquivariantNet`, which projects them
onto the unit sphere ($x/\|x\|$) before the MLP, so the MLP always sees unit-norm inputs regardless
of the width-dependent weight norms ($\|w\|_{V_n} = \Theta(\sqrt{n})$ under SP); the aggregate is
re-normalized at the next layer rather than left invariant exactly, which is why the baseline still
degrades gently rather than not at all. `GNN_layer` in plain GMN instead feeds
`MLPNet(cat(edge_attr, x_j))` raw concatenated magnitudes with no normalization, so its MLP must
handle arbitrary input scales OOD.

This bounds scale, not direction, so it buys nothing once the widening is non-uniform: under
Experiment 2's `general` family the same baseline loses $60$–$69$%.

---

## Historical Notes (deprecated)

### Size Generalization v2 (deprecated) — train w16, test w24–w96

**Deprecated (2026-08-18)**: superseded by Experiment 3, which trains on w24 (now the base
width for all current experiments) and tests out to w256. This experiment trained on w16, whose INR fits
are the lowest-quality of any width (27.3 dB, 3.0% failure), and its $\leq 6\times$ width range was too narrow to
separate the baseline from the dup-equiv variant for ScaleGMN. Kept for reference; its checkpoints were
the ones evaluated in the [deprecated w16 Experiment 2](#deprecated-experiment-2-w16-base-size-gen-v2-checkpoints).

**Goal**: Test whether duplication-equivariant modifications improve generalization from
small-width INRs to large-width INRs.

**Setup**: Train on w16 only; test OOD on w24, w32, w48, w64, w80, w96.
Non-overlapping images per width (`generate_sizegen_splits.py`): 55k train / 1k val / 2k test
for w16; 2k test-only for each other width. No image appears at multiple widths.
batch_size=64, AdamW, warmup 1000 steps, patience=50.

#### Models

Four model variants, built on `ScaleGMN` (`src/scalegmn/`). Two model families $\times \{\text{baseline}, \text{dup-equiv}\}$:

**ScaleGMN** (`symmetry: scale`) — scale-equivariant message passing:

| | Model 1a — Baseline | Model 1b — Dup-equiv |
|---|---|---|
| GNN layers | `ScaleEq_GNN_layer` (EquivariantNet msg/upd) | same |
| Aggregator | `add` (sum) | `mean` |
| Edge features | raw weights | fan-in rescaled: $d_{l-1} \cdot W^{(l)}$ |
| Readout | `PermScaleInvariantReadout` (sum pool) | `LayerWiseMeanReadout` (per-layer mean) |
| Dataset | `LabeledINRDataset` | `FanInLabeledINRDataset` |

**Plain GMN** (`symmetry: permutation`) — no scale equivariance:

| | Model 2a — Baseline | Model 2b — Dup-equiv |
|---|---|---|
| GNN layers | `GNN_layer` (concat+MLP msg/upd) | same |
| Aggregator | `add` (sum) | `mean` |
| Edge features | raw weights | fan-in rescaled: $d_{l-1} \cdot W^{(l)}$ |
| Readout | `DeepSet` (sum pool) | `LayerWiseMeanReadout` (per-layer mean, plain MLP) |
| Dataset | `LabeledINRDataset` | `FanInLabeledINRDataset` |

All variants: d_hid=128, 4 GNN layers, forward direction, `readout_range: full_graph`.
Duplication equivalence verified in the section above (d_hid=32, 2 layers for speed).

#### Data conditions

- **SP (standard parameterization)**: ReLU INRs from `w{16,24,32,48,64,80,96}_sp/`.
  Weights have $\|w\|_{V_n} = \Theta(\sqrt{n})$ — norms grow with width.
- **muP (maximal update parameterization)**: ReLU INRs from `w{16,24,32,48,64,80,96}_mup/`.
  Weights have $\|w\|_{V_n} = \Theta(1)$ — norms are width-independent.

This gives 4 conditions: $\{\text{SP}, \text{muP}\} \times \{\text{baseline}, \text{dup-equiv}\}$.

**Protocol**: LR sweep (7 values, 6.25e-5 to 4e-3, $\times 2$ grid, ~10 epochs / 8600 steps) per
condition separately, then full training (200 epochs, patience=50) at best LR per condition.

**Date**: 2026-06-25

#### Reproduction

```bash
source .env

# ScaleGMN (all 4 conditions: LR sweep → full training, ~12h on GPUs 6-7)
bash scripts/run_mnist_cls_scalegmn_sizegen.sh

# Plain GMN (all 4 conditions: LR sweep → full training, ~4h on GPUs 2-7)
bash scripts/run_mnist_cls_gmn_sizegen.sh
```

#### ScaleGMN Results

#### LR Sweep (8600 steps each, 7 LRs per condition)

**Date**: 2026-06-25

| LR | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| 6.25e-5 | 0.9460 | 0.9400 | 0.9480 | 0.9450 |
| 1.25e-4 | 0.9480 | 0.9450 | 0.9460 | 0.9500 |
| **2.5e-4** | **0.9510** | 0.9470 | **0.9510** | 0.9490 |
| 5e-4 | 0.9480 | **0.9480** | 0.9480 | 0.9460 |
| 1e-3 | 0.9390 | 0.9350 | 0.9470 | 0.9430 |
| 2e-3 | 0.9390 | 0.9390 | 0.9390 | 0.9440 |
| 4e-3 | 0.9240 | 0.9360 | 0.9350 | 0.9300 |

Best LRs selected: SP baseline=2.5e-4, SP dup-equiv=5e-4, muP baseline=2.5e-4, muP dup-equiv=1.25e-4.
(Flat plateau in 1.25e-4 to 5e-4 range; script picked argmax.)

#### Full Training (200 epochs, patience=50)

SP runs early-stopped (epoch 178, 120). muP runs early-stopped (epoch 87, 99).

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 95.3% | 95.3% | 94.5% | 94.7% |
| w24 (OOD) | 96.7% | 97.1% | 97.7% | 97.5% |
| w32 (OOD) | 97.2% | 96.9% | 96.6% | 97.0% |
| w48 (OOD) | 96.6% | 97.7% | 97.2% | 97.6% |
| w64 (OOD) | 95.9% | 96.3% | 96.9% | 97.3% |
| w80 (OOD) | 95.6% | 96.1% | 96.3% | 97.7% |
| w96 (OOD) | 95.1% | 96.2% | 96.1% | 97.8% |
| **OOD mean** | **96.2%** | **96.7%** | **96.8%** | **97.5%** |

| Variant | Run ID | Config | LR |
|---|---|---|---|
| SP baseline | [`0gybak0l`](https://wandb.ai/yuxinma/mnist_cls/runs/0gybak0l) | `scalegmn_sizegen_sp.yml` (train w16 only, non-overlapping) | 2.5e-4 |
| SP dup-equiv | [`odd4h8dy`](https://wandb.ai/yuxinma/mnist_cls/runs/odd4h8dy) | same + `--duplication-equiv True` | 5e-4 |
| muP baseline | [`7p9oxtmg`](https://wandb.ai/yuxinma/mnist_cls/runs/7p9oxtmg) | `scalegmn_sizegen_mup.yml` (train w16 only, non-overlapping) | 2.5e-4 |
| muP dup-equiv | [`felar5jw`](https://wandb.ai/yuxinma/mnist_cls/runs/felar5jw) | same + `--duplication-equiv True` | 1.25e-4 |

#### Plain GMN (symmetry: permutation)

Same setup as above but using plain GMN (`GNN_layer` with concat+MLP, `DeepSet` readout).
Config: `configs/mnist_cls/gmn_sizegen_sp.yml` / `gmn_sizegen_mup.yml`.

#### LR Sweep (8600 steps each, 7 LRs per condition)

**Date**: 2026-06-25

| LR | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| 6.25e-5 | 0.7910 | 0.7860 | 0.7850 | 0.7800 |
| 1.25e-4 | 0.8160 | 0.8260 | 0.8260 | 0.8150 |
| **2.5e-4** | **0.8410** | 0.8400 | **0.8550** | **0.8550** |
| **5e-4** | 0.8350 | **0.8530** | 0.8380 | 0.8430 |
| 1e-3 | 0.8190 | 0.8350 | 0.8210 | 0.8280 |
| 2e-3 | 0.8180 | 0.7950 | 0.7830 | 0.8030 |
| 4e-3 | 0.7630 | 0.7850 | 0.7760 | 0.7410 |

Best LRs selected: SP baseline=2.5e-4, SP dup-equiv=5e-4, muP baseline=2.5e-4, muP dup-equiv=2.5e-4.

#### Full Training (200 epochs, patience=50)

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 93.2% | 93.5% | 93.0% | 93.2% |
| w24 (OOD) | 91.9% | 95.4% | 91.9% | 95.1% |
| w32 (OOD) | 88.7% | 95.4% | 86.3% | 94.9% |
| w48 (OOD) | 74.0% | 94.3% | 56.0% | 96.0% |
| w64 (OOD) | 47.3% | 91.6% | 26.5% | 95.5% |
| w80 (OOD) | 31.2% | 89.9% | 14.2% | 95.8% |
| w96 (OOD) | 24.5% | 87.5% | 12.5% | 95.9% |
| **OOD mean** | **59.6%** | **92.3%** | **47.9%** | **95.5%** |

| Variant | Run ID | Config | LR |
|---|---|---|---|
| SP baseline | [`edjqee1e`](https://wandb.ai/yuxinma/mnist_cls/runs/edjqee1e) | `gmn_sizegen_sp.yml` | 2.5e-4 |
| SP dup-equiv | [`cbx3aq7y`](https://wandb.ai/yuxinma/mnist_cls/runs/cbx3aq7y) | same + `--duplication-equiv True` | 5e-4 |
| muP baseline | [`7r46r78v`](https://wandb.ai/yuxinma/mnist_cls/runs/7r46r78v) | `gmn_sizegen_mup.yml` | 2.5e-4 |
| muP dup-equiv | [`2j28l493`](https://wandb.ai/yuxinma/mnist_cls/runs/2j28l493) | same + `--duplication-equiv True` | 2.5e-4 |

---

### Deprecated Experiment 2 (w16 base, size-gen v2 checkpoints)

Original duplicated-INR sanity check (2026-06-25): w16 base, fixed-LR SP and muP `base_width=32`
data, size-gen v2 checkpoints (train w16), widened to w32–w96 (k=2…6). Superseded by the
[current Experiment 2](#experiment-2--size-generalization-on-wider-equivalent-networks-sanity-check)
on w24 with the Experiment 3 checkpoints.

**ScaleGMN**:

| Condition | w16 (orig) | w32 (dup) | w48 (dup) | w64 (dup) | w80 (dup) | w96 (dup) | mean_dup | max_drop |
|---|---|---|---|---|---|---|---|---|
| SP baseline | 95.3% | 94.9% | 94.1% | 93.2% | 92.4% | 91.6% | 93.2% | 3.7% |
| SP dup-equiv | 95.3% | 95.3% | 95.3% | 95.3% | 95.3% | 95.3% | 95.3% | 0.0% |
| muP baseline | 94.5% | 93.9% | 93.0% | 91.8% | 90.6% | 90.2% | 91.9% | 4.4% |
| muP dup-equiv | 94.7% | 94.7% | 94.7% | 94.7% | 94.7% | 94.7% | 94.7% | 0.0% |

Run IDs: SP (both) [`0fr54uae`](https://wandb.ai/yuxinma/mnist_cls/runs/0fr54uae),
muP (both) [`o9l3jxvy`](https://wandb.ai/yuxinma/mnist_cls/runs/o9l3jxvy).

**Plain GMN**:

| Condition | w16 (orig) | w32 (dup) | w48 (dup) | w64 (dup) | w80 (dup) | w96 (dup) | mean_dup | max_drop |
|---|---|---|---|---|---|---|---|---|
| GMN SP baseline | 93.2% | 84.9% | 69.7% | 52.6% | 40.8% | 35.6% | 56.7% | 57.6% |
| GMN SP dup-equiv | 93.5% | 93.5% | 93.5% | 93.5% | 93.5% | 93.5% | 93.5% | 0.0% |
| GMN muP baseline | 93.0% | 78.6% | 48.2% | 30.0% | 21.4% | 18.5% | 39.3% | 74.5% |
| GMN muP dup-equiv | 93.2% | 93.2% | 93.2% | 93.2% | 93.2% | 93.2% | 93.2% | 0.0% |

Run IDs: GMN SP (both) [`vjylbidt`](https://wandb.ai/yuxinma/mnist_cls/runs/vjylbidt),
GMN muP (both) [`kgtmleit`](https://wandb.ai/yuxinma/mnist_cls/runs/kgtmleit).

The `w{32,48,64,80,96}_{sp,mup}_dup/` data directories belong to this deprecated run; the current
one uses `w{48,96,144,192,240}_{sp,mup24}_dup/`.

---

### Deprecated norm statistics (w16 rows, fixed-LR SP, muP base_width=32)

Original measurements (2026-05-12), superseded by the re-measurement on per-width-LR SP fits and
muP base_width=24. Kept because size-gen v2 and the deprecated Experiment 2 were run on this data.

**SP trained ReLU INRs** (fixed lr=0.01, w16–w96):

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 16 | 39.48 ± 4.19 | 2.81 | 12.84 | 23.83 | 0.604 |
| 24 | 43.35 ± 4.24 | 2.49 | 13.61 | 27.25 | 0.629 |
| 32 | 46.21 ± 5.12 | 2.28 | 14.06 | 29.88 | 0.647 |
| 40 | 48.76 ± 5.34 | 2.14 | 14.43 | 32.18 | 0.660 |
| 48 | 51.32 ± 5.53 | 2.05 | 14.59 | 34.68 | 0.676 |
| 64 | 55.44 ± 6.50 | 1.93 | 14.90 | 38.62 | 0.697 |
| 80 | 59.33 ± 6.89 | 1.83 | 15.10 | 42.39 | 0.714 |
| 96 | 62.37 ± 7.52 | 1.77 | 15.26 | 45.33 | 0.727 |

**muP at initialization, base_width=32**:

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 16 | 12.14 ± 1.37 | 1.60 | 2.62 | 7.92 | 0.652 |
| 24 | 12.12 ± 1.12 | 1.56 | 2.67 | 7.89 | 0.651 |
| 32 | 12.12 ± 0.97 | 1.55 | 2.69 | 7.88 | 0.650 |
| 48 | 12.17 ± 0.78 | 1.53 | 2.73 | 7.91 | 0.650 |
| 64 | 12.20 ± 0.67 | 1.52 | 2.75 | 7.94 | 0.651 |
| 96 | 12.24 ± 0.60 | 1.50 | 2.77 | 7.97 | 0.651 |
| 128 | 12.25 ± 0.50 | 1.49 | 2.78 | 7.99 | 0.652 |
| 192 | 12.27 ± 0.42 | 1.48 | 2.79 | 8.00 | 0.652 |
| 256 | 12.28 ± 0.37 | 1.47 | 2.80 | 8.02 | 0.652 |

**muP trained ReLU INRs, base_width=32** (re-fitted in-script, 1000 steps, lr=0.01):

| Width | $\|w\|_{V_n}$ | L0 | L1 | L2 (last) | Last/Total |
|-------|-----------|------|------|-----------|------------|
| 16 | 52.37 ± 8.75 | 2.35 | 15.20 | 34.83 | 0.665 |
| 24 | 51.69 ± 7.89 | 2.35 | 15.02 | 34.31 | 0.664 |
| 32 | 51.20 ± 7.39 | 2.36 | 14.88 | 33.96 | 0.663 |
| 48 | 50.78 ± 6.74 | 2.36 | 14.52 | 33.90 | 0.668 |
| 64 | 50.61 ± 6.48 | 2.36 | 14.35 | 33.90 | 0.670 |
| 96 | 50.44 ± 6.22 | 2.36 | 14.20 | 33.88 | 0.672 |
| 128 | 50.40 ± 5.78 | 2.37 | 13.98 | 34.05 | 0.676 |
| 192 | 50.43 ± 5.32 | 2.37 | 13.78 | 34.28 | 0.680 |
| 256 | 50.49 ± 5.45 | 2.37 | 13.62 | 34.49 | 0.683 |

$\Theta(1)$ across widths, same conclusion as base_width=24; the absolute level differs (~50.5 vs ~42)
because w32 rather than w24 is the base width.

The SP-at-initialization table is unchanged by either issue (init does not depend on the fitting LR);
only its w16 and w40 rows were dropped.

---

### Deprecated INR fits (w16, fixed-LR SP, muP base_width=32)

**Fixed-LR SP fits** (lr=0.01 for every width) — superseded by the per-width LR sweep; the
`w{N}_sp/` directories for w24–w256 were regenerated, so these numbers no longer describe the data
on disk. Kept because size-gen v2 / the deprecated w16 Experiment 2 were run on them (and w16 was
never re-fitted).

| Width | MSE mean | MSE median | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|---|---|
| 16 | 0.002740 | 0.001757 | 27.3 dB | 27.6 dB | 3.63 | 3.0% |
| 24 | 0.001249 | 0.000878 | 30.4 dB | 30.6 dB | 3.27 | 0.1% |
| 32 | 0.000850 | 0.000585 | 32.1 dB | 32.3 dB | 3.16 | 0.1% |
| 40 | 0.000657 | 0.000477 | 33.1 dB | 33.2 dB | 3.02 | 0.0% |
| 48 | 0.000577 | 0.000400 | 33.8 dB | 34.0 dB | 3.09 | 0.1% |
| 64 | 0.000484 | 0.000318 | 34.8 dB | 35.0 dB | 3.19 | 0.2% |
| 80 | 0.000425 | 0.000267 | 35.5 dB | 35.7 dB | 3.25 | 0.2% |
| 96 | 0.000436 | 0.000249 | 35.8 dB | 36.0 dB | 3.53 | 0.1% |
| 128 | 0.000564 | 0.000223 | 36.1 dB | 36.5 dB | 3.68 | 0.3% |
| 192 | 0.011080 | 0.000249 | 33.6 dB | 36.0 dB | 8.67 | 10.2% |
| 256 | 0.028260 | 0.000330 | 29.6 dB | 34.8 dB | 11.84 | 25.1% |

**muP fits with base_width=32** (`w{N}_mup/`) — used by size-gen v2 (train w16) and the deprecated
w16 Experiment 2.
Superseded by base_width=24, which makes the training width the base width.

| Width | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|
| 16 | 28.7 dB | 29.1 dB | 4.0 | 2.5% |
| 24 | 30.8 dB | 31.2 dB | 3.4 | 0.6% |
| 32 | 31.8 dB | 32.1 dB | 3.3 | 0.3% |
| 40 | 32.5 dB | 32.8 dB | 3.1 | 0.1% |
| 48 | 32.9 dB | 33.3 dB | 3.3 | 0.0% |
| 64 | 33.3 dB | 33.8 dB | 3.3 | 0.0% |
| 80 | 33.8 dB | 34.2 dB | 3.1 | 0.0% |
| 96 | 34.0 dB | 34.4 dB | 3.1 | 0.0% |

w32 with base_width=32 is identical to SP w32 (all scaling factors $= 1$). w16 has 2.5% failure due to
$w_m = 16/32 = 0.5$ (readout amplification).

---

### SIREN INRs

SIREN INRs were used in early experiments but abandoned due to:
1. 17% catastrophic failure rate in reconstruction (bimodal quality distribution)
2. Requiring phase canonicalization (extra preprocessing step)
3. Significantly worse classification accuracy (74% vs 96% for ReLU)

Previous SIREN results are preserved in git history (commits before 2026-05-18); those muP
experiments used `symmetry: sign` (wrong for ReLU) and a broken `MuBatchINRFitter` init (uniform
instead of Kaiming He).

### Anydim ScaleGMN (mixed-width training, deprecated)

Mixed-width training on w16-w48 simultaneously. Superseded by size generalization (size-gen v2, itself deprecated)
which more directly tests OOD width generalization.

**SP ReLU**: LR sweep best = 5e-4. Full training: baseline 97.2%, dup-equiv 96.8%.
Runs: [`q88yjyux`](https://wandb.ai/yuxinma/mnist_cls/runs/q88yjyux), [`vfj0yckw`](https://wandb.ai/yuxinma/mnist_cls/runs/vfj0yckw).

**muP ReLU**: LR sweep best = 1.25e-4. Full training: baseline 96.7%, dup-equiv 96.3%.
Runs: [`2tyuemyt`](https://wandb.ai/yuxinma/mnist_cls/runs/2tyuemyt), [`agrl39sa`](https://wandb.ai/yuxinma/mnist_cls/runs/agrl39sa).

### Size Generalization v1 (reused LR from anydim sweep, deprecated)

Used LRs from the anydim (mixed-width) sweep rather than sweeping per condition on the
actual sizegen dataset (w16-32 train only). Superseded by size-gen v2 which sweeps properly.

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 94.5% | 94.7% | 94.2% | 94.2% |
| w24 (IN) | 96.5% | 96.3% | 95.6% | 96.3% |
| w32 (IN) | 96.7% | 96.3% | 96.0% | 96.9% |
| w48 (OUT) | 96.9% | 97.0% | 97.3% | 97.4% |
| w64 (OUT) | 97.3% | 97.6% | 97.3% | 97.8% |
| w80 (OUT) | 96.6% | 97.2% | 97.2% | 97.8% |
| w96 (OUT) | 97.1% | 96.1% | 97.1% | 97.7% |
| **OOD mean** | **96.9%** | **96.9%** | **97.2%** | **97.7%** |

Runs: [`x756d8fn`](https://wandb.ai/yuxinma/mnist_cls/runs/x756d8fn), [`gjxrvpk2`](https://wandb.ai/yuxinma/mnist_cls/runs/gjxrvpk2), [`dznfu9u1`](https://wandb.ai/yuxinma/mnist_cls/runs/dznfu9u1), [`zml68xdj`](https://wandb.ai/yuxinma/mnist_cls/runs/zml68xdj).

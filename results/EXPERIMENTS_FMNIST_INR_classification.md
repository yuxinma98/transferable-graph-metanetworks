# Fashion-MNIST Experiments

## Contents
- [INR Data — size-gen v2](#inr-data--size-gen-v2-train-w32-test-w32w1024) (train w32, test w32–w1024)
- [Experiment 1 — Reproduce ScaleGMN](#experiment-1--reproduce-scalegmn-fmnist-inr-classification) (fixed width w32)
- [Experiment 2 — Size generalization on wider equivalent networks](#experiment-2--size-generalization-on-wider-equivalent-networks-sanity-check)
  (sanity check; `general` vs `uniform` widening $\times$ 11 conditions)
- [Experiment 3 — Size generalization](#experiment-3--size-generalization-train-w32-test-w32w1024)
  (train w32, test w32–w1024) — the main size-generalization result
    - [Plain GMN results](#plain-gmn-results-symmetry-permutation-forward-and-bidirectional)
    - [ScaleGMN results](#scalegmn-results-symmetry-scale-forward-and-bidirectional)
    - [Matrix-Product GMN results](#matrix-product-gmn-results-message_fn_type-matrix_product-forward--bidirectional)
    - [Matrix-Product ScaleGMN results](#matrix-product-scalegmn-results-message_fn_type-matrix_product_scale-forward)
- [Historical notes (deprecated)](#historical-notes-deprecated), incl. the
  [w16-trained size-gen run](#size-generalization-deprecated--train-w16-test-w24w96-non-overlapping-images)

---

## INR Data — size-gen v2 (train w32, test w32–w1024)

**Date**: 2026-08-19.

The layout follows the MNIST size-generalization design (train one width, test many, disjoint
images per width) with **w32 as the training width** instead of MNIST's w24 — FMNIST fits are
worse at every width, and w32 is the smallest width whose fits are acceptable.

**Widths**: train w32; test w32, 48, 64, 80, 96, 128, 192, 256, 384, 512, 768, 1024
(12 widths, up to **32× the training width**). Architecture `[2, w, w, 1]`, 1000 Adam steps.

**Split** (`fmnist_sizegen_splits_v2.json`, seed 20260819, written by
`generate_sizegen_splits.py --from-dataset --create-dirs`): all 70,000 FMNIST images are
shuffled once and partitioned into **56k train + 1k val at w32, plus 1k test images per width**.
Train, val and all test sets are mutually disjoint — no image is ever seen at two widths, and
the script asserts this rather than assuming it. Both parameterizations use the identical
partition, so `_sp` and `_mup32` differ only in how the INRs were fitted.

**w1536 is reserved in the split but not fitted.** The partition was written for 13 widths
including w1536, and its 1k images stay allocated to it in the JSON (removing the width would
shift the cursor and reshuffle every other width's test set, invalidating the fits already on
disk). Fitting stopped at w1024 = 32× the training width, where SP fit quality is already the
binding constraint. `w1536_{sp,mup32}/` therefore exist holding only the split JSON; to add the
width later, fit it — nothing else changes. The driver encodes this as `TEST_WIDTHS` (split
allocation, includes w1536) vs `FIT_WIDTHS` (swept / fitted / reported, stops at w1024).

**Only the evaluated INRs are fitted above w32.** w32 has all 70k images fitted (it supplies
train + val); every larger width holds exactly the 1000 INRs its test set names. Fitting all 70k
at w1024 would be ~290 GB to support a 1k-image test set; the split JSON is written first and
acts as the fitting manifest (`generate_inrs.py --only-from-split`). File names are unaffected:
`canonical_file_ids()` derives them from a full pass over the dataset in its fixed order, so
`fmnist_png_{train,test}/{class}/{index:06d}.pth` denotes the same source image whether the width
was fitted in full or as a subset (verified by matching the split's 58k w32 paths against the
previously-fitted w32 directory).

**Reproduction**:

```bash
source .env
bash scripts/run_fmnist_cls_sizegen_data.sh 4    # arg = GPU id; ~2.5 h, logs in /tmp/fmnist_sizegen_data/
```

### Standard-Param ReLU INRs (per-width fitting LR)

SP needs a per-width LR because the stable LR shrinks as width grows. Each width is swept with
`sweep_inr_lr.py --auto-extend` (500 freshly-fitted samples per candidate, 1000 steps) under one
rule applied identically everywhere: **minimize failure rate ($\text{MSE} > 0.01$), ties broken on higher
mean PSNR**. `--auto-extend` keeps halving/doubling the grid until the winner is an *interior*
point, so no width's LR sits at the edge of its search.

**LR sweeps** (PSNR dB / Fail%, selected LR in bold; every width's winner came out interior, so
no sweep hit an `AT GRID EDGE` warning). w32: 0.0025 22.2/19.6%, 0.005 24.3/7.2%,
**0.01 25.7/4.2%**, 0.02 26.4/4.4%. w48: 0.0025 23.5/10.6%, 0.005 25.5/4.4%, 0.01 26.7/3.4%,
**0.02 27.3/3.4%**, 0.04 27.3/4.6% (grid extended to 0.04). w64: 0.0025 24.3/8.4%,
0.005 26.1/3.2%, 0.01 27.4/3.4%, **0.02 28.0/2.6%**, 0.04 27.0/9.8% (extended). w80:
0.0025 24.8/7.0%, 0.005 26.5/3.6%, **0.01 27.6/2.6%**, 0.02 28.2/3.0%. w96: 0.0025 25.2/5.2%,
0.005 26.8/3.0%, **0.01 28.0/1.8%**, 0.02 28.3/4.0%. w128: 0.001 22.7/16.2%, 0.002 25.1/7.2%,
0.005 27.2/3.6%, **0.01 28.1/3.4%**, 0.02 27.6/9.6% (extended). w192: 0.001 23.5/11.6%,
0.002 25.7/5.6%, **0.005 27.3/4.6%**, 0.01 27.6/8.2%. w256: 0.001 24.0/10.4%, 0.002 25.9/6.0%,
**0.005 27.1/5.6%**, 0.01 26.2/15.4%. w384: 0.00025 19.8/53.8%, 0.0005 22.5/20.2%,
0.001 24.5/9.6%, **0.002 25.9/7.6%**, 0.004 26.0/10.6% (extended). w512: 0.00025 20.3/47.4%,
0.0005 22.8/17.6%, 0.001 24.6/10.2%, **0.002 25.5/9.6%**, 0.004 24.0/20.6% (extended). w768:
0.00025 21.0/38.0%, 0.0005 23.1/18.0%, **0.001 24.4/11.6%**, 0.002 23.0/22.0%. w1024:
0.000125 18.8/69.6%, 0.00025 21.2/35.6%, 0.0005 23.0/19.6%, **0.001 23.4/17.0%**,
0.002 20.1/39.6% (extended).

**Reconstruction quality of the fitted INRs** (`verify_inr_quality.py --dataset fmnist
--init-type sp`, 1000 sampled fits per width, "Fail%" = fraction with $\text{MSE} > 0.01$):

| Width | LR | MSE mean | MSE median | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|---|---|---|
| 32 | 0.01 | 0.003846 | 0.002771 | 25.5 dB | 25.6 dB | 3.3 | 5.9% |
| 48 | 0.02 | 0.002640 | 0.001777 | 27.3 dB | 27.5 dB | 3.6 | 2.7% |
| 64 | 0.02 | 0.002483 | 0.001645 | 27.7 dB | 27.8 dB | 3.6 | 2.8% |
| 80 | 0.01 | 0.002545 | 0.001738 | 27.6 dB | 27.6 dB | 3.7 | 2.3% |
| 96 | 0.01 | 0.002409 | 0.001627 | 27.9 dB | 27.9 dB | 3.8 | 2.2% |
| 128 | 0.01 | 0.002713 | 0.001530 | 28.0 dB | 28.2 dB | 4.2 | 3.9% |
| 192 | 0.005 | 0.003039 | 0.001951 | 27.1 dB | 27.1 dB | 3.8 | 4.3% |
| 256 | 0.005 | 0.003600 | 0.001898 | 27.0 dB | 27.2 dB | 4.4 | 5.4% |
| 384 | 0.002 | 0.004189 | 0.002720 | 25.6 dB | 25.7 dB | 3.8 | 7.5% |
| 512 | 0.002 | 0.005846 | 0.002810 | 25.1 dB | 25.5 dB | 4.3 | 10.9% |
| 768 | 0.001 | 0.006571 | 0.003918 | 24.1 dB | 24.1 dB | 4.1 | 13.6% |
| 1024 | 0.001 | 0.008688 | 0.004300 | 23.3 dB | 23.7 dB | 4.5 | 19.5% |

### muP ReLU INRs (base_width=32)

muP uses **base_width=32 = the training width**, so w32 muP is mathematically identical to w32 SP
($\text{width\_mult} = 1$: no init rescaling, no per-layer LR rescaling).

The LR is swept **once at w32 and reused at every width**: `get_mup_optimizer` rescales per-layer
LRs by $\text{base\_width}/\text{width}$, so the base LR transfers across widths without a
per-width sweep.

**LR sweep at w32** (muP fitter, base_width 32): 0.0025 22.2 dB/19.6%, 0.005 24.3/7.2%,
**0.01 25.7/4.2%** (selected, interior), 0.02 26.4/4.4% — identical at every candidate to the SP
w32 sweep, as $\text{width\_mult} = 1$ requires. **lr = 0.01 is used at all 12 widths.**

**Reconstruction quality of the fitted INRs** (`verify_inr_quality.py --dataset fmnist
--init-type mup32`, 1000 sampled fits per width, lr = 0.01 at every width):

| Width | MSE mean | MSE median | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|---|---|
| 32 | 0.003796 | 0.002802 | 25.5 dB | 25.5 dB | 3.2 | 4.8% |
| 48 | 0.003313 | 0.002393 | 26.1 dB | 26.2 dB | 3.3 | 4.4% |
| 64 | 0.002923 | 0.002096 | 26.6 dB | 26.8 dB | 3.2 | 3.4% |
| 80 | 0.002811 | 0.001994 | 26.9 dB | 27.0 dB | 3.4 | 3.0% |
| 96 | 0.002625 | 0.001885 | 27.2 dB | 27.2 dB | 3.3 | 2.3% |
| 128 | 0.002490 | 0.001777 | 27.3 dB | 27.5 dB | 3.2 | 1.8% |
| 192 | 0.002519 | 0.001817 | 27.3 dB | 27.4 dB | 3.2 | 2.4% |
| 256 | 0.002486 | 0.001711 | 27.4 dB | 27.7 dB | 3.4 | 1.9% |
| 384 | 0.002385 | 0.001740 | 27.5 dB | 27.6 dB | 3.1 | 1.5% |
| 512 | 0.002382 | 0.001803 | 27.4 dB | 27.4 dB | 3.0 | 1.7% |
| 768 | 0.002581 | 0.001778 | 27.4 dB | 27.5 dB | 3.4 | 2.2% |
| 1024 | 0.002606 | 0.001814 | 27.3 dB | 27.4 dB | 3.2 | 2.7% |

---

## Experiment 1 — Reproduce ScaleGMN (FMNIST-INR classification)

**Dataset**: SP ReLU INRs from `w32_sp/`, width 32, Architecture `[2, 32, 32, 1]`. Split: 55k/5k/10k
(`fmnist_splits.json`, generated via `scripts/generate_splits.py`).

### Reproduction

```bash
source .env

# ScaleGMN (forward + bidirectional)
screen -dmS scalegmn_fw bash -c 'source .env && CUDA_VISIBLE_DEVICES=0 \
  python scripts/train_scalegmn.py --conf configs/fmnist_cls/scalegmn_reproduce_sp.yml --wandb True'
screen -dmS scalegmn_bidir bash -c 'source .env && CUDA_VISIBLE_DEVICES=1 \
  python scripts/train_scalegmn.py --conf configs/fmnist_cls/scalegmn_reproduce_sp_bidir.yml --wandb True'

# Plain GMN (forward)
screen -dmS gmn_fw bash -c 'source .env && CUDA_VISIBLE_DEVICES=2 \
  python scripts/train_scalegmn.py --conf configs/fmnist_cls/gmn_reproduce_sp.yml --wandb True'
```

### Results

| Variant | Best val acc | Best test acc | Config | Run ID |
|---|---|---|---|---|
| ScaleGMN (forward, scale) | 0.8386 | 0.8322 | `scalegmn_reproduce_sp.yml` | [`9dhqd6tg`](https://wandb.ai/yuxinma/fmnist_cls/runs/9dhqd6tg) |
| ScaleGMN-B (bidirectional, scale) | 0.8280 | 0.8301 | `scalegmn_reproduce_sp_bidir.yml` | [`481rncf8`](https://wandb.ai/yuxinma/fmnist_cls/runs/481rncf8) |
| GMN (forward, permutation) | 0.8262 | 0.8225 | `gmn_reproduce_sp.yml` | [`qks56yj7`](https://wandb.ai/yuxinma/fmnist_cls/runs/qks56yj7) |

ScaleGMN-B (bidirectional) uses the reciprocal backward edge feature $1/W[u,v]$ from
`src/models/bidir_reciprocal.py`, required for scale equivariance in a bidirectional model — see
[Bidirectional results (Experiment 3)](#scalegmn-results-symmetry-scale-forward-and-bidirectional).
Log: `/tmp/fmnist_sizegen_v2_bidir_recip/reproduce_exp1-sgmn-b-recip.log`.

---

## Experiment 2 — Size generalization on wider equivalent networks (sanity check)

**Goal**: Take the w32 test INRs, widen them to w64, w128, w192, w256, w320 by a *function-preserving*
construction, and measure size generalization. Because the widened INR computes exactly the same
function as its w32 source, every accuracy change is the metanetwork failing to be invariant to the
widening — never distribution shift in the underlying images. Compare against
[Experiment 3](#experiment-3--size-generalization-train-w32-test-w32w1024), where the wider inputs are
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
bidirectional ScaleGMN uses the `-recip` retrains. Widening factors are $k = 2, 4, 6, 8, 10$, matching
[MNIST Experiment 2](EXPERIMENTS_MNIST_INR_classification.md#experiment-2--size-generalization-on-wider-equivalent-networks-sanity-check)
off its w24 base, and every base-width cell reproduces that condition's w32 accuracy in Experiment 3
exactly.

**No SP / muP distinction here**, so only the muP32 arm is tabulated. The parameterization only affects
how INRs of *different* widths are fitted, and no wide INR is ever fitted in this experiment: at
`width == base_width` muP reduces exactly to SP ($w_m = 1$, so both the per-layer LR rescaling and the
$\sqrt{w_m}$ output scaling are the identity), and both w32 draws were in fact fitted at lr $=0.01$ to
the same PSNR mean of $25.5$ dB. The SP arm is therefore a **replicate**, differing only in the INR
fitting RNG and in the metanetwork checkpoint trained on that draw: it was scored in the same batch
([`q2bfqjdr`](https://wandb.ai/yuxinma/fmnist_cls/runs/q2bfqjdr),
[`ir63i8tj`](https://wandb.ai/yuxinma/fmnist_cls/runs/ir63i8tj) for `general` forward/bidirectional;
[`0kp83h9s`](https://wandb.ai/yuxinma/fmnist_cls/runs/0kp83h9s),
[`rwa1fnvj`](https://wandb.ai/yuxinma/fmnist_cls/runs/rwa1fnvj) for `uniform`).
It marks exactly the same conditions invariant, family by family and direction by direction,
agreeing to $\le 2.8$pp on those cells — its checkpoints' own w32
accuracies already differ by up to $4.8$pp — and to $\le 21.0$pp elsewhere.

**Function preservation, measured**: `generate_duplicated_inrs.py --verify 25` reports, over w64–w320
and both arms, $\max|\Delta\text{pixels}| \le 3.2\times10^{-6}$ and
$\max|\Delta\text{logits}| \le 1.9\times10^{-5}$ against the w32 source — the same order as the
`uniform` control ($5.0\times10^{-7}$ / $3.6\times10^{-6}$), i.e. pure float32 storage rounding. Blocks
are drawn in float64 and cast on write; the float32 cancellation problem that forces float64 for *conv*
general widenings does not arise for MLP INRs. Row residual $\le 4.1\times10^{-7}$; column residual
$4.55$–$15.1$ and minimum non-uniformity $2.2$–$3.9$ confirm `general` genuinely violates the column
condition and never degenerates to the uniform block.

**Scripts**: `scripts/generate_duplicated_inrs.py --dataset fmnist --family` (data),
`scripts/eval_sizegen_duplicated.py --dataset fmnist` (eval),
`scripts/run_fmnist_cls_widen_families_eval.sh` (orchestration).

**Date**: 2026-09-12. The w16-trained size-generalization run that previously occupied this slot is a
different experiment and is in
[Historical notes](#size-generalization-deprecated--train-w16-test-w24w96-non-overlapping-images).

### Reproduction

```bash
source .env

# Data: widen the w32 test INRs under each family (~0.9 GB per family per arm)
for INIT in mup32 sp; do for FAM in general uniform; do
  python scripts/generate_duplicated_inrs.py --dataset fmnist --init-type $INIT --base-width 32 \
      --target-widths 64 128 192 256 320 --family $FAM \
      --split-name fmnist_sizegen_splits_v2.json --seed 0 --verify 25
done; done

# Eval: 8 jobs = {general,uniform} x {mup32,sp} x {forward,bidirectional}, one worker per GPU
screen -dmS fmnist_widen_eval bash -c 'bash scripts/run_fmnist_cls_widen_families_eval.sh "0"'
```

Eval batch size is scaled as `bs · (32/w)²` to keep memory flat out to w320.

### `general` widening — the largest function-preserving family

Eval runs [`qopgywyr`](https://wandb.ai/yuxinma/fmnist_cls/runs/qopgywyr) (forward) and
[`cflvwd98`](https://wandb.ai/yuxinma/fmnist_cls/runs/cflvwd98) (bidirectional), extended
2026-09-15/16 to w96/w384/w512/w768/w1024 (superseding the earlier w64–w320 run of the same name).

#### ScaleGMN

| Condition | w32 | w64 | w96 | w128 | w192 | w256 | w320 | w384 | w512 | w768 | w1024 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 80.7% | 69.4% | 56.0% | 48.1% | 41.2% | 36.3% | 33.9% | 32.6% | 28.2% | 26.1% | 24.8% | 55.9% |
| ScaleGMN baseline, bd | 84.0% | 73.3% | 62.0% | 57.0% | 48.5% | 42.4% | 35.7% | 31.7% | 24.3% | 18.3% | 15.6% | 68.4% |
| ScaleGMN dup-equiv, fw | 82.4% | 68.0% | 48.5% | 38.1% | 28.6% | 26.6% | 25.7% | 25.1% | 24.9% | 24.3% | 24.5% | 58.1% |
| ScaleGMN dup-equiv, bd | 81.5% | 61.9% | 44.7% | 33.4% | 25.0% | 23.2% | 22.5% | 21.7% | 21.7% | 21.4% | 21.1% | 60.4% |
| **mp-ScaleGMN dup-equiv, fw** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **0.0%** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | — | out of scope |

#### Plain GMN

| Condition | w32 | w64 | w96 | w128 | w192 | w256 | w320 | w384 | w512 | w768 | w1024 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | 79.3% | 59.3% | 52.6% | 50.7% | 48.5% | 47.8% | 47.8% | 47.2% | 47.0% | 46.6% | 46.4% | 32.9% |
| GMN baseline, bd | 77.1% | 45.9% | 29.6% | 24.1% | 19.1% | 18.3% | 18.3% | 18.0% | 17.3% | 17.0% | 16.8% | 60.3% |
| GMN dup-equiv, fw | 81.0% | 34.2% | 20.1% | 17.7% | 16.6% | 16.1% | 16.1% | 16.0% | 15.9% | 15.7% | 15.7% | 65.3% |
| GMN dup-equiv, bd | 79.1% | 25.3% | 17.4% | 15.0% | 13.0% | 12.4% | 12.0% | 11.9% | 11.5% | 11.4% | 11.4% | 67.7% |
| **mp-GMN dup-equiv, fw** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **0.0%** |
| mp-GMN dup-equiv, bd | 81.1% | 65.4% | 47.2% | 39.2% | 28.6% | 24.0% | 22.5% | 20.9% | 19.8% | 17.5% | 16.4% | 64.7% |

### `uniform` (Kronecker) widening — the reference family

Eval runs [`kofv48ug`](https://wandb.ai/yuxinma/fmnist_cls/runs/kofv48ug) (forward) and
[`s26aqs0x`](https://wandb.ai/yuxinma/fmnist_cls/runs/s26aqs0x) (bidirectional), extended
2026-09-15/16 to w96/w384/w512/w768/w1024 (superseding the earlier w64–w320 run of the same name).

#### ScaleGMN

| Condition | w32 | w64 | w96 | w128 | w192 | w256 | w320 | w384 | w512 | w768 | w1024 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ScaleGMN baseline, fw | 80.7% | 80.0% | 79.0% | 77.9% | 76.0% | 74.9% | 74.1% | 73.6% | 72.2% | 71.7% | 71.3% | 9.4% |
| ScaleGMN baseline, bd | 84.0% | 80.8% | 76.7% | 71.7% | 49.9% | 26.8% | 15.3% | 12.6% | 11.9% | 11.3% | 10.6% | 73.4% |
| **ScaleGMN dup-equiv, fw** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **82.4%** | **0.0%** |
| **ScaleGMN dup-equiv, bd** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **81.5%** | **0.0%** |
| **mp-ScaleGMN dup-equiv, fw** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **82.6%** | **0.0%** |
| mp-ScaleGMN dup-equiv, bd | — | — | — | — | — | — | — | — | — | — | — | out of scope |

#### Plain GMN

| Condition | w32 | w64 | w96 | w128 | w192 | w256 | w320 | w384 | w512 | w768 | w1024 | max_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GMN baseline, fw | 79.3% | 67.5% | 53.9% | 46.9% | 38.9% | 34.4% | 32.5% | 32.3% | 31.4% | 31.1% | 30.7% | 48.6% |
| GMN baseline, bd | 77.1% | 36.5% | 10.8% | 10.8% | 10.8% | 10.8% | 10.8% | 10.8% | 10.8% | 10.8% | 10.9% | 66.3% |
| **GMN dup-equiv, fw** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **81.0%** | **0.0%** |
| **GMN dup-equiv, bd** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **79.1%** | **0.0%** |
| **mp-GMN dup-equiv, fw** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **82.2%** | **0.0%** |
| **mp-GMN dup-equiv, bd** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **81.1%** | **0.0%** |

### Figures

`scripts/plot_widen_families.py --dataset fmnist` (reads the tables above):

| Figure | Contents |
|---|---|
| `results/figures/widen_fmnist_gmn.png` | GMN baseline / dup-equiv / mp-GMN, `general` \| `uniform` panels |
| `results/figures/widen_fmnist_scalegmn.png` | ScaleGMN baseline / dup-equiv / mp-ScaleGMN, same panels |

Solid forward, dashed bidirectional; colour + marker carry the model, matching every other figure in
`results/figures/`.

---

## Experiment 3 — Size Generalization (train w32, test w32–w1024)

**Goal**: The FMNIST counterpart of
[MNIST Experiment 3](EXPERIMENTS_MNIST_INR_classification.md#experiment-3--size-generalization-train-w24-test-w24w1024):
measure how well a metanetwork trained at a single INR width transfers to unseen wider INRs, and
how much of that transfer comes from (i) the parameterization the INRs were fitted with and
(ii) the duplication-equivariant modifications of the metanetwork.

**Setup**, matching MNIST Experiment 3 with w32 in place of w24 as the training width:

- Train on **w32** only — the smallest FMNIST width whose fits are good enough that OOD
  degradation cannot be blamed on a poorly-fitted training set.
- muP uses **`base_width=32`** = the training width, so w32 muP $\equiv$ w32 SP exactly and the two
  data conditions differ only at the OOD widths.
- **Every test set uses disjoint images**, from each other and from train/val, so no image is ever
  seen at two widths. Architecture `[2, w, w, 1]` throughout.
- Test widths run out to **w1024 = 32× the training width** (MNIST reached 42.7×).

**Direction.** Each of the four model sections below (Plain GMN, ScaleGMN, Matrix-Product GMN,
Matrix-Product ScaleGMN) reports `--direction forward` first. The bidirectional counterparts of
the same conditions were launched on 2026-08-21 (`run_fmnist_cls_sizegen_v2_bidir.sh`, same LR
grid / epochs / patience / splits, so they are drop-in columns) and are reported as a
`#### Bidirectional` subsection immediately under each model's forward results — matrix-product
GMN reports both directions as columns in one table, and matrix-product ScaleGMN is forward-only
by scope, so it has no bidirectional subsection.

**Bidirectional status**: all 10 conditions finished 2026-08-27 19:33 (muP32 arm — the controlled
comparison — 2026-08-26 04:06, SP arm 2026-08-27 19:33). Measured cost with 2–3 workers: ~9–13 h
per condition (sweep 3.5–5 h + training 7–9 h + eval ~30 min), ~130 GPU-h for the arm.

### Data

- SP ReLU: `w{N}_sp/` (Kaiming He init, per-width swept fitting LR, 1000 steps)
- muP32: `w{N}_mup32/` (base_width=32, muP optimizer, fixed lr swept once at w32, 1000 steps)

Both read `fmnist_sizegen_splits_v2.json`: 56k train + 1k val at w32, 1k disjoint test images at
each of the 12 widths. See [INR Data](#inr-data--size-gen-v2-train-w32-test-w32w1024) above for how
the split and the fits were produced, including why only the ~1k evaluated INRs are fitted above
w32. Unlike MNIST — where the large widths were bolted on afterwards via a second split file
(`mnist_sizegen_splits_v4.json`) and an eval-only re-score — the FMNIST split covered all 12 widths
from the start, so a single split file serves both stages here.

> **Caveat for reading the SP columns.** SP fit quality peaks at w96–w128 (27.9–28.0 dB, 2–4%
> failures) and degrades monotonically to **23.3 dB / 19.5% failures at w1024**. muP32 holds
> 27.3–27.5 dB at 1.5–2.7% failures from w128 out to w1024 — a 4.0 dB / 16.8pp gap at w1024 on the
> same images. MNIST's SP still manages 29.5 dB / 3.2% at w1024.

### Model and training setup

**Ten conditions**: {ScaleGMN (`symmetry: scale`), plain GMN (`symmetry: permutation`)} ×
{SP, muP32} × {baseline, dup-equiv} = 8, plus the **matrix-product GMN**
(`message_fn_type: matrix_product`) × {SP, muP32} = 2. The matrix-product GMN is dup-equiv only:
its MSG function (`edge_attr * A(x_j)` on the raw scalar weight) only becomes the matrix product
$Z^{(l)} = W^{(l)} H^{(l-1)} A$ on top of fan-in rescaling + mean aggregation + layer-wise mean
readout, so there is no baseline variant.

Baseline = `aggregator: add`, raw edge weights, sum-pooling readout.
Dup-equiv = `aggregator: mean`, fan-in rescaled edges ($d_{l-1} \cdot W^{(l)}$), `LayerWiseMeanReadout`.
All variants: d_hid=128, 4 GNN layers, `readout_range: full_graph`, forward direction.

**Training**: batch_size=64 (875 steps/epoch), AdamW, warmup 1000 steps, 200 epochs, patience=50.
Eval batch size scales as $\text{bs} \cdot (32^2 / w^2)$ to bound memory at large widths (batch 1
from w256 on).

**Metanetwork LR protocol**: a 7-value sweep (6.25e-5 to 4e-3, ×2 grid, 8600 steps ≈ 10 epochs) per
condition, selected on **best val accuracy at w32** — in-distribution only, so no OOD width
influences model selection.

**Two-stage evaluation**: during training only w32–w256 are scored per epoch (every test width is
evaluated every epoch, and w384–w1024 at batch size 1 would dominate the epoch time); the best
checkpoint is then re-scored once over the full w32–w1024 range via `--eval-only-ckpt` with the
matching `*_v2_all.yml` config. Both stages use the same split file, so the per-width numbers are
directly comparable and no training image appears in any test set.

**Date**: launched 2026-08-20, **all 10 conditions finished 2026-08-21 00:10**
(~7 h per condition; two GPUs draining the queue).

### Reproduction

```bash
source .env

# Data (already generated; ~2.5 h):
bash scripts/run_fmnist_cls_sizegen_data.sh 4

# All 10 conditions: sweep -> full training -> w32-w1024 re-score, one worker per GPU,
# resumable (re-run to ADD a worker; delete /tmp/fmnist_sizegen_v2/queue.txt to reset):
screen -dmS fmnist_sizegen bash -c 'bash scripts/run_fmnist_cls_sizegen_v2.sh "5"'

# Add a worker automatically once a busy GPU of yours goes idle:
screen -dmS fmnist_sizegen_w2 bash -c 'bash scripts/add_worker_when_free.sh 1 \
    scripts/run_fmnist_cls_sizegen_v2.sh'

# Bidirectional counterparts of the same 10 conditions (same protocol, same splits):
screen -dmS fmnist_bidir bash -c 'bash scripts/run_fmnist_cls_sizegen_v2_bidir.sh "1"'
screen -dmS fmnist_bidir_w2 bash -c 'bash scripts/add_worker_when_free.sh 0 \
    scripts/run_fmnist_cls_sizegen_v2_bidir.sh'

# Matrix-product ScaleGMN, 2 conditions ({SP, muP32}, dup-equiv, forward). An empty GPU
# list seeds the queue without starting a worker, for when the GPU budget is full:
bash scripts/run_fmnist_cls_mpsgmn_sizegen_v2.sh ""
```

**Configs**: `configs/fmnist_cls/{scalegmn,gmn,mpgmn,mpsgmn}_sizegen_{sp,mup32}_v2.yml` for training
(test w32–w256); the `_v2_all.yml` variants are identical except for the four extra test widths.
**Logs**: `/tmp/fmnist_sizegen_v2/` (and `/tmp/mpsgmn_fmnist_v2/` for the matrix-product ScaleGMN)
— incremental `SUMMARY.md`, plus `sweep_*`, `full_*` and `eval_all_*` logs per condition. The queue runs the muP32 block first (the controlled comparison),
then SP.

### Plain GMN Results (`symmetry: permutation`, forward and bidirectional)

#### Forward

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.7200 / 0.7390 / **0.7540** / 0.7380 / 0.7530 / 0.7290 / 0.7210 → best **2.5e-4**
- SP dup-equiv: 0.7150 / 0.7350 / **0.7380** / 0.7360 / 0.7200 / 0.7040 / 0.6850 → best **2.5e-4**
- muP32 baseline: 0.7120 / 0.7300 / **0.7320** / 0.7290 / 0.7260 / 0.7260 / 0.7020 → best **2.5e-4**
- muP32 dup-equiv: 0.6990 / **0.7330** / 0.7180 / 0.7060 / 0.7190 / 0.7130 / 0.6850 → best **1.25e-4**

**Per-width test accuracy, w32 → w1024** (eval-only re-score, 1k disjoint test images per width;
row-best in bold):

| Width | SP baseline | SP dup-equiv | muP32 baseline | muP32 dup-equiv |
|---|---|---|---|---|
| w32 (IN) | 79.4% | **82.6%** | 79.3% | 81.0% |
| w48 | 79.2% | 81.5% | 77.7% | **81.8%** |
| w64 | 76.0% | 78.7% | 70.6% | **83.2%** |
| w80 | 74.7% | 79.6% | 62.3% | **84.0%** |
| w96 | 62.6% | 76.6% | 56.1% | **84.4%** |
| w128 | 48.4% | 68.0% | 49.1% | **83.6%** |
| w192 | 28.5% | 46.9% | 40.2% | **77.4%** |
| w256 | 24.1% | 30.3% | 36.9% | **74.9%** |
| w384 | 16.6% | 12.0% | 33.4% | **63.2%** |
| w512 | 19.0% | 10.9% | 32.8% | **59.9%** |
| w768 | 16.7% | 8.7% | 33.7% | **48.4%** |
| w1024 | 15.1% | 11.4% | 34.9% | **38.0%** |
| **OOD mean** (w48–w1024) | 41.9% | 45.9% | 48.0% | **70.8%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-fmnist-v2-gmn-sp-base_4595gtdh` | [`7uga44nb`](https://wandb.ai/yuxinma/fmnist_cls/runs/7uga44nb) |
| SP dup-equiv | `sizegen-fmnist-v2-gmn-sp-deq_ubsaeszh` | [`r4joqvjo`](https://wandb.ai/yuxinma/fmnist_cls/runs/r4joqvjo) |
| muP32 baseline | `sizegen-fmnist-v2-gmn-mup32-base_yyl5ku0p` | [`06i261kt`](https://wandb.ai/yuxinma/fmnist_cls/runs/06i261kt) |
| muP32 dup-equiv | `sizegen-fmnist-v2-gmn-mup32-deq_rm86w1di` | [`eece0qj7`](https://wandb.ai/yuxinma/fmnist_cls/runs/eece0qj7) |

**Training runs** (200 epochs, patience=50; all four early-stopped):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`4595gtdh`](https://wandb.ai/yuxinma/fmnist_cls/runs/4595gtdh) | 2.5e-4 | 107 | 81.9% |
| SP dup-equiv | [`ubsaeszh`](https://wandb.ai/yuxinma/fmnist_cls/runs/ubsaeszh) | 2.5e-4 | 129 | 82.7% |
| muP32 baseline | [`yyl5ku0p`](https://wandb.ai/yuxinma/fmnist_cls/runs/yyl5ku0p) | 2.5e-4 | 117 | 81.4% |
| muP32 dup-equiv | [`rm86w1di`](https://wandb.ai/yuxinma/fmnist_cls/runs/rm86w1di) | 1.25e-4 | 128 | 82.2% |

#### Bidirectional

Same protocol as the forward block above, with `direction: bidirectional`. Run from the same
`--direction bidirectional` queue as the ScaleGMN bidirectional block below: identical LR grid,
epochs, patience and split files, so these are drop-in columns against the forward table above.

**Date**: muP32 arm 2026-08-26 04:06, SP arm 2026-08-27 19:33.
**Launch**: `scripts/run_fmnist_cls_sizegen_v2_bidir.sh`.
**LR sweep**: selected LR per condition is in the training-run table below; full sweep curves are
in `/tmp/fmnist_sizegen_v2_bidir/SUMMARY.md`.

**Per-width test accuracy, w32 → w1024** (eval-only re-score of the best checkpoint, 1k disjoint
test images per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP32 baseline | muP32 dup-equiv |
|---|---|---|---|---|
| w32 (IN) | **81.9%** | 80.9% | 77.1% | 79.1% |
| w48 | 71.8% | 78.9% | 71.8% | **82.1%** |
| w64 | 47.9% | 79.7% | 37.7% | **83.4%** |
| w80 | 14.7% | 78.8% | 12.2% | **82.6%** |
| w96 | 11.3% | 77.3% | 10.7% | **82.6%** |
| w128 | 13.2% | 68.8% | 10.7% | **81.2%** |
| w192 | 11.1% | 44.4% | 9.5% | **75.3%** |
| w256 | 10.3% | 40.6% | 12.4% | **70.4%** |
| w384 | 10.6% | 21.0% | 9.9% | **50.1%** |
| w512 | 10.4% | 23.6% | 9.0% | **48.4%** |
| w768 | 11.2% | 27.4% | 9.9% | **42.1%** |
| w1024 | 9.9% | 30.0% | 10.6% | **33.9%** |
| **OOD mean** (w48–w1024) | 20.2% | 51.9% | 18.6% | **66.6%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-fmnist-v2-gmn-sp-base-bidir_hw8n6bql` | [`6afvjnm4`](https://wandb.ai/yuxinma/fmnist_cls/runs/6afvjnm4) |
| SP dup-equiv | `sizegen-fmnist-v2-gmn-sp-deq-bidir_tvl5gdjl` | [`vjq14mnu`](https://wandb.ai/yuxinma/fmnist_cls/runs/vjq14mnu) |
| muP32 baseline | `sizegen-fmnist-v2-gmn-mup32-base-bidir_dlukne93` | [`i37rzn31`](https://wandb.ai/yuxinma/fmnist_cls/runs/i37rzn31) |
| muP32 dup-equiv | `sizegen-fmnist-v2-gmn-mup32-deq-bidir_3saj2l9f` | [`89hpe6n6`](https://wandb.ai/yuxinma/fmnist_cls/runs/89hpe6n6) |

**Training runs** (200 epochs, patience=50):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`hw8n6bql`](https://wandb.ai/yuxinma/fmnist_cls/runs/hw8n6bql) | 1.25e-4 | 109 | 81.6% |
| SP dup-equiv | [`tvl5gdjl`](https://wandb.ai/yuxinma/fmnist_cls/runs/tvl5gdjl) | 5e-4 | 104 | 82.1% |
| muP32 baseline | [`dlukne93`](https://wandb.ai/yuxinma/fmnist_cls/runs/dlukne93) | 2.5e-4 | 172 | 81.5% |
| muP32 dup-equiv | [`3saj2l9f`](https://wandb.ai/yuxinma/fmnist_cls/runs/3saj2l9f) | 2.5e-4 | 119 | 81.8% |

### ScaleGMN Results (`symmetry: scale`, forward and bidirectional)

#### Forward

**LR sweep** (7 LRs from 6.25e-5 to 4e-3, ×2 grid, 8600 steps each; val acc at
6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP baseline: 0.7940 / 0.7980 / 0.8120 / 0.8100 / **0.8150** / 0.7990 / 0.7940 → best **1e-3**
- SP dup-equiv: 0.8090 / **0.8220** / **0.8220** / 0.8120 / 0.8210 / 0.8070 / 0.8140 → best **1.25e-4**
  (tied with 2.5e-4; argmax picked the smaller)
- muP32 baseline: 0.7940 / 0.8130 / **0.8260** / 0.8160 / 0.8130 / 0.8020 / 0.7980 → best **2.5e-4**
- muP32 dup-equiv: 0.8030 / 0.8180 / **0.8250** / 0.8240 / 0.8200 / 0.8220 / 0.7930 → best **2.5e-4**

**Per-width test accuracy, w32 → w1024** (eval-only re-score of the best checkpoint, 1k disjoint
test images per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP32 baseline | muP32 dup-equiv |
|---|---|---|---|---|
| w32 (IN) | 84.2% | **84.4%** | 80.7% | 82.4% |
| w48 | 84.5% | 82.1% | 82.8% | **85.2%** |
| w64 | **84.8%** | 83.3% | 84.1% | 84.6% |
| w80 | 83.4% | 83.0% | 83.0% | **85.9%** |
| w96 | 85.9% | 83.3% | 86.5% | **86.9%** |
| w128 | 83.1% | 81.8% | 84.9% | **87.7%** |
| w192 | 76.6% | 78.5% | 80.9% | **87.9%** |
| w256 | 72.4% | 74.6% | 78.9% | **87.5%** |
| w384 | 52.9% | 57.9% | 75.8% | **85.6%** |
| w512 | 42.7% | 50.2% | 74.3% | **86.8%** |
| w768 | 22.7% | 33.1% | 65.6% | **86.5%** |
| w1024 | 20.8% | 19.8% | 68.0% | **84.0%** |
| **OOD mean** (w48–w1024) | 64.5% | 66.1% | 78.6% | **86.2%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-fmnist-v2-sgmn-sp-base_jbd9luuz` | [`s9k7zaeu`](https://wandb.ai/yuxinma/fmnist_cls/runs/s9k7zaeu) |
| SP dup-equiv | `sizegen-fmnist-v2-sgmn-sp-deq_f33nszbu` | [`lmcrz6x6`](https://wandb.ai/yuxinma/fmnist_cls/runs/lmcrz6x6) |
| muP32 baseline | `sizegen-fmnist-v2-sgmn-mup32-base_4e850ho2` | [`sdpfhomw`](https://wandb.ai/yuxinma/fmnist_cls/runs/sdpfhomw) |
| muP32 dup-equiv | `sizegen-fmnist-v2-sgmn-mup32-deq_go46ekpy` | [`78281i4c`](https://wandb.ai/yuxinma/fmnist_cls/runs/78281i4c) |

**Training runs** (200 epochs, patience=50; all four early-stopped):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`jbd9luuz`](https://wandb.ai/yuxinma/fmnist_cls/runs/jbd9luuz) | 1e-3 | 71 | 84.3% |
| SP dup-equiv | [`f33nszbu`](https://wandb.ai/yuxinma/fmnist_cls/runs/f33nszbu) | 1.25e-4 | 71 | 83.8% |
| muP32 baseline | [`4e850ho2`](https://wandb.ai/yuxinma/fmnist_cls/runs/4e850ho2) | 2.5e-4 | 86 | 84.0% |
| muP32 dup-equiv | [`go46ekpy`](https://wandb.ai/yuxinma/fmnist_cls/runs/go46ekpy) | 2.5e-4 | 97 | 83.5% |

#### Bidirectional

Same eight-condition-worth of protocol as the forward runs, with `direction: bidirectional`.

> A bidirectional ScaleGMN requires the reciprocal backward edge feature $1/W[u,v]$: without it a
> backward message into `u` carries $\lambda_v^2/\lambda_u$ where equivariance requires $\lambda_u$,
> so there is no scale equivariance. The four conditions below use `src/models/bidir_reciprocal.py`,
> which applies this feature (`ScaleGMN_GNN_bidir.forward` otherwise hardcodes
> `bw_edge_attr = batch.edge_attr`); without the fix the equivariance check fails by 5–6 orders of
> magnitude, with it the same weights are equivariant to 4.1e-7 (node) / 3.0e-7 (graph), per
> [VERIFICATION_gmn_properties.md § Verification of scale equivariance](VERIFICATION_gmn_properties.md#verification-of-scale-equivariance).
> Launched 2026-08-29 01:58 via `scripts/run_fmnist_cls_scalegmn_bidir_recip.sh`; all five jobs
> (including the Experiment 1 rerun) finished 2026-08-30 20:07.

**Date**: muP32 arm 2026-08-26 04:06, SP arm 2026-08-27 19:33.
**Logs**: `/tmp/fmnist_sizegen_v2_bidir_recip/` (incremental `SUMMARY.md`, per-run sweep / full /
eval logs).
**LR sweep**: selected LR per condition is in the training-run table below; full sweep curves are
in `/tmp/fmnist_sizegen_v2_bidir_recip/SUMMARY.md`.

**Per-width test accuracy, w32 → w1024** (eval-only re-score of the best checkpoint, 1k disjoint
test images per width; row-best in bold):

| Width | SP baseline | SP dup-equiv | muP32 baseline | muP32 dup-equiv |
|---|---|---|---|---|
| w32 (IN) | 83.4% | **84.3%** | 84.0% | 81.5% |
| w48 | 81.9% | 83.6% | 83.5% | **84.4%** |
| w64 | 80.7% | 83.6% | 84.2% | **85.1%** |
| w80 | 80.0% | 84.4% | 82.1% | **85.7%** |
| w96 | 80.8% | 85.2% | 81.5% | **86.8%** |
| w128 | 75.4% | 86.2% | 77.8% | **86.7%** |
| w192 | 65.5% | 83.7% | 61.5% | **87.0%** |
| w256 | 58.0% | 80.5% | 36.2% | **87.8%** |
| w384 | 34.0% | 65.3% | 14.5% | **84.3%** |
| w512 | 25.2% | 60.9% | 10.8% | **85.8%** |
| w768 | 13.7% | 44.2% | 8.9% | **84.1%** |
| w1024 | 11.4% | 40.1% | 10.6% | **81.8%** |
| **OOD mean** (w48–w1024) | 55.2% | 72.5% | 50.2% | **85.4%** |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP baseline | `sizegen-fmnist-v2-recip-sgmn-sp-base-bidir-recip_wsu0u4lj` | [`be4pxdjq`](https://wandb.ai/yuxinma/fmnist_cls/runs/be4pxdjq) |
| SP dup-equiv | `sizegen-fmnist-v2-recip-sgmn-sp-deq-bidir-recip_t7p9n6bb` | [`t2e4w9zn`](https://wandb.ai/yuxinma/fmnist_cls/runs/t2e4w9zn) |
| muP32 baseline | `sizegen-fmnist-v2-recip-sgmn-mup32-base-bidir-recip_5gyysz75` | [`vltsez5g`](https://wandb.ai/yuxinma/fmnist_cls/runs/vltsez5g) |
| muP32 dup-equiv | `sizegen-fmnist-v2-recip-sgmn-mup32-deq-bidir-recip_38r10ibl` | [`duusopxf`](https://wandb.ai/yuxinma/fmnist_cls/runs/duusopxf) |

**Training runs** (200 epochs, patience=50):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP baseline | [`wsu0u4lj`](https://wandb.ai/yuxinma/fmnist_cls/runs/wsu0u4lj) | 2.5e-4 | 136 | 84.0% |
| SP dup-equiv | [`t7p9n6bb`](https://wandb.ai/yuxinma/fmnist_cls/runs/t7p9n6bb) | 1e-3 | 77 | 83.4% |
| muP32 baseline | [`5gyysz75`](https://wandb.ai/yuxinma/fmnist_cls/runs/5gyysz75) | 5e-4 | 68 | 83.9% |
| muP32 dup-equiv | [`38r10ibl`](https://wandb.ai/yuxinma/fmnist_cls/runs/38r10ibl) | 2.5e-4 | 84 | 83.7% |

### Matrix-Product GMN Results (`message_fn_type: matrix_product`, forward + bidirectional)

Dup-equiv only, in both parameterizations, both directions. Same protocol and LR grid as above.
On MNIST this variant was the strongest OOD model at large widths (97.5% OOD mean under muP24 vs
82.9% for the unconstrained dup-equiv plain GMN), which is the comparison this reruns on FMNIST.
The bidirectional pair was launched alongside the plain-GMN and ScaleGMN bidirectional queues
(`run_fmnist_cls_sizegen_v2_bidir.sh`); muP32 finished 2026-08-26 04:06, SP 2026-08-27 19:33.

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP forward: 0.7510 / 0.7630 / **0.7670** / **0.7670** / 0.7470 / 0.7360 / 0.7320 → best **2.5e-4**
  (tied with 5e-4; argmax picked the smaller)
- muP32 forward: 0.7580 / 0.7660 / **0.7720** / 0.7650 / 0.7490 / 0.7430 / 0.7270 → best **2.5e-4**
- SP bidirectional and muP32 bidirectional: selected LR is in the training-run table below; full
  sweep curves are in `/tmp/fmnist_sizegen_v2_bidir/SUMMARY.md`.

**Per-width test accuracy, w32 → w1024** (eval-only re-score of the best checkpoint, 1k disjoint
test images per width; row-best in bold):

| Width | SP forward | muP32 forward | SP bidir | muP32 bidir |
|---|---|---|---|---|
| w32 (IN) | 81.0% | **82.2%** | 80.8% | 81.1% |
| w48 | 80.3% | **83.2%** | 79.7% | 83.1% |
| w64 | 82.1% | **85.3%** | 79.9% | 84.9% |
| w80 | 80.8% | **85.2%** | 79.8% | 83.8% |
| w96 | 81.7% | **86.2%** | 79.6% | 85.4% |
| w128 | 78.0% | **86.5%** | 74.7% | 85.2% |
| w192 | 73.9% | 84.9% | 66.1% | **86.4%** |
| w256 | 68.8% | **86.0%** | 62.2% | 85.9% |
| w384 | 62.2% | 83.5% | 50.3% | **84.8%** |
| w512 | 53.9% | **84.3%** | 44.5% | 84.1% |
| w768 | 49.2% | 84.9% | 38.5% | **85.2%** |
| w1024 | 35.6% | **86.7%** | 30.9% | 85.4% |
| **OOD mean** (w48–w1024) | 67.9% | **85.2%** | 62.4% | 84.9% |

| Condition | Checkpoint | Eval run |
|---|---|---|
| SP forward | `sizegen-fmnist-v2-mpgmn-sp_gvq4iwp6` | [`vvk0m77a`](https://wandb.ai/yuxinma/fmnist_cls/runs/vvk0m77a) |
| muP32 forward | `sizegen-fmnist-v2-mpgmn-mup32_qkvcq7qs` | [`najlda6y`](https://wandb.ai/yuxinma/fmnist_cls/runs/najlda6y) |
| SP bidirectional | `sizegen-fmnist-v2-mpgmn-sp-bidir_nonii5lk` | [`erpkss6j`](https://wandb.ai/yuxinma/fmnist_cls/runs/erpkss6j) |
| muP32 bidirectional | `sizegen-fmnist-v2-mpgmn-mup32-bidir_fi9lrcgp` | [`9vhv13og`](https://wandb.ai/yuxinma/fmnist_cls/runs/9vhv13og) |

**Training runs** (200 epochs, patience=50; forward runs early-stopped):

| Variant | Run ID | LR | Epochs | Best Val |
|---|---|---|---|---|
| SP forward | [`gvq4iwp6`](https://wandb.ai/yuxinma/fmnist_cls/runs/gvq4iwp6) | 2.5e-4 | 140 | 83.7% |
| muP32 forward | [`qkvcq7qs`](https://wandb.ai/yuxinma/fmnist_cls/runs/qkvcq7qs) | 2.5e-4 | 120 | 83.3% |
| SP bidirectional | [`nonii5lk`](https://wandb.ai/yuxinma/fmnist_cls/runs/nonii5lk) | 2.5e-4 | 148 | 81.7% |
| muP32 bidirectional | [`fi9lrcgp`](https://wandb.ai/yuxinma/fmnist_cls/runs/fi9lrcgp) | 2.5e-4 | 120 | 81.9% |

### Matrix-Product ScaleGMN Results (`message_fn_type: matrix_product_scale`, forward)

The same MSG constraint on top of **ScaleGMN's** scale-equivariant node states and node update
rather than the plain GMN's concat-MLP (`paper/main.tex` § "Forward matrix-product ScaleGMN").
Dup-equiv only and forward only; identical LR grid, epochs, patience and split files to the
matrix-product GMN block above, so these are drop-in columns in the same tables.

Properties verified before queueing — see
[VERIFICATION_gmn_properties.md § Verification of the matrix-product models](VERIFICATION_gmn_properties.md#verification-of-the-matrix-product-models).

**Date**: 2026-08-28 (queued 00:46, both conditions done 18:22)
**Configs**: `configs/fmnist_cls/mpsgmn_sizegen_{sp,mup32}_v2.yml` for training,
`..._v2_all.yml` for the w32–w1024 re-score, with `--duplication-equiv True --direction forward`
**Launch**: `bash scripts/run_fmnist_cls_mpsgmn_sizegen_v2.sh ""` seeds the queue (2 conditions:
SP, muP32); workers are armed per GPU via `add_worker_when_free.sh`
**Logs**: `/tmp/mpsgmn_fmnist_v2/` (incremental `SUMMARY.md`, per-run sweep / full / eval logs)

**LR sweep** (same grid; val acc at 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3):
- SP: 0.7870 / 0.8030 / **0.8060** / 0.8000 / 0.7940 / 0.7890 / 0.7960 → best **2.5e-4**
- muP32: 0.7860 / 0.7870 / **0.7970** / 0.7940 / 0.7850 / 0.7860 / 0.7770 → best **2.5e-4**

**Per-width test accuracy, w32 → w1024** (eval-only re-score, 1k disjoint test images per width;
row-best in bold):

| Width | SP | muP32 |
|---|---|---|
| w32 (IN) | 82.5% | **82.6%** |
| w48 | **82.7%** | 82.5% |
| w64 | 82.9% | **84.5%** |
| w80 | 83.5% | **84.6%** |
| w96 | 84.5% | **85.4%** |
| w128 | 84.0% | **85.1%** |
| w192 | 82.2% | **86.3%** |
| w256 | 82.6% | **87.5%** |
| w384 | 79.0% | **84.8%** |
| w512 | 76.2% | **86.5%** |
| w768 | 73.1% | **85.0%** |
| w1024 | 68.9% | **86.3%** |
| **OOD mean** (w48–w1024) | 80.0% | **85.3%** |

| Condition | Training run | LR | Epochs | Best val | Checkpoint | Eval run |
|---|---|---|---|---|---|---|
| SP | [`odojljv7`](https://wandb.ai/yuxinma/fmnist_cls/runs/odojljv7) | 2.5e-4 | 76 | 83.5% | `sizegen-fmnist-v2-mpsgmn-sp_odojljv7` | [`9buv07ab`](https://wandb.ai/yuxinma/fmnist_cls/runs/9buv07ab) |
| muP32 | [`a41u4mq3`](https://wandb.ai/yuxinma/fmnist_cls/runs/a41u4mq3) | 2.5e-4 | 84 | 83.1% | `sizegen-fmnist-v2-mpsgmn-mup32_a41u4mq3` | [`fmrhwh99`](https://wandb.ai/yuxinma/fmnist_cls/runs/fmrhwh99) |

---

## Historical Notes (deprecated)

### Size Generalization (deprecated) — train w16, test w24–w96, non-overlapping images

**Deprecated (2026-08-19)**: superseded by
[Experiment 3](#experiment-3--size-generalization-train-w32-test-w32w1024), which trains on w32
— the smallest FMNIST width whose fits are acceptable — and tests out to w1024 ($32\times$).
This run trained on w16, whose fits are the worst of any width (22.9 dB, 15.6% failures), over a
$\leq 6\times$ width range. It used the deprecated fixed-lr$=0.01$ fits tabulated below, deleted
2026-08-19, so it cannot be re-evaluated.

**Goal**: Rigorous size-generalization test where each width uses a *disjoint* set of
FMNIST images. No image appears at multiple widths, eliminating identity leakage.

**Setup**: Train on w16 only; test OOD on w24, w32, w40, w48, w64, w80, w96.
Non-overlapping images per width (`generate_sizegen_splits.py`): 53k train / 1k val / 2k test
for w16; 2k test-only for each other width. No image appears at multiple widths.
batch_size=64, AdamW, warmup 1000 steps, patience=50.

#### Models

Two model families × {baseline, dup-equiv} = 4 model variants per data condition:

**ScaleGMN** (`symmetry: scale`) — scale-equivariant message passing:

| | Baseline | Dup-equiv |
|---|---|---|
| GNN layers | `ScaleEq_GNN_layer` (EquivariantNet msg/upd) | same |
| Aggregator | `add` (sum) | `mean` |
| Edge features | raw weights | fan-in rescaled: $d_{l-1} \cdot W^{(l)}$ |
| Readout | `PermScaleInvariantReadout` (sum pool) | `LayerWiseMeanReadout` (per-layer mean) |
| Dataset | `LabeledINRDataset` | `FanInLabeledINRDataset` |

**Plain GMN** (`symmetry: permutation`) — no scale equivariance:

| | Baseline | Dup-equiv |
|---|---|---|
| GNN layers | `GNN_layer` (concat+MLP msg/upd) | same |
| Aggregator | `add` (sum) | `mean` |
| Edge features | raw weights | fan-in rescaled: $d_{l-1} \cdot W^{(l)}$ |
| Readout | `DeepSet` (sum pool) | `LayerWiseMeanReadout` (per-layer mean, plain MLP) |
| Dataset | `LabeledINRDataset` | `FanInLabeledINRDataset` |

All variants: d_hid=128, 4 GNN layers, forward direction, `readout_range: full_graph`.

#### Data conditions

- **SP (standard parameterization)**: ReLU INRs from `w{16,24,32,40,48,64,80,96}_sp/`.
  Weights have $\|w\|_{V_n} = \Theta(\sqrt{n})$ — norms grow with width.
- **muP (maximal update parameterization)**: ReLU INRs from `w{16,24,32,40,48,64,80,96}_mup/`.
  Weights have $\|w\|_{V_n} = \Theta(1)$ — norms are width-independent.

This gives 8 conditions total: {ScaleGMN, GMN} × {SP, muP} × {baseline, dup-equiv}.

**Protocol**: LR sweep (7 values, 6.25e-5 to 4e-3, ×2 grid, ~10 epochs / 8600 steps) per
condition separately, then full training (200 epochs, patience=50) at best LR per condition.

**Configs**:
- ScaleGMN: `configs/fmnist_cls/scalegmn_sizegen_sp.yml`, `scalegmn_sizegen_mup.yml`
- GMN: `configs/fmnist_cls/gmn_sizegen_sp.yml`, `gmn_sizegen_mup.yml`

#### Reproduction

```bash
source .env

# ScaleGMN (all 4 conditions: LR sweep → full training, ~6h on GPUs 6-7)
bash scripts/run_fmnist_cls_scalegmn_sizegen.sh

# Plain GMN (all 4 conditions: LR sweep → full training, ~4h on GPUs 2-7)
bash scripts/run_fmnist_cls_gmn_sizegen.sh
```

#### ScaleGMN Results

#### LR Sweep (8600 steps each, 7 LRs per condition)

**Date**: 2026-06-26

| LR | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| 6.25e-5 | 0.7840 | 0.7890 | 0.7970 | 0.8020 |
| **1.25e-4** | 0.7940 | **0.8020** | **0.8150** | **0.8200** |
| **2.5e-4** | **0.8170** | 0.8010 | 0.8100 | 0.8130 |
| 5e-4 | 0.7950 | 0.7960 | 0.8060 | 0.8000 |
| 1e-3 | 0.8150 | 0.7950 | 0.8070 | 0.7910 |
| 2e-3 | 0.8000 | 0.7900 | 0.7940 | 0.7860 |
| 4e-3 | 0.8010 | 0.7840 | 0.7760 | 0.7830 |

Best LRs selected: SP baseline=2.5e-4, SP dup-equiv=1.25e-4, muP baseline=1.25e-4, muP dup-equiv=1.25e-4.

#### Full Training (200 epochs, patience=50)

SP runs early-stopped (epoch ~62). muP runs early-stopped (epoch ~67).

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 80.7% / 0.519 | 80.5% / 0.563 | 80.5% / 0.510 | 80.6% / 0.523 |
| w24 (OUT) | 81.8% / 0.470 | 82.4% / 0.478 | 81.5% / 0.488 | 82.4% / 0.467 |
| w32 (OUT) | 82.4% / 0.441 | 81.7% / 0.484 | 83.4% / 0.437 | 83.8% / 0.462 |
| w40 (OUT) | 83.4% / 0.463 | 81.2% / 0.531 | 83.1% / 0.453 | **83.8%** / 0.450 |
| w48 (OUT) | 82.4% / 0.476 | 80.4% / 0.545 | 82.0% / 0.497 | 82.5% / 0.488 |
| w64 (OUT) | 81.7% / 0.481 | 80.8% / 0.524 | 80.1% / 0.502 | **84.1%** / 0.417 |
| w80 (OUT) | 79.1% / 0.540 | 78.7% / 0.581 | 78.0% / 0.582 | **83.1%** / 0.452 |
| w96 (OUT) | 80.3% / 0.552 | 80.5% / 0.535 | 78.0% / 0.584 | **84.2%** / 0.426 |
| **OOD mean** | 81.6% | 80.8% | 80.9% | **83.4%** |

| Variant | Run ID | Config | LR |
|---|---|---|---|
| SP baseline | [`yn59ydp7`](https://wandb.ai/yuxinma/fmnist_cls/runs/yn59ydp7) | `scalegmn_sizegen_sp.yml` | 2.5e-4 |
| SP dup-equiv | [`xvezjk9g`](https://wandb.ai/yuxinma/fmnist_cls/runs/xvezjk9g) | same + `--duplication-equiv` | 1.25e-4 |
| muP baseline | [`7oa98rax`](https://wandb.ai/yuxinma/fmnist_cls/runs/7oa98rax) | `scalegmn_sizegen_mup.yml` | 1.25e-4 |
| muP dup-equiv | [`2w37bisc`](https://wandb.ai/yuxinma/fmnist_cls/runs/2w37bisc) | same + `--duplication-equiv` | 1.25e-4 |

#### Plain GMN Results

#### LR Sweep (8600 steps each, 7 LRs per condition)

**Date**: 2026-06-26

| LR | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| 6.25e-5 | 0.7230 | 0.7140 | — | 0.6840 |
| **1.25e-4** | **0.7500** | 0.7340 | 0.7180 | 0.7200 |
| **2.5e-4** | 0.7410 | 0.7280 | **0.7320** | 0.7240 |
| **5e-4** | 0.7330 | **0.7440** | 0.7310 | 0.7230 |
| **1e-3** | 0.7380 | 0.7310 | 0.7110 | **0.7280** |
| 2e-3 | 0.7160 | 0.7420 | 0.7110 | 0.7040 |
| 4e-3 | 0.7030 | 0.6850 | 0.6830 | 0.6570 |

Best LRs selected: SP baseline=1.25e-4, SP dup-equiv=5e-4, muP baseline=2.5e-4, muP dup-equiv=1e-3.

#### Full Training (200 epochs, patience=50)

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 77.9% / 0.578 | 78.0% / 0.581 | 79.2% / 0.616 | 80.3% / 0.545 |
| w24 (OUT) | 78.7% / 0.565 | 80.1% / 0.533 | 78.7% / 0.633 | 80.5% / 0.518 |
| w32 (OUT) | 72.6% / 0.701 | 78.4% / 0.553 | 72.6% / 0.768 | 81.8% / 0.475 |
| w40 (OUT) | 61.2% / 0.996 | 76.9% / 0.587 | 60.0% / 1.149 | 80.7% / 0.474 |
| w48 (OUT) | 48.1% / 1.272 | 76.7% / 0.620 | 48.3% / 1.598 | 80.7% / 0.526 |
| w64 (OUT) | 31.6% / 1.753 | 74.2% / 0.666 | 35.8% / 2.230 | **80.8%** / 0.508 |
| w80 (OUT) | 24.7% / 1.990 | 67.4% / 0.821 | 30.8% / 2.660 | **79.3%** / 0.529 |
| w96 (OUT) | 21.4% / 2.108 | 64.0% / 0.869 | 30.4% / 2.777 | **80.1%** / 0.532 |
| **OOD mean** | 48.3% | 73.9% | 50.9% | **80.5%** |

| Variant | Run ID | Config | LR |
|---|---|---|---|
| SP baseline | [`cijpxize`](https://wandb.ai/yuxinma/fmnist_cls/runs/cijpxize) | `gmn_sizegen_sp.yml` | 1.25e-4 |
| SP dup-equiv | [`j5mhkghy`](https://wandb.ai/yuxinma/fmnist_cls/runs/j5mhkghy) | same + `--duplication-equiv` | 5e-4 |
| muP baseline | [`7g25j0jw`](https://wandb.ai/yuxinma/fmnist_cls/runs/7g25j0jw) | `gmn_sizegen_mup.yml` | 2.5e-4 |
| muP dup-equiv | [`hlo4s2d4`](https://wandb.ai/yuxinma/fmnist_cls/runs/hlo4s2d4) | same + `--duplication-equiv` | 1e-3 |

### Deprecated INR fits (w16–w96, fixed lr=0.01)

The fits Experiment 1 and the w16-trained size-generalization run above were scored on: every width
fitted with **lr=0.01** (no per-width sweep), all 70k images per width, muP with base_width=32 but
no LR sweep either. Measured on 1000
random samples per width, `[2, w, w, 1]`, 1000 steps, "Fail%" = fraction with $\text{MSE} > 0.01$.

**Standard-param ReLU INRs**:

| Width | MSE mean | MSE median | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|---|---|
| 16 | 0.006486 | 0.005139 | 22.9 dB | 22.9 dB | 2.99 | 15.6% |
| 24 | 0.004762 | 0.003582 | 24.4 dB | 24.5 dB | 3.17 | 8.6% |
| 32 | 0.003812 | 0.002746 | 25.5 dB | 25.6 dB | 3.24 | 5.6% |
| 48 | 0.003061 | 0.002147 | 26.6 dB | 26.7 dB | 3.48 | 3.2% |
| 64 | 0.002721 | 0.001861 | 27.2 dB | 27.3 dB | 3.54 | 3.3% |
| 80 | 0.002664 | 0.001794 | 27.4 dB | 27.5 dB | 3.69 | 3.6% |
| 96 | 0.002475 | 0.001602 | 27.7 dB | 28.0 dB | 3.70 | 2.7% |

**muP ReLU INRs (base_width=32, lr=0.01)**:

| Width | PSNR mean | PSNR median | PSNR std | Fail% |
|---|---|---|---|---|
| 16 | 23.5 dB | 23.6 dB | 3.29 | 13.4% |
| 24 | 24.7 dB | 24.7 dB | 3.26 | 7.8% |
| 32 | 25.5 dB | 25.6 dB | 3.22 | 5.8% |
| 48 | 26.3 dB | 26.4 dB | 3.31 | 3.6% |
| 64 | 26.6 dB | 26.7 dB | 3.28 | 3.2% |
| 80 | 26.8 dB | 26.8 dB | 3.32 | 2.5% |
| 96 | 27.0 dB | 27.2 dB | 3.25 | 2.9% |

### Anydim ScaleGMN (mixed-width training, deprecated)

Mixed-width training on w16-w48 simultaneously.

**Protocol**: LR sweep (7 values, 6.25e-5 to 4e-3, x2 grid, 10 epochs) then full run at best LR.

The `scalegmn_anydim_*.yml` configs and `train_anydim.py` referenced below were deleted in
the deprecated-path cleanup; recover them from git history if this run needs reproducing.

**SP ReLU** (`configs/fmnist_cls/scalegmn_anydim_sp.yml`):
LR sweep best = 1e-3. Full training: baseline 83.8%/81.7%, dup-equiv 84.4%/82.1%.
Runs: [`6z4vkauy`](https://wandb.ai/yuxinma/fmnist_cls/runs/6z4vkauy), [`afrsz5pm`](https://wandb.ai/yuxinma/fmnist_cls/runs/afrsz5pm).

**muP ReLU** (`configs/fmnist_cls/scalegmn_anydim_mup.yml`):
LR sweep best = 5e-4. Full training: baseline 83.8%/81.4%, dup-equiv 83.6%/81.7%.
Runs: [`8aixyva0`](https://wandb.ai/yuxinma/fmnist_cls/runs/8aixyva0), [`z05vjovw`](https://wandb.ai/yuxinma/fmnist_cls/runs/z05vjovw).

### Size Generalization v1 (shared images across widths, deprecated)

Used the same FMNIST images at all widths (same splits, seed=42). The baseline's strong
performance may be explained by image-identity leakage across widths.

| Width | SP baseline | SP dup-equiv | muP baseline | muP dup-equiv |
|---|---|---|---|---|
| w16 (IN) | 79.8% | 77.9% | 76.4% | 78.1% |
| w24 (IN) | 80.6% | 80.5% | 79.8% | 80.6% |
| w32 (IN) | 81.4% | 81.3% | 81.6% | 80.8% |
| w48 (OUT) | 82.8% | 81.4% | 82.3% | 82.3% |
| w64 (OUT) | 82.5% | 82.9% | 82.6% | 83.1% |
| w80 (OUT) | 81.4% | 82.7% | 81.3% | 83.4% |
| w96 (OUT) | 81.3% | 81.5% | 81.1% | **84.6%** |
| **OOD mean** | 82.0% | 82.1% | 81.8% | **83.3%** |

Runs: [`t97ca3pz`](https://wandb.ai/yuxinma/fmnist_cls/runs/t97ca3pz), [`g3cipr64`](https://wandb.ai/yuxinma/fmnist_cls/runs/g3cipr64), [`qnhp7s9j`](https://wandb.ai/yuxinma/fmnist_cls/runs/qnhp7s9j), [`opyabcqj`](https://wandb.ai/yuxinma/fmnist_cls/runs/opyabcqj).

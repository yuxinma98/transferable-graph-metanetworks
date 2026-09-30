# Transferable Graph Metanetworks

Code for the paper **"Transferable Graph Metanetworks"** by Yuxin Ma, Adir Dayan, Yam Eitan,
Haggai Maron, and Soledad Villar.

<!-- TODO before release: replace with the arXiv link. -->
📄 [Paper](https://arxiv.org/abs/XXXX.XXXXX)

A *weight space network* (or *metanetwork*) takes the weights of another neural network as input
and predicts properties of it. Prior work trains such models on input networks of one or a few
fixed widths and evaluates them in-distribution. This repository implements **Transferable Graph
Metanetworks**: a set of modifications that make an existing graph metanetwork (GMN) transferable
across input networks of different *widths*, so that a metanetwork trained only on small input
networks still performs well on much larger ones.

The modifications follow two principles — **invariance** to the ways networks of different widths
represent the same function, and **continuity**, so that weights representing similar functions
receive similar predictions. They come in two levels:

| Level | Model | What changes | Invariant to |
|---|---|---|---|
| — | **Baseline GMN** | unmodified GMN | nothing |
| 1 | **Duplication-compatible GMN** | fan-in rescaling of input weights, mean aggregation, layer-wise mean readout | the duplication widening (copy and rescale weights) |
| 2 | **Matrix-product GMN** | additionally restricts the message function to multiplication by the raw weight | the block-wise widening (the largest function-preserving widening), and Lipschitz in the normalized spectral norm |

Both levels are generic and apply on top of any standard GMN taking MLP or CNN input networks.
We apply them to a plain GMN and to [ScaleGMN](https://github.com/jkalogero/scalegmn), in forward
and bidirectional variants.

Empirically the modifications improve size generalization on every task considered. Performance is
strongest on input networks trained under the maximal-update parameterization (μP), where it stays
robust up to **42× the training width**.

> **Terminology.** The paper says *duplication-compatible*; the code and configs say
> `duplication-equiv` / `dup-equiv` / `--duplication-equiv` for the same thing.

## Tasks

Two tasks, each with two datasets, all adapted from [ScaleGMN](https://github.com/jkalogero/scalegmn)
and used repeatedly in prior work:

| Task | Input networks | Metric | Trained at | Evaluated up to |
|---|---|---|---|---|
| **INR classification** | MLPs fitted to MNIST / FMNIST images | accuracy | width 24 (MNIST), 32 (FMNIST) | width 1024 (42× / 32×) |
| **Generalization prediction** | small CNNs trained on CIFAR-10-GS / SVHN-GS | R², Kendall τ | width 16 | width 512 (32×) |

Input networks come in two parameterizations, **SP** (standard) and **μP** (maximal-update, which
is calibrated to coincide with SP at the training width), and at each width they are either
hand-crafted function-preserving widenings of the training-width networks or trained independently
from random initialization.

## Installation

```bash
conda env create -n anydim-metanet --file environment.yml
conda activate anydim-metanet
pip install -e .          # installs src/ as a package

cp .env.example .env      # then edit ANYDIM_DATA_ROOT
```

`ANYDIM_DATA_ROOT` is where all datasets, fitted input networks, and checkpoints live; it falls
back to `./data`. Every script sources `.env` if present. Checkpoints are written to
`$ANYDIM_DATA_ROOT/checkpoints/<run_name>_<wandb_id>/best.pt`.

**Disk and compute.** Regenerating everything is expensive: the input networks *are* the dataset.
The CIFAR-10 CNN zoo is ~6 GB up to width 128 and ~89 GB with the width 192–512 extension, SVHN the
same again, and the large INR widths add tens of GB. Fitting the INRs and training the ~1.4k input
CNNs per width needs a GPU. The metanetworks themselves are small and train in hours.

## Repository map

```
src/
  models/          our GMN modifications: layer-wise mean readout, last-layer readout,
                   matrix-product layers (scalar + conv, plain + ScaleGMN),
                   bidirectional reciprocal edge features
  data/            input-network generation and graph datasets:
                   INR fitters (SP + muP), CNN zoo trainer, widening families,
                   fan-in-rescaled graph datasets
  scalegmn/        vendored copy of ScaleGMN (MIT) — not modified; see Attribution
configs/
  mnist_cls/, fmnist_cls/        INR classification
  cifar10_predgen/, svhn_predgen/  generalization prediction
scripts/
  generate_*.py    build the input-network datasets and splits
  train_*.py       train / evaluate the metanetworks
  check_*.py       numerical verification of the claimed invariances
  plot_*.py        figures, parsed directly out of results/*.md
  run_*.sh         launch scripts for full experiment sweeps
results/           recorded numbers, with the wandb run ID for every result
```

Our modifications are deliberately **additive**: `src/scalegmn/` is a vendored copy of upstream
ScaleGMN that is never edited. Everything in `src/models/` installs onto a constructed model by
instance-level wrappers or by rebinding layer classes during model construction, so upstream can be
updated independently.

### Config naming

`{model}_{experiment}_{parameterization}[_{variant}].yml`, where `{model}` is one of

- `gmn` — plain GMN (`symmetry: permutation`)
- `scalegmn` — ScaleGMN (`symmetry: scale`)
- `mpgmn` — matrix-product GMN
- `mpsgmn` — matrix-product ScaleGMN

The duplication-compatible variant of each is not a separate config; it is the
`--duplication-equiv True` flag. Likewise the bidirectional arm is `--direction bidirectional`.

## Reproducing the experiments

Each stage needs the previous one's output on disk. Pick a free GPU and set
`CUDA_VISIBLE_DEVICES` explicitly.

### 1. Build the input networks

INR classification — fit MLPs to images, then partition images across widths so that no image is
reused at two widths:

```bash
# SP and muP INRs at each width
python scripts/generate_inrs.py --widths 24 32 48 64 80 96 128 192 256
python scripts/generate_inrs.py --widths 24 32 48 64 80 96 128 192 256 --mup --base-width 24

# non-overlapping per-width splits
python scripts/generate_sizegen_splits.py --dataset mnist --init-type sp \
    --train-width 24 --test-widths 32 48 64 80 96 128 192 256

# FMNIST: one script does splits, learning-rate sweeps, and fitting
bash scripts/run_fmnist_cls_sizegen_data.sh <gpu_id>
```

Fitting learning rates are swept per width with `scripts/sweep_inr_lr.py`; fit quality is checked
with `scripts/verify_inr_quality.py`. For the very large widths, only each width's ~1k test INRs
are fitted — fitting all 70k at width 1024 would need ~300 GB.

Generalization prediction — train our own multi-width CNN zoo (the upstream zoo is single-width):

```bash
bash scripts/download_cnn_zoo.sh          # upstream single-width zoo, for the width-16 baseline
bash scripts/run_cifar10_zoo_gen.sh       # our zoo, widths 16-128, SP and muP16 arms
bash scripts/run_cifar10_zoo_gen_v2.sh    # extends to widths 192-512
```

Hand-crafted widenings, for the controlled experiment:

```bash
python scripts/generate_duplicated_inrs.py --init-type sp --base-width 24 \
    --target-widths 48 96 144 192 240 --family uniform    # and --family general
python scripts/generate_duplicated_cnns.py --dataset cifar10 --family uniform
```

`--family` selects the equivalence class the widening is drawn from (`uniform` is the duplication
widening, `general` the largest function-preserving one). Both scripts verify as they go
(`--verify N` on the INR side, `--check-every N` on the CNN side): they report the residual change
in the represented function *and* the non-uniformity of the blocks drawn, so a widening that
silently degenerated to the uniform case cannot pass unnoticed.

### 2. Size generalization on hand-crafted widenings

The controlled check: the widened input networks represent exactly the same functions, so an
invariant metanetwork's error must stay flat as the test width grows.

```bash
bash scripts/run_mnist_cls_widen_families_eval.sh "0 1"
bash scripts/run_fmnist_cls_widen_families_eval.sh "0"
bash scripts/run_predgen_widen_families.sh cifar10 "0 1"
```

### 3. Size generalization on independently trained weights

The realistic setting, and the paper's main figure: every width's input networks are trained
independently from random initialization. The metanetwork trains at the smallest width only, with
hyperparameters selected on the validation metric at that width only.

```bash
# INR classification
bash scripts/run_mnist_cls_sizegen_v3.sh              # train w24, test w24-w256
bash scripts/run_mnist_cls_sizegen_v4_eval.sh "0 2"   # re-score those checkpoints on w24-w1024
bash scripts/run_fmnist_cls_sizegen_v2.sh "0"         # train w32, test w32-w1024

# generalization prediction
bash scripts/run_cifar10_predgen_sizegen_v1.sh "0 1"  # train w16, test w16-w128
bash scripts/run_cifar10_predgen_sizegen_v2.sh "0 1"  # re-score on w16-w512
bash scripts/run_svhn_predgen_sizegen_v1.sh "0 1"
```

Runners that take a GPU list use it as `"<id> <id> ..."`, one worker per GPU; they have a default
that will not match your machine, so pass it explicitly.

Extending the reach to larger test widths retrains nothing: the `_v2`/`_v4` configs differ from
`_v1`/`_v3` only in their test widths, so the same checkpoints are re-scored via
`--eval-only-ckpt`, and the runners verify that the overlapping widths reproduce the original
numbers.

The `run_*.sh` runners drain a resumable job queue with one worker per GPU. Re-running is safe and
is also how you add a worker: finished work is detected and skipped. A single run, without the
queue machinery:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sizegen.py \
    --conf configs/mnist_cls/mpgmn_sizegen_mup24_v4.yml \
    --duplication-equiv True --lr 5e-4 --wandb True
```

### 4. Verifying the claimed invariances

Every property proved in the paper has a numerical check, reported in
[results/VERIFICATION_gmn_properties.md](results/VERIFICATION_gmn_properties.md):

| Script | Checks |
|---|---|
| `check_widening_equiv.py`, `check_cnn_widening_equiv.py` | the widenings preserve the represented function |
| `check_duplication_equiv.py`, `check_duplication_equiv_cnn.py` | duplication invariance, at node and graph level |
| `check_matrix_product_gmn.py`, `check_matrix_product_gmn_cnn.py` | the matrix-product identity, over five widening families, and spectral continuity |
| `check_scale_equiv.py`, `check_scale_equiv_cnn.py` | scale equivariance of the ScaleGMN variants |
| `check_mup_cnn.py` | the μP implementation, against the reference `mup` library |

## Results

Recorded numbers live in [`results/`](results/), one document per task and dataset, each carrying
the wandb run ID for every result:

- [`EXPERIMENTS_MNIST_INR_classification.md`](results/EXPERIMENTS_MNIST_INR_classification.md)
- [`EXPERIMENTS_FMNIST_INR_classification.md`](results/EXPERIMENTS_FMNIST_INR_classification.md)
- [`EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md`](results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md)
- [`EXPERIMENTS_SVHN_CNN_accuracy_prediction.md`](results/EXPERIMENTS_SVHN_CNN_accuracy_prediction.md)
- [`VERIFICATION_gmn_properties.md`](results/VERIFICATION_gmn_properties.md)

The plotting scripts parse these documents directly rather than keeping their own copy of the
numbers, so a figure cannot disagree with a recorded result.

One negative result is worth flagging for anyone building on this: training a **bidirectional,
positive-scale-equivariant** metanetwork (`scalegmn` with `--direction bidirectional` on CNN
inputs) is numerically unstable, and we report it as out of scope rather than as a result. It needs
the reciprocal of the raw weight, and the CNN zoo contains weights near 1e-15. This is the same
instability the ScaleGMN paper reports in its Appendix A.2 for elementwise-division backward
messages; their only bidirectional result uses a sign symmetry, where the reciprocal is the
identity and the division disappears. The forward variants and all `permutation`-symmetry
bidirectional variants are unaffected.

## Attribution

`src/scalegmn/` is a vendored, unmodified copy of **ScaleGMN** by Ioannis Kalogeropoulos, Giorgos
Bouritsas, and Yannis Panagakis ([paper](https://arxiv.org/abs/2406.10685),
[code](https://github.com/jkalogero/scalegmn)), used under the MIT License, which is preserved at
`src/scalegmn/LICENSE`. Our GMN baselines, the INR classification and generalization prediction
task setups, and the data splits follow that work.

The upstream single-width CNN zoo is the *Small CNN Zoo* of
[Unterthiner et al. (2020)](https://arxiv.org/abs/2002.11448). The multi-width zoos are ours.

## Citation

<!-- TODO before release: fill in the arXiv eprint number. -->
```bibtex
@article{ma2026transferable,
  title   = {Transferable Graph Metanetworks},
  author  = {Ma, Yuxin and Dayan, Adir and Eitan, Yam and Maron, Haggai and Villar, Soledad},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

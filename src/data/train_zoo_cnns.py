"""Batched trainer for the small-CNN-zoo architecture, in SP and muP parameterizations.

The CNN analog of :mod:`src.data.fit_inrs`: train ``N`` independent CNNs of one width
simultaneously on one GPU, each with its **own** hyperparameter draw (learning rate,
init scale, weight decay, dropout, training-set fraction, epoch budget) and its own
stream of training data.

Batching uses ``torch.func.vmap`` over ``torch.func.functional_call`` on a single
:class:`src.data.zoo_cnn.ZooCNN` template, *not* grouped convolutions.  With grouped
convolutions the model index is fused into ``out_channels``, which would make
``mup.set_base_shapes`` infer the wrong ``infshape`` on the template and silently
corrupt the muP arm; with ``vmap`` the template stays an ordinary ``nn.Conv2d`` network
that ``mup`` can be applied to directly (that is exactly what
``scripts/check_mup_cnn.py`` does to obtain a reference to compare against).  PyTorch
lowers ``vmap`` over ``conv2d`` with a batched weight to a grouped convolution anyway,
so nothing is lost in throughput.

muP with an explicit readout multiplier
--------------------------------------
Reference semantics are the installed ``mup`` package (v1.0.0), read directly:

* ``MuAdam`` divides the lr by ``width_mult`` for parameters with
  ``infshape.ninf() == 2`` only, and multiplies their weight decay by the same factor.
* ``MuReadout`` multiplies its weight and bias by ``width_mult**0.5`` once, and its
  forward is ``super().forward(output_mult * x / width_mult())``.
* ``InfShape.width_mult()`` reads the **last** infinite dimension, so for a
  ``[c_out, c_in, 3, 3]`` conv weight it is ``c_in / c_in_base`` — correct only because
  the kernel size is fixed across widths (trap T1).

For ``[1 -> c -> c -> c -> GAP -> 10]`` with base ``c0`` and ``wm = c / c0`` that gives

    conv1.weight   [c,1,3,3]   ninf 1   input   default init (fan_in 9)      lr
    conv{2,3}.w    [c,c,3,3]   ninf 2   hidden  default init (fan_in 9c)     lr / wm
    conv*.bias     [c]         ninf 1   vector  (we zero-init, see below)    lr
    fc.weight      [10,c]      ninf 1   output  default init * wm**0.5       lr
    fc.bias        [10]        ninf 0   fixed   (zero-init)                   lr

and this trainer implements exactly that, with ``MuReadout``'s division carried in the
forward pass as :attr:`ZooCNN.readout_mult` ``= 1 / wm``.  So the lr table below is
literally ``MuAdam``'s rule --- divide by ``wm`` iff ``ninf == 2``, no exception for the
readout --- and the same convention as :class:`MuBatchINRFitter` on the INR side.

The consequence is that ``self.params["fc.weight"]`` is **not** the function's readout
weight: the function uses ``W_eff = W / wm``.  The metanetwork consumes the exported
``.pth``, not the trainer, so :meth:`export_state_dicts` folds the multiplier in and
what lands on disk is always ``W_eff`` (trap T3); :meth:`spectral_norm_components` does
the same, since it reports a property of the function.  ``scripts/check_mup_cnn.py`` V4
is what holds that division in place.

(The alternative is to store ``W_eff`` and give the readout a ``1/wm`` lr instead, which
is the same update in different variables and makes the export a no-op, at the cost of a
readout lr that no longer matches ``MuAdam``'s table and an Adam ``eps`` that no longer
matches the reference's.  We took that route first and moved off it.)

Weight decay (trap T2).  The installed ``mup`` has no ``decoupled_wd`` flag; ``MuAdam``
instead does ``lr /= wm`` **and** ``weight_decay *= wm`` for matrix-like parameters.
Under AdamW's decoupled decay ``p -= lr * wd * p`` those two cancel, so the *effective*
decay per step is width-independent — i.e. a sampled ``wd`` means the same thing at
every width, which is what a swept hyperparameter has to mean.  This trainer therefore
applies decoupled decay with the **base** lr for every parameter, and never scales
``wd``.  (``MuAdam``'s default ``impl=Adam`` puts wd in the gradient instead, where the
cancellation does not hold; ``MuAdamW`` is the reference here, matching the AdamW used
throughout this repo.)

Adam's ``eps`` (trap T7).  Adam's update ``mhat/(sqrt(vhat) + eps)`` is scale-invariant
in the gradient *only* while ``eps`` is negligible against ``sqrt(vhat)``.  Under muP the
gradient shrinks with width, so a fixed ``eps`` eventually dominates and the update
collapses --- the parameterization silently stops being muP at large width.  Everett et
al. (2024), *Scaling Exponents Across Parameterizations and Optimizers*, Sec. 4.3, make
``eps`` part of the parameterization for exactly this reason and prescribe a per-layer
``eps_l = eps_base * (n / n_base) ** (-g_l)`` with ``g_l`` the layer's gradient-scale
exponent.  Measured on this network in this convention (grad RMS over a 32x width
increase, at ``t = 1`` and ``t = 20``), every parameter that *has* an infinite dimension
sits at ``g = 1`` --- ratios 0.025..0.034 against the ``1/32 = 0.031`` prediction, for the
input conv, both hidden convs, the conv biases and the readout weight alike --- while
``fc.bias`` (``ninf 0``, ten finite coordinates whose gradient is ``dL/df``) stays flat at
ratio 1.00.  Hence the rule implemented here:

    eps(theta) = eps_base / wm   if ninf(theta) >= 1,   else eps_base

i.e. ``1/wm`` on every vector-like and matrix-like parameter and nothing on the finite
output bias.  As with every other factor it is a function of ``wm``, not of ``c``, so it
is exactly ``1.0`` at ``c == c0`` and the base-width identity survives.  This is a
deviation from the installed ``mup``, which does not scale ``eps`` at all; ``eps_scaling
= False`` restores the library's behaviour, and ``check_mup_cnn.py`` V3 tests our table
against an independently declared one (plus a ``[ctl-eps]`` control at an inflated
``eps`` where the difference is numerically visible).  The INR fitter deliberately keeps
a fixed ``eps``.

Biases are **zero-initialized** in both arms.  ``mup`` leaves conv biases at the torch
default, whose scale ``1/sqrt(9c)`` is width-dependent and therefore not muP-correct,
and ``MuReadout._rescale_parameters`` scales the readout bias by ``wm**0.5`` on top of
a ``1/sqrt(c)`` default.  Zero init removes both ambiguities and is common in the
original zoo's sweep; it is a deliberate deviation, recorded in the results file.

Trap T6: the swept init scale is a **multiplier** on the fan-in-dependent default init,
never an absolute std.  The default (Kaiming-uniform, fan_in ``= c_in * kh * kw``)
already has variance ``prop 1/c`` for the hidden convs, which *is* the muP scaling; an
absolute std would break it.

Trap T5: the architecture has no normalization layers, so there are no additional
vector-like parameters with their own rules.

Trap T4: every model has its own sampled lr, so there is no shared param-group lr to
scale.  The muP per-layer factor multiplies a per-model lr **vector** inside the
hand-written Adam below; :func:`scripts.check_mup_cnn` V3 compares that path against
``mup.MuAdamW`` on a single model to make sure it did not diverge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from .zoo_cnn import NUM_CONV_LAYERS, ZooCNN, wb_to_state_dict

#: activation-memory budget used to pick cohort / eval batch sizes: roughly
#: n_models * batch * width elements across the three feature maps.
_ACT_BUDGET = 2_000_000


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

@dataclass
class ZooHP:
    """One hyperparameter draw.

    ``lr``, ``init_mult`` and ``weight_decay`` are the *base* values: under muP they
    are interpreted at the base width and the per-layer factors are applied on top, so
    the same draw means the same thing at every width.
    """
    lr: float
    init_mult: float
    weight_decay: float
    dropout: float
    train_frac: float
    epochs: int

    def as_dict(self):
        return asdict(self)


#: the swept ranges. Optimizer (Adam) and activation (ReLU) are FIXED, unlike
#: Unterthiner et al., who sweep both: per-optimizer muP rules and ScaleGMN's
#: ``symmetry: scale`` each require one of them held constant. Recorded in the
#: results file as a deliberate deviation.
LR_RANGE = (5e-5, 5e-2)          # log-uniform
INIT_MULT_RANGE = (0.25, 4.0)    # log-uniform, multiplier on the default init
WD_RANGE = (1e-6, 1e-1)          # log-uniform
DROPOUT_CHOICES = (0.0, 0.2, 0.45)
TRAIN_FRAC_CHOICES = (0.1, 0.25, 0.5, 1.0)
EPOCH_CHOICES = (5, 10, 20)


def sample_hps(n: int, seed: int) -> list[ZooHP]:
    """Draw ``n`` independent hyperparameter settings.

    The draw depends only on ``(n, seed)`` and **not** on the width, so the same seed
    reproduces the same draws at every width — that is what makes the paired diagnostic
    set paired, and what makes "disjoint draws across widths" a matter of using
    different seeds.
    """
    rng = np.random.default_rng(seed)

    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    return [
        ZooHP(lr=logu(*LR_RANGE),
              init_mult=logu(*INIT_MULT_RANGE),
              weight_decay=logu(*WD_RANGE),
              dropout=float(rng.choice(DROPOUT_CHOICES)),
              train_frac=float(rng.choice(TRAIN_FRAC_CHOICES)),
              epochs=int(rng.choice(EPOCH_CHOICES)))
        for _ in range(n)
    ]


def step_budget(hp: ZooHP, n_train: int, batch_size: int) -> int:
    """Optimizer steps for one draw: ``epochs`` passes over its own data fraction."""
    return max(1, int(round(hp.epochs * hp.train_frac * n_train / batch_size)))


def cohort_size(width: int, batch_size: int, cap: int = 128) -> int:
    """How many models of this width to train in one ``vmap`` batch."""
    return int(max(4, min(cap, _ACT_BUDGET // (batch_size * width))))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: ITU-R 601-2 luma coefficients, as used by ``torchvision.transforms.Grayscale``.
_LUMA = (0.299, 0.587, 0.114)


def load_cifar10_gray(root, device="cpu"):
    """CIFAR-10 as grayscale 32x32 floats in ``[0, 1]``.

    Returns ``(train_x, train_y, test_x, test_y)`` with ``x`` of shape
    ``[n, 1, 32, 32]`` (float32, kept on ``device``) and ``y`` int64.
    """
    from torchvision import datasets

    root = str(root)
    out = []
    for train in (True, False):
        ds = datasets.CIFAR10(root=root, train=train, download=False)
        x = torch.from_numpy(ds.data).float().div_(255.0)          # [n, 32, 32, 3]
        gray = (x[..., 0] * _LUMA[0] + x[..., 1] * _LUMA[1] + x[..., 2] * _LUMA[2])
        out.append(gray.unsqueeze(1).to(device))                   # [n, 1, 32, 32]
        out.append(torch.tensor(ds.targets, dtype=torch.long, device=device))
    return tuple(out)


def load_svhn_gray(root, device="cpu"):
    """SVHN (cropped digits) as grayscale 32x32 floats in ``[0, 1]``.

    Same signature and output contract as :func:`load_cifar10_gray`.  Three
    differences from the CIFAR-10 loader, all forced by ``torchvision``'s SVHN reader:
    ``.data`` is already channel-first ``[n, 3, 32, 32]`` (not ``[n, 32, 32, 3]``), the
    labels live in ``.labels`` rather than ``.targets`` (already remapped so digit `0`
    is class 0, not 10), and ``download=True`` because the ``.mat`` files are small and
    not staged on this machine the way CIFAR-10's pickles are.

    The ``train``/``test`` splits (73257 / 26032) are the ones TFDS' ``svhn_cropped``
    uses, i.e. the same data the upstream zoo was trained on; the 531k ``extra`` split
    is deliberately not used.
    """
    from torchvision import datasets

    root = str(root)
    out = []
    for split in ("train", "test"):
        ds = datasets.SVHN(root=root, split=split, download=True)
        x = torch.from_numpy(ds.data).float().div_(255.0)          # [n, 3, 32, 32]
        gray = (x[:, 0] * _LUMA[0] + x[:, 1] * _LUMA[1] + x[:, 2] * _LUMA[2])
        out.append(gray.unsqueeze(1).to(device))                   # [n, 1, 32, 32]
        out.append(torch.tensor(ds.labels, dtype=torch.long, device=device))
    return tuple(out)


#: ``dataset name -> (loader, human-readable name)``.  The only dataset-specific code
#: in the CNN study: everything downstream of the exported ``.pth`` files depends on the
#: architecture, not on which images the CNNs were fitted to.
GRAY_LOADERS = {
    "cifar10": (load_cifar10_gray, "CIFAR-10"),
    "svhn": (load_svhn_gray, "SVHN"),
}


# ---------------------------------------------------------------------------
# Batched trainer
# ---------------------------------------------------------------------------

def _param_ninf(width, in_channels=1, num_classes=10, num_conv_layers=NUM_CONV_LAYERS):
    """``{parameter name: ninf}`` — how many of its dimensions are infinite.

    Inferred the way ``mup.set_base_shapes`` does it — a dimension is infinite if it
    differs between two reference widths — rather than hardcoded, so a change to the
    architecture cannot silently leave a layer on the wrong lr or ``eps``.  Built on the
    ``meta`` device: only shapes are needed.
    """
    def shapes(w):
        with torch.device("meta"):
            model = ZooCNN(width=w, in_channels=in_channels, num_classes=num_classes,
                           num_conv_layers=num_conv_layers)
        return {n: tuple(p.shape) for n, p in model.named_parameters()}

    a, b = shapes(width), shapes(width * 2)
    return {n: sum(1 for d in range(len(s)) if s[d] != b[n][d])
            for n, s in a.items()}


class BatchedZooCNNTrainer:
    """Train ``len(hps)`` zoo CNNs of one width in parallel.

    Args:
        hps: one :class:`ZooHP` per model.
        width: channel count ``c``.
        base_width: ``None`` for the SP arm; an int enables muP with that base width.
            ``base_width == width`` reduces to SP **bit-identically** (every muP factor
            is exactly ``1.0``), which is the base-width identity the size-generalization
            comparison relies on.
        seed: seeds the parameter init and the per-model data streams.
        eps: Adam's epsilon at the *base* width.
        eps_scaling: muP arm only — divide ``eps`` by ``wm`` on every parameter with an
            infinite dimension (trap T7).  ``False`` reproduces the installed ``mup``,
            which leaves ``eps`` fixed.
    """

    def __init__(self, hps, width, base_width=None, seed=0, device="cuda",
                 batch_size=128, num_conv_layers=3, in_channels=1, num_classes=10,
                 betas=(0.9, 0.999), eps=1e-8, eps_scaling=True):
        self.hps = list(hps)
        self.n_models = len(self.hps)
        self.width = width
        self.base_width = base_width
        self.mup = base_width is not None
        self.wm = (width / base_width) if self.mup else 1.0
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.betas = betas
        self.eps = eps
        self.eps_scaling = eps_scaling
        self.step_count = 0

        # muP's readout division lives in the forward pass; 1.0 in the SP arm
        self.template = ZooCNN(width=width, in_channels=in_channels,
                               num_classes=num_classes,
                               num_conv_layers=num_conv_layers,
                               readout_mult=1.0 / self.wm).to(self.device)
        for p in self.template.parameters():
            p.requires_grad_(False)

        self.param_names = [n for n, _ in self.template.named_parameters()]
        self._NINF = _param_ninf(width, in_channels, num_classes, num_conv_layers)
        self._NINF2 = frozenset(n for n, k in self._NINF.items() if k == 2)
        self._init_params(seed)
        self._build_forwards()

        # per-parameter Adam epsilon (trap T7)
        self.eps_factors = {n: self._eps_factor(n) for n in self.param_names}
        self.eps_values = {n: eps * f for n, f in self.eps_factors.items()}

        dev = self.device
        self.lr = torch.tensor([h.lr for h in self.hps], device=dev)
        self.wd = torch.tensor([h.weight_decay for h in self.hps], device=dev)
        self.dropout_p = torch.tensor([h.dropout for h in self.hps], device=dev)

    # -- initialization -------------------------------------------------------

    def _lr_factor(self, name: str) -> float:
        """muP per-layer lr factor: ``MuAdam``'s rule, ``1/wm`` iff ``ninf == 2``.

        Here that is the two hidden convs only.  The input weight (``ninf 1``), the
        readout (``ninf 1``, whose width correction is the forward multiplier) and every
        bias keep the base lr.
        """
        if not self.mup:
            return 1.0
        return 1.0 / self.wm if name in self._NINF2 else 1.0

    def _eps_factor(self, name: str) -> float:
        """muP per-parameter Adam-``eps`` factor: ``1/wm`` iff ``ninf >= 1`` (trap T7).

        Every parameter with an infinite dimension has a ``Theta(1/c)`` gradient in this
        parameterization, so its ``eps`` has to shrink with it or it stops being
        negligible; ``fc.bias`` (``ninf 0``) has a ``Theta(1)`` gradient and keeps the
        base ``eps``.  Not a rule the ``mup`` package implements — see the module
        docstring.
        """
        if not (self.mup and self.eps_scaling):
            return 1.0
        return 1.0 / self.wm if self._NINF[name] >= 1 else 1.0

    def _init_params(self, seed):
        """Draw per-model initial parameters.

        Weights use the torch default (Kaiming-uniform with ``a=sqrt(5)``, i.e.
        ``U(+-1/sqrt(fan_in))``) times the per-model ``init_mult``; the readout weight
        additionally gets ``wm**+0.5``, which is ``MuReadout._rescale_parameters``.
        Biases are zero.  Composed with the ``1/wm`` forward multiplier this makes the
        *effective* readout variance ``Theta(c**-2)``, as muP requires.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.params, self.lr_factors = {}, {}
        for name, p in self.template.named_parameters():
            shape = (self.n_models,) + tuple(p.shape)
            if name.endswith("bias"):
                t = torch.zeros(shape)
            else:
                fan_in = int(np.prod(p.shape[1:]))
                bound = 1.0 / math.sqrt(fan_in)
                t = torch.empty(shape).uniform_(-bound, bound, generator=gen)
                if name == "fc.weight":
                    t *= self.wm ** 0.5
            self.params[name] = t.to(self.device).requires_grad_(True)
            self.lr_factors[name] = self._lr_factor(name)

        # per-model init multiplier on the weights only (trap T6)
        mult = torch.tensor([h.init_mult for h in self.hps], device=self.device)
        with torch.no_grad():
            for name, t in self.params.items():
                if not name.endswith("bias"):
                    t.mul_(self._bview(mult, t.dim()))

        self.m = {n: torch.zeros_like(t) for n, t in self.params.items()}
        self.v = {n: torch.zeros_like(t) for n, t in self.params.items()}

    @staticmethod
    def _bview(vec, ndim):
        """View a per-model vector so it broadcasts against an ``[N, ...]`` tensor."""
        return vec.view(-1, *([1] * (ndim - 1)))

    # -- forward --------------------------------------------------------------

    def _build_forwards(self):
        def single(params, buffers, x, training):
            return functional_call(self.template, (params, buffers), (x, training))

        # per-model x (each model has its own data stream)
        self._fwd_batched = vmap(single, in_dims=(0, 0, 0, None),
                                 randomness="different")
        # shared x (evaluation on one common test set)
        self._fwd_shared = vmap(single, in_dims=(0, 0, None, None),
                                randomness="different")

    def _buffers(self):
        return {"dropout_p": self.dropout_p}

    def forward(self, x, training=False, shared_x=False):
        """``x``: ``[N, B, 1, 32, 32]`` (or ``[B, 1, 32, 32]`` if ``shared_x``)."""
        fwd = self._fwd_shared if shared_x else self._fwd_batched
        return fwd(self.params, self._buffers(), x, training)

    # -- optimization ---------------------------------------------------------

    def step(self, x, y, alive=None):
        """One Adam step on all models. ``x``: ``[N,B,1,32,32]``, ``y``: ``[N,B]``.

        Returns the per-model mean cross-entropy, ``[N]``.
        """
        logits = self.forward(x, training=True)                     # [N, B, 10]
        n, b, k = logits.shape
        per_sample = F.cross_entropy(logits.reshape(n * b, k), y.reshape(n * b),
                                     reduction="none").view(n, b)
        per_model = per_sample.mean(1)
        # summing per-model means gives each model exactly the gradient of *its own*
        # mean loss, so the per-model lrs below are the only thing coupling them
        per_model.sum().backward()

        self.step_count += 1
        t = self.step_count
        b1, b2 = self.betas
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        with torch.no_grad():
            for name, p in self.params.items():
                g = p.grad
                if g is None:
                    continue
                m, v = self.m[name], self.v[name]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                # eps shrinks with the gradient scale, i.e. by 1/wm unless ninf == 0 (T7)
                upd = (m / bc1) / ((v / bc2).sqrt() + self.eps_values[name])

                lr_eff = self.lr * self.lr_factors[name]
                # decoupled decay uses the BASE lr, never the muP-scaled one (T2)
                delta = (self._bview(lr_eff, p.dim()) * upd
                         + self._bview(self.lr * self.wd, p.dim()) * p)
                if alive is not None:
                    delta = delta * self._bview(alive, p.dim())
                p.sub_(delta)
                p.grad = None
        return per_model.detach()

    # -- evaluation -----------------------------------------------------------

    def _eval_chunk(self):
        return int(max(1, _ACT_BUDGET // (self.n_models * self.width)))

    @torch.no_grad()
    def evaluate(self, x, y):
        """Accuracy and cross-entropy of every model on one shared set.

        ``x``: ``[M, 1, 32, 32]``, ``y``: ``[M]``.  Returns ``(acc [N], ce [N])``.
        """
        correct = torch.zeros(self.n_models, device=self.device)
        ce_sum = torch.zeros(self.n_models, device=self.device)
        chunk = self._eval_chunk()
        for i in range(0, x.shape[0], chunk):
            xb, yb = x[i:i + chunk], y[i:i + chunk]
            logits = self.forward(xb, training=False, shared_x=True)   # [N, m, 10]
            correct += (logits.argmax(-1) == yb).sum(1).float()
            ce_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.repeat(self.n_models), reduction="none"
            ).view(self.n_models, -1).sum(1)
        n = x.shape[0]
        return correct / n, ce_sum / n

    @torch.no_grad()
    def evaluate_per_model(self, images, labels, idx):
        """Accuracy of every model on **its own** index set. ``idx``: ``[N, M]``."""
        correct = torch.zeros(self.n_models, device=self.device)
        chunk = self._eval_chunk()
        m = idx.shape[1]
        for i in range(0, m, chunk):
            sub = idx[:, i:i + chunk]
            logits = self.forward(images[sub], training=False)
            correct += (logits.argmax(-1) == labels[sub]).sum(1).float()
        return correct / m

    # -- export ---------------------------------------------------------------

    def effective_weights(self):
        """Weight tensors of the function each model computes: ``{name: [N, ...]}``.

        Everything except the readout is already the function's weight; the readout
        folds in the forward multiplier, ``W_eff = W / wm`` (trap T3).
        """
        out = {}
        for name in self.param_names:
            t = self.params[name].detach()
            out[name] = t * self.template.readout_mult if name == "fc.weight" else t
        return out

    def export_state_dicts(self):
        """One ``layers.{i}.{weight,bias}`` state dict per model, on CPU.

        The tensors are the **effective** weights of the function the model computes,
        so a plain (multiplier-free) forward pass such as ``zoo_cnn_forward``
        reproduces the trainer's output exactly.  ``check_mup_cnn.py`` V4 asserts it.

        Every per-model tensor is ``clone()``d.  ``w[i]`` is a *view* into the cohort's
        ``[N, ...]`` batch storage, and ``torch.save`` serializes a view's whole
        underlying storage — so without the clone each of the N files written from one
        cohort holds all N models' weights, inflating the zoo by a factor of the cohort
        size (measured: 145 MB per w128 file for 1.19 MB of tensors).
        """
        eff = self.effective_weights()
        order = [f"convs.{i}.weight" for i in range(self.template.num_conv_layers)]
        order += ["fc.weight"]
        bias_order = [f"convs.{i}.bias" for i in range(self.template.num_conv_layers)]
        bias_order += ["fc.bias"]
        ws = [eff[n].cpu() for n in order]
        bs = [eff[n].cpu() for n in bias_order]
        return [wb_to_state_dict([w[i].clone() for w in ws], [b[i].clone() for b in bs])
                for i in range(self.n_models)]

    @torch.no_grad()
    def spectral_norm_components(self):
        """``sqrt(n_{l-1}/n_l) * ||W^(l)||_2`` per layer, per model: ``[N, L]``.

        Same convention as ``scripts/verify_norm_stats.py``, with a convolution
        flattened to ``[c_out, c_in * kh * kw]`` so that ``n_{l-1}`` counts its true
        fan-in.  Reads the **effective** weights: this is a property of the function,
        not of the internal muP variables.
        """
        eff = self.effective_weights()
        order = [f"convs.{i}.weight" for i in range(self.template.num_conv_layers)]
        order += ["fc.weight"]
        cols = []
        for name in order:
            w = eff[name]
            wm = w.reshape(w.shape[0], w.shape[1], -1)          # [N, out, in*k]
            n_out, n_in = wm.shape[1], wm.shape[2]
            sv = torch.linalg.svdvals(wm.float())[:, 0]
            cols.append(math.sqrt(n_in / n_out) * sv)
        return torch.stack(cols, dim=1).cpu()


# ---------------------------------------------------------------------------
# Per-model data streams
# ---------------------------------------------------------------------------

class PerModelSampler:
    """Per-model training subsets and i.i.d. minibatch sampling.

    Model ``m`` sees only the first ``round(train_frac_m * n)`` entries of its own
    permutation of the training pool, and draws minibatches from that subset with
    replacement.  Sampling with replacement rather than epoch-shuffling keeps the
    batched trainer simple; the step budget is still ``epochs * subset / batch``, so
    each model sees the same expected number of examples as an epoch-based schedule.
    """

    def __init__(self, hps, n_train, batch_size, seed, device):
        self.n_models = len(hps)
        self.batch_size = batch_size
        self.device = torch.device(device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        perms = torch.stack([torch.randperm(n_train, generator=g)
                             for _ in range(self.n_models)])
        self.perm = perms.to(self.device)
        self.subset_size = torch.tensor(
            [max(batch_size, int(round(h.train_frac * n_train))) for h in hps],
            device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed + 1)

    def batch_indices(self):
        r = torch.rand(self.n_models, self.batch_size, device=self.device,
                       generator=self.gen)
        j = (r * self.subset_size.unsqueeze(1)).long()
        return torch.gather(self.perm, 1, j)

    def train_eval_indices(self, max_items=2000):
        """A fixed slice of each model's own subset, for the train-accuracy metric."""
        m = int(min(max_items, int(self.subset_size.min().item())))
        return self.perm[:, :m]

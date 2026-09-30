"""The small-CNN-zoo architecture and its channel-widening construction.

Architecture (Unterthiner et al. "small CNN zoo", CIFAR-10 grayscale variant, the
one ScaleGMN's ``predicting_generalization.py`` consumes)::

    [1] --conv3x3/s2--> [c] --conv3x3/s2--> [c] --conv3x3/s2--> [c] --GAP--> dense --> [10]

with ReLU after every convolution.  At ``c = 16`` this has
``160 + 2320 + 2320 + 170 = 4970`` parameters, matching the zoo's ``layout.csv``.
Channels are the width axis of the size-generalization experiments.

State dicts follow the repo convention of :mod:`src.data.fit_inrs`: keys
``layers.{i}.weight`` / ``layers.{i}.bias`` in layer order, with PyTorch shapes
(``[out, in, kh, kw]`` for convolutions, ``[out, in]`` for the dense head).  This
is what ``src/scalegmn/src/data/cifar10_dataset.py`` expects: it filters on
``"weight" in key`` / ``"bias" in key`` and relies on insertion order.

Channel widening
----------------
For each layer ``l`` pick a multiplier ``k_l`` (with ``k_0 = k_L = 1``) and a
factor ``P^(l) in R^{k_l x k_{l-1}}``, and set

    W_up^(l)[(i,b), (j,a), :, :] = W^(l)[i, j, :, :] * P^(l)[b, a]
    b_up^(l)[(i,b)]             = b^(l)[i]

Copies of base channel ``i`` occupy contiguous indices ``[i*k, (i+1)*k)`` — the
same block-Kronecker ordering as ``scripts/check_duplication_equiv.py``.

Function preservation needs only the **row condition** ``P^(l) 1 = 1``: a
convolution is linear in its input channels (``y_i = sum_c W[i,c] * x_c``), so if
the layer-``l-1`` feature maps of the widened net are the base feature maps
repeated ``k_{l-1}`` times, then

    sum_{(j,a)} W[i,j] P[b,a] * x_j = sum_j W[i,j] * x_j * sum_a P[b,a]
                                    = (base pre-activation of channel i)

for every copy ``b``, and ReLU acts elementwise.  Global average pooling is
linear and per-channel, so it commutes with the duplication and the dense head
behaves exactly like an MLP layer (with ``k_L = 1``).

The blockwise generalization drops the Kronecker form: each ``(i, j)`` channel
pair gets its own block ``B_{ij} in R^{k_l x k_{l-1}}`` (per kernel offset),
constrained only by ``B_{ij} 1 = W[i,j] 1``.  ``general-bidir`` additionally
imposes the column condition ``1^T B_{ij} = (k_l / k_{l-1}) W[i,j] 1^T``, which is
what the *bidirectional* GMN needs (see ``paper/main.tex``,
``sec:matrix-product-gmn``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Widening families, in the order the check scripts report them.
FAMILIES = ("uniform", "row-stoch", "doubly-stoch", "general-bidir", "general")

#: Fixed architecture constants of the zoo CNN.
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
NUM_CONV_LAYERS = 3
NUM_CLASSES = 10
IN_CHANNELS = 1


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class ZooCNN(nn.Module):
    """The zoo CNN as an ordinary ``nn.Module`` (no muP scaling).

    Kept as plain ``nn.Conv2d`` / ``nn.Linear`` layers so that
    ``mup.set_base_shapes`` can be applied to it directly (see
    ``scripts/check_mup_cnn.py``): grouped-convolution batching would fuse the
    model index into ``out_channels`` and make ``mup`` infer the wrong
    ``infshape``.

    Dropout probability is a **buffer**, not a constructor argument, so that the
    batched trainer (:mod:`src.data.train_zoo_cnns`) can give every model in a
    ``vmap`` batch its own sampled probability by passing a stacked ``dropout_p``
    through ``torch.func.functional_call``.  It defaults to 0, and buffers are
    invisible to ``state_dict_to_wb`` (which filters on ``weight`` / ``bias``) and
    to ``mup.set_base_shapes`` (which walks parameters), so nothing else changes.
    """

    def __init__(self, width: int = 16, in_channels: int = IN_CHANNELS,
                 num_classes: int = NUM_CLASSES, num_conv_layers: int = NUM_CONV_LAYERS,
                 readout_cls=nn.Linear, readout_mult: float = 1.0):
        super().__init__()
        self.width = width
        self.num_conv_layers = num_conv_layers
        #: multiplier applied to the readout's *input*, i.e. muP's ``1 / width_mult``.
        #: A plain float (not a buffer) because it is per-width, not per-model, so it
        #: needs no ``vmap`` batching; ``1.0`` leaves the module in SP.  Leave it at
        #: ``1.0`` when passing ``readout_cls=MuReadout``, which divides internally.
        self.readout_mult = float(readout_mult)
        convs = []
        c_in = in_channels
        for _ in range(num_conv_layers):
            convs.append(nn.Conv2d(c_in, width, KERNEL_SIZE, stride=STRIDE,
                                   padding=PADDING, bias=True))
            c_in = width
        self.convs = nn.ModuleList(convs)
        self.fc = readout_cls(width, num_classes, bias=True)
        self.register_buffer("dropout_p", torch.zeros(()))

    def forward(self, x, training: bool = False):
        for conv in self.convs:
            x = F.relu(conv(x))
            if training:
                # hand-rolled inverted dropout: F.dropout needs a python float for p,
                # but under vmap p is a batched tensor (one probability per model)
                keep = 1.0 - self.dropout_p
                x = x * (torch.rand_like(x) < keep).to(x.dtype) / keep
        x = x.mean(dim=(-2, -1))          # global average pool
        return self.fc(x * self.readout_mult)

    # -- state-dict conversion ------------------------------------------------

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a ``layers.{i}.{weight,bias}`` state dict on CPU.

        The readout weight is folded with :attr:`readout_mult`, so the exported tensors
        are the **effective** weights of the function this module computes and a plain
        (multiplier-free) forward such as :func:`zoo_cnn_forward` reproduces it.
        """
        weights = [c.weight for c in self.convs] + [self.fc.weight * self.readout_mult]
        biases = [c.bias for c in self.convs] + [self.fc.bias]
        return wb_to_state_dict([w.detach().cpu() for w in weights],
                                [b.detach().cpu() for b in biases])


def state_dict_to_wb(state_dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split a ``layers.{i}.*`` state dict into ordered weight / bias lists."""
    weights = [v for k, v in state_dict.items() if "weight" in k]
    biases = [v for k, v in state_dict.items() if "bias" in k]
    assert len(weights) == len(biases), f"{len(weights)} weights vs {len(biases)} biases"
    return weights, biases


def wb_to_state_dict(weights, biases) -> dict[str, torch.Tensor]:
    """Inverse of :func:`state_dict_to_wb` (insertion order = layer order)."""
    sd = {}
    for i, (w, b) in enumerate(zip(weights, biases)):
        sd[f"layers.{i}.weight"] = w
        sd[f"layers.{i}.bias"] = b
    return sd


def zoo_cnn_forward(weights, biases, x: torch.Tensor) -> torch.Tensor:
    """Functional forward pass of the zoo CNN from raw weight / bias tensors.

    Args:
        weights: ``[out, in, kh, kw]`` per conv layer, then ``[out, in]`` for the head.
        biases:  ``[out]`` per layer.
        x: ``[B, in_channels, H, W]``
    Returns:
        ``[B, num_classes]`` logits.
    """
    h = x
    n_conv = sum(1 for w in weights if w.ndim == 4)
    for i, (w, b) in enumerate(zip(weights, biases)):
        if i < n_conv:
            h = F.relu(F.conv2d(h, w, b, stride=STRIDE, padding=PADDING))
        else:
            if h.ndim == 4:
                h = h.mean(dim=(-2, -1))
            h = h @ w.t() + b
    return h


def layer_layout(width: int, in_channels: int = IN_CHANNELS,
                 num_classes: int = NUM_CLASSES,
                 num_conv_layers: int = NUM_CONV_LAYERS) -> list[int]:
    """Graph layer layout ``[in_channels, c, ..., c, num_classes]``."""
    return [in_channels] + [width] * num_conv_layers + [num_classes]


# ---------------------------------------------------------------------------
# Widening factors P
# ---------------------------------------------------------------------------

def _row_stochastic(k_out, k_in, generator=None, dtype=torch.float32):
    """Random ``P in R^{k_out x k_in}`` with ``P 1 = 1`` (entries may be negative)."""
    p = torch.randn(k_out, k_in, generator=generator, dtype=dtype)
    return p + (1.0 - p.sum(-1, keepdim=True)) / k_in


def _equal_row_col_sums(shape, c, generator=None, dtype=torch.float32):
    """Random matrices with all row sums AND all column sums equal to ``c``.

    ``shape`` is ``(..., k, k)``; ``c`` broadcasts against the leading dims.
    Projection onto that affine subspace (``J = 1 1^T``)::

        B = R - R J / k - J R / k + J R J / k^2 + (c / k) J

    Entries may be negative, and ``c = 0`` is allowed — so this also covers zero
    base weights, which a "``W_ij`` times a doubly stochastic matrix"
    parameterization would miss.
    """
    k = shape[-1]
    assert shape[-2] == k, "equal row/column sums require a square block"
    r = torch.randn(*shape, generator=generator, dtype=dtype)
    j = torch.ones(k, k, dtype=dtype)
    return r - r @ j / k - j @ r / k + j @ r @ j / k ** 2 + c[..., None, None] * j / k


def _doubly_stochastic(k, generator=None, dtype=torch.float32):
    """Random ``P in R^{k x k}`` with ``P 1 = 1`` and ``P^T 1 = 1``."""
    return _equal_row_col_sums((k, k), torch.ones((), dtype=dtype),
                               generator=generator, dtype=dtype)


def _layer_P(family, k_out, k_in, generator=None, dtype=torch.float32):
    """Kronecker factor ``P^(l)`` for one layer.

    On the boundary layers (``k_in == 1`` or ``k_out == 1``) the constraints
    essentially force ``P``, so the families agree there.
    """
    if family == "uniform" or k_in == 1 or k_out == 1:
        if family == "row-stoch" and k_out == 1:
            # row sum 1, but not uniform
            return _row_stochastic(k_out, k_in, generator, dtype)
        return torch.ones(k_out, k_in, dtype=dtype) / k_in
    if family == "row-stoch":
        return _row_stochastic(k_out, k_in, generator, dtype)
    if family == "doubly-stoch":
        assert k_out == k_in, "doubly-stochastic widening needs k_l == k_{l-1}"
        return _doubly_stochastic(k_in, generator, dtype)
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Widening
# ---------------------------------------------------------------------------

def widen_wb(weights, biases, multipliers, family="uniform", generator=None):
    """Widen a zoo CNN's channels.

    Args:
        weights: ``[out, in, kh, kw]`` per conv layer, then ``[out, in]`` for the head.
        biases:  ``[out]`` per layer.
        multipliers: ``[k_0, ..., k_L]`` with ``k_0 = k_L = 1``.
        family: one of :data:`FAMILIES`.
    Returns:
        ``(wide_weights, wide_biases)`` in the same shapes / conventions.
    """
    assert len(multipliers) == len(weights) + 1, \
        f"need {len(weights) + 1} multipliers, got {len(multipliers)}"
    assert multipliers[0] == 1 and multipliers[-1] == 1, \
        "input and output layers are never widened (k_0 = k_L = 1)"

    wide_weights, wide_biases = [], []
    for l, (w, b) in enumerate(zip(weights, biases)):
        k_in, k_out = multipliers[l], multipliers[l + 1]
        is_conv = w.ndim == 4
        w4 = w if is_conv else w[:, :, None, None]           # [out, in, kh, kw]
        n_out, n_in, kh, kw = w4.shape
        dtype = w4.dtype

        if family == "general":
            # blocks[i, j, b, a, s] free apart from sum_a blocks == W[i, j, s]
            blocks = torch.randn(n_out, n_in, k_out, k_in, kh, kw,
                                 generator=generator, dtype=dtype)
            blocks = blocks + (w4[:, :, None, None] - blocks.sum(3, keepdim=True)) / k_in
        elif family == "general-bidir":
            # additionally sum_b blocks == (k_out / k_in) * W[i, j, s]
            if k_in == k_out and k_in > 1:
                # the projection needs the (k_out, k_in) block in the trailing dims
                blocks = _equal_row_col_sums(
                    (n_out, n_in, kh, kw, k_out, k_in), w4,
                    generator=generator, dtype=dtype)
                blocks = blocks.permute(0, 1, 4, 5, 2, 3)    # -> [out, in, k_out, k_in, kh, kw]
            else:
                # boundary layers: the two conditions together force the uniform block
                uni = torch.ones(k_out, k_in, dtype=dtype) / k_in
                blocks = w4[:, :, None, None] * uni[..., None, None]
        else:
            p = _layer_P(family, k_out, k_in, generator, dtype)
            blocks = w4[:, :, None, None] * p[..., None, None]

        # [i, j, b, a, kh, kw] -> [(i,b), (j,a), kh, kw]
        w_up = blocks.permute(0, 2, 1, 3, 4, 5).reshape(
            n_out * k_out, n_in * k_in, kh, kw)
        if not is_conv:
            w_up = w_up[:, :, 0, 0]
        wide_weights.append(w_up)
        wide_biases.append(b.repeat_interleave(k_out))

    return wide_weights, wide_biases


def widen_state_dict(state_dict, multipliers, family="uniform", generator=None):
    """:func:`widen_wb` on a ``layers.{i}.*`` state dict."""
    weights, biases = state_dict_to_wb(state_dict)
    wide_w, wide_b = widen_wb(weights, biases, multipliers, family, generator)
    return wb_to_state_dict(wide_w, wide_b)


def uniform_multipliers(state_dict_or_len, k: int) -> list[int]:
    """``[1, k, ..., k, 1]`` for a network with ``L`` weight layers."""
    n_layers = (state_dict_or_len if isinstance(state_dict_or_len, int)
                else len(state_dict_to_wb(state_dict_or_len)[0]))
    return [1] + [k] * (n_layers - 1) + [1]


def check_family_conditions(weights, wide_weights, multipliers):
    """Which blockwise conditions the widened weights actually satisfy.

    Returns ``(row_err, col_err, nonuniformity)`` as max abs deviations over all
    layers:

    * ``row_err``       forward condition  ``sum_a B_{ij}[b, a] == W[i, j]``
    * ``col_err``       backward condition ``sum_b B_{ij}[b, a] == (k_out/k_in) W[i, j]``
    * ``nonuniformity`` distance from the uniform widening, so a family cannot
      pass the equivalence test vacuously by collapsing to uniform.
    """
    row_err = col_err = nonuniformity = 0.0
    for l, w in enumerate(weights):
        k_in, k_out = multipliers[l], multipliers[l + 1]
        w4 = w if w.ndim == 4 else w[:, :, None, None]
        n_out, n_in, kh, kw = w4.shape
        wide = wide_weights[l]
        wide4 = wide if wide.ndim == 4 else wide[:, :, None, None]
        blocks = wide4.reshape(n_out, k_out, n_in, k_in, kh, kw).permute(0, 2, 1, 3, 4, 5)
        row_err = max(row_err, (blocks.sum(3) - w4[:, :, None]).abs().max().item())
        col_err = max(col_err,
                      (blocks.sum(2) - (k_out / k_in) * w4[:, :, None]).abs().max().item())
        uni = torch.ones(k_out, k_in, dtype=w4.dtype) / k_in
        uniform = w4[:, :, None, None] * uni[..., None, None]
        nonuniformity = max(nonuniformity, (blocks - uniform).abs().max().item())
    return row_err, col_err, nonuniformity

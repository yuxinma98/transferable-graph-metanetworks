"""Matrix-product GMN message function for **convolutional** input networks.

The scalar matrix-product GMN (``gnn_args.message_fn_type: matrix_product``, in
``src/scalegmn/src/scalegmn/gnn_layers.py``) constrains the MSG function to a
multiplication by the raw scalar weight::

    msg_{u -> v} = a_{u,v} * (h_u A),        A in R^{d_in_v x d_hid}

so that with ``aggregator: mean`` the forward aggregate is a matrix product
``Z^(l) = (1 / n_{l-1}) Wbar^(l) H^(l-1) A``.  That is what makes the model
Lipschitz w.r.t. the normalized spectral norm ``sqrt(n_{l-1}/n_l) ||W^(l)||_2``
(``paper/main.tex``, ``sec:matrix-product-gmn``).

On a CNN graph an edge does not carry a scalar: nodes are *channels*, so one edge
carries a whole ``k_h * k_w`` kernel (zero-padded and flattened to R^9 by
``pad_and_flatten_kernel``; dense-layer edges put their scalar in channel 0).
The bilinear generalization keeps one learned map **per kernel offset**::

    msg_{u -> v} = sum_{s=1}^{k_h k_w} a_{u,v}[s] * (h_u A_s),   A in R^{9 x d_in_v x d_hid}

which, again with mean aggregation, is a sum of ``k_h k_w`` matrix products — one
per spatial offset of the kernel::

    Z^(l) = (1 / n_{l-1}) sum_s Wbar_s^(l) H^(l-1) A_s,
                                        Wbar_s^(l) in R^{n_l x n_{l-1}}

so the input weights still enter the metanetwork *only* through matrix
multiplication.  It reduces to the scalar layer when ``k_h k_w == 1``.

Duplication equivalence carries over verbatim, and for the same reason: the
message is linear in the edge feature, so with fan-in rescaling
(``Wbar^(l) = n_{l-1} W^(l)``, see :class:`src.data.cnn_zoo_dataset.FanInCNNZooDataset`)
and mean aggregation, the aggregate over the ``n_{l-1} k_{l-1}`` incoming edges of
a widened node ``(i, b)`` is

    (1 / (n k_in)) sum_j sum_a sum_s [n k_in B_{ij}[b, a, s]] (h_j A_s)
        = sum_j sum_s (sum_a B_{ij}[b, a, s]) (h_j A_s)
        = sum_j sum_s W[i, j, s] (h_j A_s),

which equals the base node's aggregate for **every** widening satisfying the row
condition ``sum_a B_{ij}[b, a, :] = W[i, j, :]`` — not just uniform duplication.
The bidirectional variant additionally needs the column condition.  Both claims
are checked by ``scripts/check_matrix_product_gmn_cnn.py``.

Installation
------------
``src/scalegmn/`` is a git subtree and must not be edited, so the layer is
installed by *name*: ``ScaleGMN_GNN.__init__`` resolves ``GNN_layer`` /
``GNN_layer_aggr`` as globals of the ``src.scalegmn.models`` module at
construction time, so swapping those names there is enough::

    from models.conv_matrix_product_layer import conv_matrix_product_context

    with conv_matrix_product_context(conf["scalegmn_args"]):
        net = ScaleGMN(conf["scalegmn_args"])

The context manager is a no-op unless ``gnn_args.message_fn_type`` is
``matrix_product_conv``, and restores the original names on exit.

Required config (asserted by the layer):
    ``symmetry: permutation``            — defined on top of the plain GMN
    ``graph_init.project_edge_feats: False`` and ``graph_init.d_edge: 9``
                                         — the message consumes raw kernels
    ``gnn_args.update_edge_attr: False`` — no edge states are formed
    ``gnn_args.pos_embed_msg: False`` / ``edge_pos_embed: False``
    ``gnn_args.update_as_act: False``    — plain concatenation UPD
    ``gnn_args.aggregator: mean``        — what makes the aggregate a matrix product
"""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"
if str(SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.gnn_layers import base_GNN_layer  # noqa: E402
from src.scalegmn.layers import MLPNet  # noqa: E402

#: value of ``gnn_args.message_fn_type`` that selects this layer
MESSAGE_FN_TYPE = "matrix_product_conv"

#: number of kernel offsets a CNN-graph edge feature carries (``max_kernel_size``
#: squared; see :func:`src.data.cnn_zoo_dataset.build_cnn_graph_data`)
KERNEL_NUMEL = 9


class ConvMatrixProductMessage(nn.Module):
    """``msg = sum_s edge_attr[:, s] * (x_j @ A[s])``, i.e. one map per kernel offset.

    Held as a submodule (rather than a bare parameter on the layer) so that
    ``base_GNN_layer.reset_parameters`` and ``__repr__``, which both reach for
    ``self.message_fn``, keep working unchanged.
    """

    def __init__(self, d_in_e: int, d_in_v: int, d_out: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(d_in_e, d_in_v, d_out))
        self.reset_parameters()

    def reset_parameters(self):
        # ``nn.Linear``'s bound, with the message's true fan-in (offsets x features):
        # the message sums d_in_e * d_in_v products, not d_in_v.
        d_in_e, d_in_v, _ = self.A.shape
        bound = 1.0 / math.sqrt(d_in_e * d_in_v)
        nn.init.uniform_(self.A, -bound, bound)

    def forward(self, edge_attr, x_j):
        return torch.einsum("es,ed,sdk->ek", edge_attr, x_j, self.A)

    def extra_repr(self):
        d_in_e, d_in_v, d_out = self.A.shape
        return f"A=[{d_in_e}, {d_in_v}, {d_out}] (kernel offsets, d_in_v, d_out)"


def _assert_conv_matrix_product_args(layer, d_in_e):
    """Config sanity checks shared by the two conv matrix-product layers."""
    assert d_in_e > 1, (
        f"conv matrix-product GMN expects raw kernel edge features (d_in_e = k_h*k_w, "
        f"normally {KERNEL_NUMEL}), got {d_in_e}. Set graph_init.project_edge_feats: "
        f"False and graph_init.d_edge: {KERNEL_NUMEL}. For scalar (MLP) input networks "
        f"use message_fn_type: matrix_product instead.")
    assert layer.symmetry == "permutation", (
        "conv matrix-product GMN is defined on top of the plain GMN; use "
        "symmetry: permutation.")
    assert not layer.update_edge_attr, \
        "conv matrix-product GMN carries no edge states; set gnn_args.update_edge_attr: False."
    assert not layer.pos_embed_msg, \
        "conv matrix-product GMN messages take no extra features; set gnn_args.pos_embed_msg: False."
    assert not layer.update_as_act, \
        "conv matrix-product GMN uses the plain concatenation update; set gnn_args.update_as_act: False."


class ConvMatrixProduct_GNN_layer(base_GNN_layer):
    """Conv matrix-product GNN layer (forward variant).

    Same signature as ``GNN_layer`` / ``MatrixProduct_GNN_layer`` so that
    ``ScaleGMN_GNN.__init__`` can construct it in their place.  The UPD function
    is the same concatenation MLP as in the plain GMN.
    """

    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                 last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden,
                 upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden,
                 sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                         last_layer, sign_symmetrization, direction='forward', **kwargs)
        _assert_conv_matrix_product_args(self, d_in_e)

        self.message_fn = ConvMatrixProductMessage(d_in_e, d_in_v, d_hid)
        self.update_node_feats_fn = MLPNet(
            d_hid + d_in_v,
            1 if self.equivariant_gnn and self.last_layer else d_hid,
            kwargs['mlp_args'],
        )

    def message(self, edge_index, x_i, x_j, edge_attr, mask_hidden=None,
                mask_first_layer=None, pos_embed=None, sign_mask=None):
        return self.message_fn(edge_attr, x_j)


class ConvMatrixProduct_GNN_layer_aggr(base_GNN_layer):
    """Conv matrix-product GNN layer for the bidirectional variant (MSG + AGGR only).

    The UPD function is applied externally by ``ScaleGMN_GNN_bidir.forward``, which
    reuses the *forward* edge features with ``bw_edge_index = flip(edge_index)``,
    realising the transposed aggregate
    ``Z^(l),out = (1 / n_{l+1}) sum_s (Wbar_s^(l+1))^T H^(l+1) A_s``.
    """

    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                 last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden,
                 upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden,
                 sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                         last_layer, sign_symmetrization, direction='bidirectional', **kwargs)
        _assert_conv_matrix_product_args(self, d_in_e)

        self.message_fn = ConvMatrixProductMessage(d_in_e, d_in_v, d_hid)

    def message(self, edge_index, x_j, edge_attr, mask_hidden=None,
                mask_first_layer=None, pos_embed=None, sign_mask=None):
        return self.message_fn(edge_attr, x_j)


# ---------------------------------------------------------------------------
# Installation (no subtree edit)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def conv_matrix_product_layers():
    """Swap ``GNN_layer`` / ``GNN_layer_aggr`` in ``src.scalegmn.models``.

    ``ScaleGMN_GNN.__init__`` looks those names up as module globals when it builds
    ``self.gnn_layer``, so any ``ScaleGMN`` constructed inside this block gets the
    conv matrix-product layers instead of the plain GMN's edgewise MLP.  Models
    already built are unaffected, and the names are restored on exit.
    """
    import src.scalegmn.models as sgmn_models

    saved = (sgmn_models.GNN_layer, sgmn_models.GNN_layer_aggr)
    sgmn_models.GNN_layer = ConvMatrixProduct_GNN_layer
    sgmn_models.GNN_layer_aggr = ConvMatrixProduct_GNN_layer_aggr
    try:
        yield
    finally:
        sgmn_models.GNN_layer, sgmn_models.GNN_layer_aggr = saved


def uses_conv_matrix_product(conf) -> bool:
    """Whether a ``scalegmn_args`` dict selects the conv matrix-product MSG."""
    return conf.get("gnn_args", {}).get("message_fn_type") == MESSAGE_FN_TYPE


def conv_matrix_product_context(conf):
    """:func:`conv_matrix_product_layers` if ``conf`` asks for it, else a no-op.

    Safe to wrap every ``ScaleGMN(...)`` construction in — it keys off
    ``gnn_args.message_fn_type``, so the other conditions are untouched.
    """
    if not uses_conv_matrix_product(conf):
        return contextlib.nullcontext()
    assert conf["symmetry"] == "permutation", (
        f"message_fn_type: {MESSAGE_FN_TYPE} requires symmetry: permutation "
        f"(got {conf['symmetry']}).")
    assert not conf["graph_init"].get("project_edge_feats", True), (
        f"message_fn_type: {MESSAGE_FN_TYPE} consumes raw kernel edge features; "
        f"set graph_init.project_edge_feats: False.")
    return conv_matrix_product_layers()

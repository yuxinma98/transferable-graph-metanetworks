"""Forward matrix-product **ScaleGMN** layers (``paper/main.tex``, ``sec:matrix-product-gmn``).

The matrix-product modification of the duplication-compatible plain GMN removes all
learned edge states and constrains the MSG function to a multiplication by the raw
fan-in-rescaled weight (``MatrixProduct_GNN_layer`` in
``src/scalegmn/src/scalegmn/gnn_layers.py``).  The same modification applies to the
duplication-compatible **ScaleGMN**: we drop the learned hidden edge states, but keep
ScaleGMN's node states and ScaleGMN's node update.  For ``l >= 1``::

    Z_sc^(l),in = (1 / n_{l-1}) Wbar^(l) H_sc^(l-1) A_in = W^(l) H_sc^(l-1) A_in
    Htilde_sc^(l) = Psi_h^sc( H_sc^(l), Z_sc^(l),in )

with ``A_in in R^{d_hid x d_hid}`` bias-free and ``Psi_h^sc`` the usual ScaleGMN
:class:`~src.scalegmn.layers.EquivariantNet` update applied row-wise to the
concatenation of the current node state and the incoming aggregate.

Equivalently, this is the restriction of the ScaleGMN forward message rule
``msg = Psi_m( w_e(e) (*) w_v(h) )`` obtained by setting ``w_e(e) = e`` (scalar
multiplication on the node-feature channels), ``w_v(h) = h`` and ``Psi_m(x) = x A_in``.
Because the message is still degree-one homogeneous in both the edge feature and the
source node state, it is scale-equivariant exactly as the original ScaleGMN message is;
because the input weights now enter *only* through matrix multiplication, the model is
Lipschitz w.r.t. the normalized spectral norm ``sqrt(n_{l-1}/n_l) ||W^(l)||_2``.

The fan-in normalization (:class:`src.data.duplication_equiv_dataset.FanInLabeledINRDataset`)
and the layer-wise mean readout (:mod:`models.layerwise_mean_readout`) of the
duplication-compatible ScaleGMN are used unchanged, so there is no baseline variant:
always train with ``--duplication-equiv True``.

CNN input networks
------------------
On a CNN graph an edge carries a whole ``k_h * k_w`` kernel rather than a scalar, so the
message becomes bilinear with one learned map per kernel offset.  That message function
already exists as :class:`models.conv_matrix_product_layer.ConvMatrixProductMessage` and
is reused verbatim here; only the UPD function differs from the plain-GMN conv variant.

Installation
------------
``src/scalegmn/`` is a git subtree and must not be edited, so the layers are installed by
*name*: ``ScaleGMN_GNN.__init__`` resolves ``ScaleEq_GNN_layer`` /
``ScaleGMN_GNN_layer_aggr`` as globals of the ``src.scalegmn.models`` module at
construction time (its ``message_fn_type == 'matrix_product'`` branch asserts
``symmetry: permutation`` and is deliberately *not* reused), so swapping those names is
enough::

    from models.matrix_product_scale_layer import matrix_product_context

    with matrix_product_context(conf["scalegmn_args"]):
        net = ScaleGMN(conf["scalegmn_args"])

:func:`matrix_product_context` is a no-op for every other condition and also dispatches
``matrix_product_conv`` to :func:`models.conv_matrix_product_layer.conv_matrix_product_context`,
so it is a drop-in superset of that context manager.

Required config (asserted):
    ``symmetry: scale``                  — ScaleGMN node states and node update
    ``graph_init.project_edge_feats: False``
        and ``graph_init.d_edge: 1`` (MLP) / ``9`` (CNN)  — raw edge features
    ``gnn_args.update_edge_attr: False`` — no edge states are formed
    ``gnn_args.pos_embed_msg: False`` / ``edge_pos_embed: False``
    ``gnn_args.update_as_act: False``    — the EquivariantNet UPD, not SineUpdate
    ``gnn_args.aggregator: mean``        — what makes the aggregate a matrix product
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"
if str(SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.gnn_layers import base_GNN_layer  # noqa: E402
from src.scalegmn.layers import EquivariantNet  # noqa: E402

from models.conv_matrix_product_layer import (  # noqa: E402
    KERNEL_NUMEL,
    ConvMatrixProductMessage,
    conv_matrix_product_context,
    uses_conv_matrix_product,
)

#: ``gnn_args.message_fn_type`` selecting the scalar (MLP/INR) variant
MESSAGE_FN_TYPE = "matrix_product_scale"

#: ``gnn_args.message_fn_type`` selecting the conv (CNN) variant
MESSAGE_FN_TYPE_CONV = "matrix_product_conv_scale"

MESSAGE_FN_TYPES = (MESSAGE_FN_TYPE, MESSAGE_FN_TYPE_CONV)


def _assert_matrix_product_scale_args(layer, d_in_e, conv):
    """Config sanity checks shared by the two matrix-product ScaleGMN layers."""
    name = MESSAGE_FN_TYPE_CONV if conv else MESSAGE_FN_TYPE
    if conv:
        assert d_in_e > 1, (
            f"message_fn_type: {name} expects raw kernel edge features "
            f"(d_in_e = k_h*k_w, normally {KERNEL_NUMEL}), got {d_in_e}. Set "
            f"graph_init.project_edge_feats: False and graph_init.d_edge: {KERNEL_NUMEL}. "
            f"For scalar (MLP) input networks use message_fn_type: {MESSAGE_FN_TYPE}.")
    else:
        assert d_in_e == 1, (
            f"message_fn_type: {name} requires raw scalar edge features (d_in_e == 1), "
            f"got {d_in_e}. Set graph_init.project_edge_feats: False and "
            f"graph_init.d_edge: 1. For convolutional input networks use "
            f"message_fn_type: {MESSAGE_FN_TYPE_CONV}.")
    assert layer.symmetry != "permutation", (
        f"message_fn_type: {name} keeps the ScaleGMN node states and update; use "
        f"symmetry: scale. For the plain-GMN variant use message_fn_type: "
        f"{'matrix_product_conv' if conv else 'matrix_product'}.")
    assert not layer.update_edge_attr, \
        f"message_fn_type: {name} carries no edge states; set gnn_args.update_edge_attr: False."
    assert not layer.pos_embed_msg, \
        f"message_fn_type: {name} messages take no extra features; set gnn_args.pos_embed_msg: False."
    assert not layer.update_as_act, \
        f"message_fn_type: {name} uses the EquivariantNet update; set gnn_args.update_as_act: False."


def _build_message_fn(d_in_e, d_in_v, d_hid, conv):
    """``A_in``: the only learned map applied to the message.

    Scalar variant: ``msg = a_{u,v} * (h_u A_in)``, a bias-free ``nn.Linear`` exactly as
    in ``MatrixProduct_GNN_layer``.  Conv variant: one map per kernel offset (see
    :class:`models.conv_matrix_product_layer.ConvMatrixProductMessage`).
    """
    if conv:
        return ConvMatrixProductMessage(d_in_e, d_in_v, d_hid)
    return nn.Linear(d_in_v, d_hid, bias=False)


class MatrixProductScale_GNN_layer(base_GNN_layer):
    """Forward matrix-product ScaleGMN layer.

    Same constructor signature as ``ScaleEq_GNN_layer`` so that
    ``ScaleGMN_GNN.__init__`` can build it in that class's place.  MSG is the
    matrix-product message; UPD is ScaleGMN's ``EquivariantNet``, which
    ``base_GNN_layer.forward`` already calls on ``cat(x, aggregated)`` with the masks it
    needs -- so no ``forward`` override is required.
    """

    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                 last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden,
                 upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden,
                 sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                         last_layer, sign_symmetrization, direction='forward', **kwargs)
        self.conv = kwargs.get('message_fn_type') == MESSAGE_FN_TYPE_CONV
        _assert_matrix_product_scale_args(self, d_in_e, self.conv)

        self.message_fn = _build_message_fn(d_in_e, d_in_v, d_hid, self.conv)

        # Psi_h^sc: the ScaleGMN UPD function, verbatim from ScaleEq_GNN_layer's
        # non-update_as_act branch.
        self.update_node_feats_fn = EquivariantNet(
            kwargs['update_node_feats_fn_layers'],
            d_hid + d_in_v,
            1 if self.equivariant_gnn and self.last_layer else d_hid,
            kwargs['mlp_args'],
            d_extra=d_hid * self.pos_embed_upd,
            symmetry=symmetry,
            equiv_on_hidden=upd_equiv_on_hidden,
            layer_equiv_on_hidden=layer_upd_equiv_on_hidden,
            mlp_on_io=kwargs['mlp_on_io'],
            num_mlps=upd_num_mlps,
            skip_connections=kwargs['update_node_feats_fn_skip_connections'],
            sign_symmetrization=sign_symmetrization)

    def message(self, edge_index, x_i, x_j, edge_attr, mask_hidden=None,
                mask_first_layer=None, pos_embed=None, sign_mask=None):
        if self.conv:
            return self.message_fn(edge_attr, x_j)
        return edge_attr * self.message_fn(x_j)


class MatrixProductScale_GNN_layer_aggr(base_GNN_layer):
    """Matrix-product ScaleGMN layer for the bidirectional variant (MSG + AGGR only).

    The UPD function is applied externally by ``ScaleGMN_GNN_bidir.forward``, which
    reuses the *forward* edge features with ``bw_edge_index = flip(edge_index)``,
    realising the transposed aggregate
    ``Z^(l),out = (1 / n_{l+1}) (Wbar^(l+1))^T H^(l+1) A_out``.

    Provided so that the verification scripts can contrast the forward and
    bidirectional widening-equivalence classes; no experiment uses it (the paper
    defines only the forward variant).
    """

    def __init__(self, d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                 last_layer, msg_equiv_on_hidden, msg_num_mlps, upd_equiv_on_hidden,
                 upd_num_mlps, layer_msg_equiv_on_hidden, layer_upd_equiv_on_hidden,
                 sign_symmetrization, **kwargs):
        super().__init__(d_in_v, d_in_e, d_hid, layer_layout, update_edge_attr, symmetry,
                         last_layer, sign_symmetrization, direction='bidirectional', **kwargs)
        self.conv = kwargs.get('message_fn_type') == MESSAGE_FN_TYPE_CONV
        _assert_matrix_product_scale_args(self, d_in_e, self.conv)

        self.message_fn = _build_message_fn(d_in_e, d_in_v, d_hid, self.conv)

    def message(self, edge_index, x_j, edge_attr, mask_hidden=None,
                mask_first_layer=None, pos_embed=None, sign_mask=None):
        if self.conv:
            return self.message_fn(edge_attr, x_j)
        return edge_attr * self.message_fn(x_j)


# ---------------------------------------------------------------------------
# Installation (no subtree edit)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def matrix_product_scale_layers():
    """Swap ``ScaleEq_GNN_layer`` / ``ScaleGMN_GNN_layer_aggr`` in ``src.scalegmn.models``.

    ``ScaleGMN_GNN.__init__`` looks those names up as module globals when it builds
    ``self.gnn_layer``, so any ``ScaleGMN`` constructed inside this block gets the
    matrix-product ScaleGMN layers instead of ScaleGMN's edgewise ``EquivariantNet``
    message.  Models already built are unaffected, and the names are restored on exit.
    """
    import src.scalegmn.models as sgmn_models

    saved = (sgmn_models.ScaleEq_GNN_layer, sgmn_models.ScaleGMN_GNN_layer_aggr)
    sgmn_models.ScaleEq_GNN_layer = MatrixProductScale_GNN_layer
    sgmn_models.ScaleGMN_GNN_layer_aggr = MatrixProductScale_GNN_layer_aggr
    try:
        yield
    finally:
        sgmn_models.ScaleEq_GNN_layer, sgmn_models.ScaleGMN_GNN_layer_aggr = saved


def uses_matrix_product_scale(conf) -> bool:
    """Whether a ``scalegmn_args`` dict selects a matrix-product ScaleGMN MSG."""
    return conf.get("gnn_args", {}).get("message_fn_type") in MESSAGE_FN_TYPES


def matrix_product_context(conf):
    """The right layer-swap context for ``conf``, or a no-op.

    Dispatches on ``gnn_args.message_fn_type``:

    * ``matrix_product_scale`` / ``matrix_product_conv_scale`` -> this module's layers
    * ``matrix_product_conv`` -> :func:`models.conv_matrix_product_layer.conv_matrix_product_context`
    * anything else (including the in-subtree ``matrix_product``) -> ``nullcontext``

    Safe to wrap every ``ScaleGMN(...)`` construction in.
    """
    if not uses_matrix_product_scale(conf):
        return conv_matrix_product_context(conf)
    name = conf["gnn_args"]["message_fn_type"]
    assert conf["symmetry"] == "scale", (
        f"message_fn_type: {name} requires symmetry: scale (got {conf['symmetry']}).")
    assert not conf["graph_init"].get("project_edge_feats", True), (
        f"message_fn_type: {name} consumes raw edge features; set "
        f"graph_init.project_edge_feats: False.")
    assert not conf.get("edge_pos_embed", False), (
        f"message_fn_type: {name} keeps the edge feature raw; set edge_pos_embed: False.")
    return matrix_product_scale_layers()

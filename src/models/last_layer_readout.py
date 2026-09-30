"""Width-agnostic last-layer readout.

``readout_range: last_layer`` (what the upstream cifar10 configs use for the
accuracy-prediction task) pools only the output-layer node features.  That is
already **duplication-invariant**, because the output layer is never widened
(``k_L = 1``): with fan-in rescaling + mean aggregation the output nodes' GNN
representations are equal between a base net and any of its channel-widened
copies, so a readout that reads only those nodes is too.  No layer-wise mean
pooling is needed — see :mod:`src.models.layerwise_mean_readout` for the
alternative that also survives ``readout_range: full_graph``.

The *upstream implementation* of that readout is not width-agnostic, though.
Both ``ScaleGMN_GNN_fw.forward`` and ``ScaleGMN_GNN_bidir.forward`` do

    node_features = x.reshape(batch.num_graphs, num_nodes, x.shape[-1])
    node_features = node_features[:, -self.last_layer_nodes:]

with ``num_nodes`` fixed at construction from the *train* layout (the source even
carries the comment "change if processing varying architectures"), so evaluating
a wider CNN raises a reshape error.  ``LastLayerReadout`` below selects the
output nodes from ``node2type`` instead, which is width-independent, and is
installed into the ``readout_range: full_graph`` slot (i.e. the config must say
``full_graph`` so that ``ScaleGMN_GNN`` takes the branch that calls
``self.readout(x, batch=..., mask_hidden=..., sign_mask=...)``).

Selection rule: ``get_node_types([n_in, c, ..., c, n_out])`` gives each input node
its own type ``0..n_in-1``, each hidden layer one shared type, and each output
node its own type — the ``n_out`` largest ones.  So the output nodes are exactly
those with ``node2type > max(node2type) - n_out``, at every width.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"
if str(SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.mlp import mlp  # noqa: E402


class LastLayerReadout(nn.Module):
    """Concatenate the output-layer node features and apply an MLP.

    Interface matches ``PermScaleInvariantReadout.forward`` so it can be dropped
    into ``gnn.readout``::

        forward(x, batch, mask_hidden=None, sign_mask=None)

    ``node2type`` must be stashed on the module before each forward pass; this is
    what :func:`install_last_layer_readout` wires up.
    """

    def __init__(self, d_in, d_out, mlp_args, num_output_nodes,
                 num_input_nodes=0, include_input_nodes=False):
        super().__init__()
        self.num_output_nodes = num_output_nodes
        self.num_input_nodes = num_input_nodes
        self.include_input_nodes = include_input_nodes
        n_read = num_output_nodes + (num_input_nodes if include_input_nodes else 0)
        self.rho = mlp(in_features=d_in * n_read, out_features=d_out, **mlp_args)
        self._node2type: torch.Tensor | None = None

    def forward(self, x, batch, mask_hidden=None, sign_mask=None):
        assert self._node2type is not None, (
            "LastLayerReadout._node2type must be set before forward. "
            "Use install_last_layer_readout() to wrap the model.")
        node2type = self._node2type
        num_graphs = int(batch.max().item()) + 1

        max_type = int(node2type.max().item())
        out_mask = node2type > max_type - self.num_output_nodes
        feats = [self._gather(x, out_mask, num_graphs, self.num_output_nodes)]
        if self.include_input_nodes:
            in_mask = node2type < self.num_input_nodes
            feats.append(self._gather(x, in_mask, num_graphs, self.num_input_nodes))

        return self.rho(torch.cat(feats, dim=-1))

    @staticmethod
    def _gather(x, mask, num_graphs, n_per_graph):
        """Select masked nodes and reshape to ``[num_graphs, n_per_graph * d]``.

        PyG concatenates graphs in order and preserves within-graph node order, so
        boolean masking keeps the ``(graph, node)`` ordering intact.
        """
        selected = x[mask]
        assert selected.shape[0] == num_graphs * n_per_graph, (
            f"expected {num_graphs * n_per_graph} selected nodes, "
            f"got {selected.shape[0]}")
        return selected.reshape(num_graphs, n_per_graph * x.shape[-1])


def install_last_layer_readout(model, conf, include_input_nodes=False):
    """Replace a ScaleGMN model's readout with :class:`LastLayerReadout`.

    The config must use ``readout_range: full_graph`` so that the GNN forward
    takes the ``self.readout(x, batch=...)`` branch; the readout module the model
    built there is discarded.

    Args:
        model: a ``ScaleGMN`` instance (invariant variant).
        conf: the ``scalegmn_args`` config dict.
        include_input_nodes: also read the input-layer nodes (the bidirectional
            upstream variant concatenates first- and last-layer features).

    Returns:
        The modified model (in place).
    """
    gnn = model.gnn
    assert not getattr(gnn, "only_last_layer", False), (
        "install_last_layer_readout expects readout_range: full_graph in the config "
        "(the last_layer branch of ScaleGMN_GNN.forward is not width-agnostic).")

    layer_layout = conf["layer_layout"]
    new_readout = LastLayerReadout(
        d_in=conf["gnn_args"]["d_hid"],
        d_out=conf["readout_args"]["d_out"],
        mlp_args=conf["mlp_args"],
        num_output_nodes=layer_layout[-1],
        num_input_nodes=layer_layout[0],
        include_input_nodes=include_input_nodes,
    )
    new_readout = new_readout.to(next(gnn.parameters()).device)
    gnn.readout = new_readout

    original_forward = gnn.forward

    def wrapped_forward(batch, num_nodes=None, pos_embed=None, pos_embed_edge=None):
        gnn.readout._node2type = batch.node2type
        return original_forward(batch, num_nodes=num_nodes, pos_embed=pos_embed,
                                pos_embed_edge=pos_embed_edge)

    gnn.forward = wrapped_forward
    return model

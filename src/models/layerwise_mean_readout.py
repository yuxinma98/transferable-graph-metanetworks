"""Layer-wise mean readout for duplication-equivariant ScaleGMN.

Replaces PermScaleInvariantReadout's sum pooling with per-layer mean pooling
over hidden nodes.  This ensures graph-level output is invariant to uniform
width duplication: each widened layer has k_l identical node representations,
and mean(k copies of x) = x.

Interface matches PermScaleInvariantReadout.forward():
    forward(x, batch, mask_hidden, sign_mask=None)

The readout needs node2type to identify which layer each node belongs to.
Before each forward pass, the caller must set self._node2type = batch.node2type.
This is done by install_layerwise_mean_readout() which wraps the GNN forward.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, scatter

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"
if str(SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.layers import InvariantLayer  # noqa: E402
from src.scalegmn.mlp import mlp  # noqa: E402


class LayerWiseMeanReadout(nn.Module):
    """Duplication-equivariant readout: per-layer mean pooling + I/O features."""

    def __init__(self, d_in, d_rho, d_out, mlp_args, sym, num_io_nodes,
                 sign_symmetrization=True, num_hidden_layers=None):
        """
        Args:
            d_in: input feature dim (d_hid of GNN)
            d_rho: intermediate dimension for invariant layer output
            d_out: number of classes
            mlp_args: kwargs for MLP construction
            sym: symmetry type ('sign', 'scale', 'hetero', 'permutation')
            num_io_nodes: number of input + output nodes
            sign_symmetrization: apply sign canonicalization
            num_hidden_layers: number of hidden layers in the INR (e.g. 2 for
                [2, W, W, 1]). If None, global_mlp is built lazily.
        """
        super().__init__()
        self.symmetry = sym
        self.sign_symmetrization = sign_symmetrization
        self.num_hidden_layers = num_hidden_layers
        self._d_rho = d_rho
        self._d_out = d_out
        self._num_io_nodes = num_io_nodes
        self._mlp_args = mlp_args

        if self.symmetry == 'permutation':
            self.g1 = mlp(in_features=d_in, out_features=d_rho, **mlp_args)
        elif self.symmetry != 'hetero':
            self.g1 = InvariantLayer(d_in, d_rho, mlp_args, sym, sign_symmetrization)
        else:
            self.g_sign = InvariantLayer(d_in, d_rho, mlp_args, 'sign', sign_symmetrization, final_mlp=True)
            self.g_scale = InvariantLayer(d_in, d_rho, mlp_args, 'scale', final_mlp=True)

        if num_hidden_layers is not None:
            global_in = d_rho * num_hidden_layers + d_rho * num_io_nodes
            self.global_mlp = mlp(in_features=global_in, out_features=d_out, **mlp_args)
        else:
            self.global_mlp = None

        self._node2type: torch.Tensor | None = None

    def forward(self, x, batch, mask_hidden=None, sign_mask=None):
        """Same call signature as PermScaleInvariantReadout.

        Args:
            x: node features [N, d_hid]
            batch: graph membership index [N] (batch.batch from PyG)
            mask_hidden: [N, 1] bool mask (True for hidden nodes)
            sign_mask: optional [N, 1] for hetero symmetry
        """
        assert self._node2type is not None, (
            "LayerWiseMeanReadout._node2type must be set before forward. "
            "Use install_layerwise_mean_readout() to wrap the model.")

        node2type = self._node2type

        if self.symmetry == 'permutation':
            x_canon = self.g1(x)
        elif self.symmetry == 'hetero':
            x_canon = sign_mask * self.g_sign(x) + (~sign_mask) * self.g_scale(x)
        else:
            x_canon = self.g1(x)

        hidden_mask_flat = mask_hidden.squeeze(-1)  # [N]
        num_graphs = batch.max().item() + 1

        # I/O node features (same approach as PermScaleInvariantReadout)
        io_nodes = to_dense_batch(x[~hidden_mask_flat], batch[~hidden_mask_flat])[0]
        io_nodes = io_nodes.reshape(num_graphs, -1)  # [B, num_io * d_rho]

        # Layer-wise mean over hidden nodes
        hidden_x = x_canon[hidden_mask_flat]          # [N_hidden, d_rho]
        hidden_batch = batch[hidden_mask_flat]         # [N_hidden]
        hidden_types = node2type[hidden_mask_flat]     # [N_hidden]

        layer_types = hidden_types.unique(sorted=True)
        n_layers = layer_types.shape[0]

        # Map type IDs to contiguous offsets [0, n_layers)
        type_to_offset = torch.zeros(layer_types.max().item() + 1,
                                     dtype=torch.long, device=x.device)
        for i, t in enumerate(layer_types):
            type_to_offset[t] = i

        # Composite scatter index: one slot per (graph, layer)
        composite_idx = hidden_batch * n_layers + type_to_offset[hidden_types]
        layer_means = scatter(hidden_x, composite_idx, dim=0, reduce='mean',
                              dim_size=num_graphs * n_layers)

        layer_means = layer_means.reshape(num_graphs, n_layers * hidden_x.shape[-1])
        graph_emb = torch.cat((layer_means, io_nodes), dim=-1)

        if self.global_mlp is None:
            global_in = graph_emb.shape[-1]
            self.global_mlp = mlp(
                in_features=global_in, out_features=self._d_out, **self._mlp_args
            ).to(x.device)

        return self.global_mlp(graph_emb)


def install_layerwise_mean_readout(model, conf):
    """Replace the ScaleGMN model's readout with LayerWiseMeanReadout.

    Wraps the GNN's forward method so that node2type is stashed on the readout
    before each forward call.

    Args:
        model: a ScaleGMN instance (invariant variant)
        conf: the full scalegmn_args config dict

    Returns:
        The modified model (in-place).
    """
    gnn = model.gnn
    layer_layout = conf['layer_layout']
    num_hidden_layers = len(layer_layout) - 2  # exclude input and output layers
    num_io_nodes = layer_layout[0] + layer_layout[-1]

    new_readout = LayerWiseMeanReadout(
        d_in=conf['gnn_args']['d_hid'],
        d_rho=conf['readout_args']['d_rho'],
        d_out=conf['readout_args']['d_out'],
        mlp_args=conf['mlp_args'],
        sym=conf['symmetry'],
        num_io_nodes=num_io_nodes,
        sign_symmetrization=conf['gnn_args']['sign_symmetrization'],
        num_hidden_layers=num_hidden_layers,
    )

    device = next(gnn.parameters()).device
    new_readout = new_readout.to(device)
    gnn.readout = new_readout

    original_forward = gnn.forward

    def wrapped_forward(batch, num_nodes=None, pos_embed=None, pos_embed_edge=None):
        gnn.readout._node2type = batch.node2type
        return original_forward(batch, num_nodes=num_nodes, pos_embed=pos_embed,
                                pos_embed_edge=pos_embed_edge)

    gnn.forward = wrapped_forward
    return model

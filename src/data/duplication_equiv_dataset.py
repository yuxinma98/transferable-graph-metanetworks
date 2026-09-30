"""
Dataset that rescales MLP weights by fan-in for duplication-equivalence.

The widening construction (W_up = W ⊗ P, b_up = b ⊗ 1_k) produces a functionally
equivalent MLP for any row-stochastic P (rows sum to 1).  For a MPNN with **mean**
aggregation to respect this equivalence, edge features must be initialized as

    W_bar^(l) = d_{l-1} * W^(l),

where d_{l-1} = layer_layout[l-1] is the fan-in of layer l.

Why this works: in the widened graph, edge (v_{j,a}^{(l-1)}, v_{i,b}^{(l)}) has feature
    e^up_{j,a -> i,b} = d_{l-1} * k_{l-1} * W_{ij} * P_{ba}.
Mean over the k_{l-1} copies of predecessor j:
    (1/k_{l-1}) * sum_a e^up = d_{l-1} * W_{ij} * sum_a P_{ba} = d_{l-1} * W_{ij} = e_{j->i}.
So the MPNN's mean-aggregated message at each widened node equals the base node's, and
by induction the full GNN output is identical for equivalent MLPs of different widths.

This module provides FanInLabeledINRDataset, a drop-in replacement for
LabeledINRDataset that applies the fan-in rescaling at graph construction time.
"""

import sys
from pathlib import Path

# Ensure ScaleGMN root is on sys.path so its internal "from src.data import ..."
# imports resolve correctly.  __file__ is at PROJECT_ROOT/src/data/; go up two
# levels to reach PROJECT_ROOT/src/, then into scalegmn/.
_SCALEGMN_ROOT = Path(__file__).parent.parent / "scalegmn"
if str(_SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCALEGMN_ROOT))

from src.data.mnist_inr_dataset import LabeledINRDataset  # noqa: E402  (ScaleGMN src)


class FanInLabeledINRDataset(LabeledINRDataset):
    """Drop-in replacement for LabeledINRDataset with fan-in weight rescaling.

    Overrides batch_to_graphs() to multiply each weight matrix W^(l) by its fan-in
    d_{l-1} before building the edge-feature matrix.  Use together with mean aggregation
    (aggregator: mean in the GNN config) for duplication-equivariant training.

    All constructor arguments are identical to LabeledINRDataset.
    """

    def batch_to_graphs(self, weights, biases, input_emb=None, **kwargs):
        """Build graph with edge features = d_{l-1} * W^(l) (fan-in rescaling).

        Args:
            weights: tuple of L tensors, each [num_in, num_out, 1]
                     (weight matrix for layer l, stored as [in, out] per ScaleGMN convention)
            biases:  tuple of L tensors, each [num_out, 1]
        """
        # weights[i].shape[0] == num_in == d_{l-1} for layer i+1
        scaled_weights = tuple(w * w.shape[0] for w in weights)
        return super().batch_to_graphs(scaled_weights, biases, input_emb, **kwargs)

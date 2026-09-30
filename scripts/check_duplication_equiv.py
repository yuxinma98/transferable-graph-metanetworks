"""
Check that ScaleGMN with fan-in weight rescaling and mean aggregation produces
identical representations for functionally equivalent MLPs at different widths.

Widening construction (pure duplication / uniform P):
------------------------------------------------------
Given a base MLP with layer layout [d_0, n_1, ..., n_{L-1}, d_L] and width
multipliers k_0=k_L=1, k_1,...,k_{L-1} ∈ N, the widened MLP has layout
[d_0, n_1*k_1, ..., n_{L-1}*k_{L-1}, d_L] with

    W^(l)_up = W^(l) ⊗ P^(l),   P^(l) = (1/k_{l-1}) * ones_{k_l × k_{l-1}}
    b^(l)_up = b^(l) ⊗ ones_{k_l}

Edge-feature equivalence with fan-in rescaling:
-----------------------------------------------
After fan-in rescaling (edge feature = d_{l-1} * W^(l)), the widened edge feature is

    d_{l-1}*k_{l-1} * W_{ij} * P_{ba} = d_{l-1}*k_{l-1} * W_{ij} * (1/k_{l-1})
                                       = d_{l-1} * W_{ij}   (= base edge feature)

So edge features are INDIVIDUALLY equal. Since all k_{l-1} messages into each widened
node are identical, mean aggregation gives the same aggregated message as in the base
graph. By induction over layers, every copy of base node j in the widened graph gets
the same GNN node representation as j in the base graph.

Part 1 — Node-level tests:
---------------------------
Three conditions tested per direction (forward / bidirectional; --forward-only drops the
bidirectional half, for models that have no bidirectional variant — mpsgmn_*):
  1. fan-in rescaling + mean aggregation  →  PASS  (< tol)
  2. no fan-in rescaling + mean aggregation → FAIL  (wrong edge scale)
  3. fan-in rescaling + sum aggregation   →  FAIL  (messages accumulate k copies)

Part 2 — Graph-level output tests (with LayerWiseMeanReadout):
--------------------------------------------------------------
The standard PermScaleInvariantReadout uses SUM pooling over hidden nodes (via
DeepSetsAggregation), which scales with k_l and breaks graph-level equivalence.
LayerWiseMeanReadout replaces this with per-layer MEAN pooling:
  mean(k copies of x) = x  →  graph-level output is invariant to duplication.

Conditions tested, per direction — the full condition and each of its three
components switched off in turn:
  1. fan-in + mean aggr + layer-wise mean readout → PASS  (full end-to-end equiv)
  2. no fan-in + mean aggr + layer-wise mean readout → FAIL  (edge scale wrong)
  3. fan-in + sum aggr  + layer-wise mean readout → FAIL  (node repr differ)
  4. fan-in + mean aggr + sum-pooling readout    → FAIL  (pooling scales with k)

Usage
-----
    python scripts/check_duplication_equiv.py
    python scripts/check_duplication_equiv.py --n-inrs 20 --k-max 4
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch_geometric
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.models import ScaleGMN                          # noqa: E402
from src.data.data_utils import (                                  # noqa: E402
    get_node_types, get_edge_types, nn_to_edge_index, to_pyg_batch,
)
from src.utils.helpers import overwrite_conf                       # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from data.duplication_equiv_dataset import FanInLabeledINRDataset  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context  # noqa: E402


# ---------------------------------------------------------------------------
# Weight / graph helpers
# ---------------------------------------------------------------------------

def state_dict_to_wb(state_dict):
    """Extract (weights, biases) in ScaleGMN [num_in, num_out, 1] convention."""
    weights = tuple(v.permute(1, 0).unsqueeze(-1) for k, v in state_dict.items() if "weight" in k)
    biases  = tuple(v.unsqueeze(-1)               for k, v in state_dict.items() if "bias"   in k)
    return weights, biases


def widen_wb(weights, biases, multipliers):
    """Widen weights/biases via uniform-P Kronecker duplication.

    P^(l) = (1/k_{l-1}) * ones_{k_l × k_{l-1}}.

    Weights are in ScaleGMN's [n_in, n_out, 1] convention.  The Kronecker
    product W^(l)_up = W^(l) ⊗ P^(l) is defined on the [out, in] transpose:
        kron(W^T, P)  ∈ R^{N_out × N_in},   then transposed back to [N_in, N_out].
    This ensures the widened MLP is functionally equivalent to the base:
        W^(l)_up^T @ (x ⊗ 1_k) = (W^(l)^T @ x) ⊗ (P @ 1_k) = (W^(l)^T @ x) ⊗ 1_{k_out}.
    """
    L = len(weights)
    assert len(multipliers) == L + 1
    assert multipliers[0] == 1 and multipliers[-1] == 1

    wide_weights, wide_biases = [], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        n_in, n_out, _ = W.shape
        k_in  = multipliers[l]
        k_out = multipliers[l + 1]
        P     = torch.ones(k_out, k_in) / k_in            # uniform row-stochastic

        # kron(W^T [n_out, n_in], P [k_out, k_in]) → [N_out, N_in]; transpose → [N_in, N_out]
        W_up_T = torch.kron(W.squeeze(-1).t().contiguous(), P)  # [n_out*k_out, n_in*k_in]
        W_up   = W_up_T.t().unsqueeze(-1)                       # [n_in*k_in, n_out*k_out, 1]

        # b^(l) ⊗ 1_{k_out}: kron([n_out], [k_out]) = [n_out*k_out]
        b_up = torch.kron(b.squeeze(-1), torch.ones(k_out)).unsqueeze(-1)

        wide_weights.append(W_up)
        wide_biases.append(b_up)

    return tuple(wide_weights), tuple(wide_biases)


def infer_layout(weights):
    """Infer layer layout from weight tuple."""
    return [weights[0].shape[0]] + [w.shape[1] for w in weights]


def batch_to_graphs(weights, biases):
    """Build dense node/edge feature matrices from weights/biases.

    Mirrors BaseDataset.batch_to_graphs (no fan-in rescaling).
    weights: tuple of [num_in, num_out, 1]
    biases:  tuple of [num_out, 1]
    """
    num_nodes    = weights[0].shape[0] + sum(w.shape[1] for w in weights)
    node_features = torch.zeros(num_nodes, biases[0].shape[-1])
    edge_features = torch.zeros(num_nodes, num_nodes, weights[0].shape[-1])

    row_off = 0
    col_off = weights[0].shape[0]
    for w in weights:
        n_in, n_out, _ = w.shape
        edge_features[row_off:row_off + n_in, col_off:col_off + n_out] = w
        row_off += n_in
        col_off += n_out

    row_off = weights[0].shape[0]
    node_features[:row_off] = 1.0          # input nodes initialised to 1
    for b in biases:
        n_out, _ = b.shape
        node_features[row_off:row_off + n_out] = b
        row_off += n_out

    return node_features, edge_features


def build_pyg_data(weights, biases, layout, fanin_rescale, direction='forward'):
    """Construct a PyG Data object for one MLP.

    Args:
        fanin_rescale: multiply each W^(l) by its fan-in d_{l-1} before
                       building edge features (required for duplication equiv).
        direction: 'forward' or 'bidirectional'. Bidirectional adds bw_edge_index
                   (torch.flip of edge_index) to the Data object.
    """
    if fanin_rescale:
        weights = tuple(w * w.shape[0] for w in weights)

    node_feats, edge_feats = batch_to_graphs(weights, biases)

    edge_index = nn_to_edge_index(layout, device="cpu")
    node2type  = get_node_types(layout)
    edge2type  = get_edge_types(layout)
    hidden_mask = torch.tensor(
        [False] * layout[0] +
        [True]  * sum(layout[1:-1]) +
        [False] * layout[-1]
    ).unsqueeze(-1)

    data, _ = to_pyg_batch(
        node_feats, edge_feats, edge_index,
        node2type=node2type,
        edge2type=edge2type,
        direction=direction,
        label=torch.tensor([0]),
        hidden_nodes=hidden_mask,
    )
    return data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

#: Config used purely as an architecture template — every field the checks care
#: about is overridden in build_conf() below.  Overridable with --template-conf,
#: which is how the matrix-product variants (mpgmn_*, mpsgmn_*) are checked: their
#: message_fn_type selects a different MSG function, and everything else the
#: ablation touches (aggregator, fan-in, readout) is set here anyway.
TEMPLATE_CONF = PROJECT_ROOT / "configs" / "mnist_cls" / "scalegmn_sizegen_sp.yml"


def build_conf(base_layout, aggregator, direction='forward', symmetry='sign'):
    """Load the template config and override it for a single equivalence check."""
    with open(TEMPLATE_CONF) as f:
        conf = yaml.safe_load(f)

    conf = overwrite_conf(conf, {})
    # Use a smaller hidden dim to keep tests fast
    conf["scalegmn_args"]["d_hid"]      = 32
    conf["scalegmn_args"]["num_layers"] = 2
    conf["scalegmn_args"]["gnn_args"]["aggregator"] = aggregator
    conf["scalegmn_args"]["layer_layout"] = base_layout
    conf["scalegmn_args"]["direction"] = direction
    conf["scalegmn_args"]["symmetry"] = symmetry
    if symmetry == "scale":
        conf["scalegmn_args"]["gnn_args"]["sign_symmetrization"] = False
    return overwrite_conf(conf, {})   # re-propagate d_hid


def build_model(base_layout, aggregator, direction='forward', symmetry='sign'):
    """Build a small ScaleGMN with the given aggregator/direction."""
    conf = build_conf(base_layout, aggregator, direction, symmetry)
    # No-op unless the template config selects a matrix-product MSG function.
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    model.eval()
    return model


def get_node_features(model, data):
    """Run forward pass and return node representations (before readout).

    Temporarily enables equivariant mode on the GNN so that ScaleGMN_GNN_fw
    returns (x, edge_attr) instead of the graph-level pooled output.  This
    lets us compare node representations directly for the equivalence test.
    """
    batch = torch_geometric.data.Batch.from_data_list([data])
    model.gnn.equivariant = True
    with torch.no_grad():
        x, _ = model(batch)
    model.gnn.equivariant = False
    return x  # [num_nodes, d_hid]


def get_graph_output(model, data):
    """Run full forward pass (including readout) and return graph-level output."""
    batch = torch_geometric.data.Batch.from_data_list([data])
    with torch.no_grad():
        out = model(batch)
    return out  # [1, d_out]


def build_model_with_layerwise_readout(base_layout, aggregator, direction='forward', symmetry='sign'):
    """Build ScaleGMN with LayerWiseMeanReadout installed."""
    conf = build_conf(base_layout, aggregator, direction, symmetry)
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    install_layerwise_mean_readout(model, conf["scalegmn_args"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-INR check
# ---------------------------------------------------------------------------

def compare_node_feats(x_base, x_wide, base_layout, multipliers):
    """Max abs error between each base node and its k_l widened copies.

    Node ordering for the widened graph: copies of base node j at positions
    [off + j*k_l, off + j*k_l + k_l) (same block-Kronecker ordering as widen_wb).
    """
    max_err = 0.0
    off_base = 0
    off_wide = 0
    for n_l, k_l in zip(base_layout, multipliers):
        for j in range(n_l):
            base_feat  = x_base[off_base + j]                              # [d_hid]
            wide_feats = x_wide[off_wide + j*k_l: off_wide + (j+1)*k_l]   # [k_l, d_hid]
            err = (wide_feats - base_feat.unsqueeze(0)).abs().max().item()
            max_err = max(max_err, err)
        off_base += n_l
        off_wide += n_l * k_l
    return max_err


def check_file(pth_path, multipliers, fw_mean, fw_sum, bidir_mean, bidir_sum,
               fw_mean_lw, fw_sum_lw, bidir_mean_lw, bidir_sum_lw, tol,
               forward_only=False):
    """Check conditions for one INR file: node-level and graph-level.

    Returns dict with max_abs_err for each condition.  With ``forward_only`` the
    ``bd/*`` entries are omitted (used for the matrix-product ScaleGMN, which has no
    bidirectional variant in any experiment).
    """
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    weights, biases = state_dict_to_wb(sd)
    base_layout = infer_layout(weights)

    wide_weights, wide_biases = widen_wb(weights, biases, multipliers)
    wide_layout = [n * k for n, k in zip(base_layout, multipliers)]

    # Build forward graphs
    g_fw_base_fi   = build_pyg_data(weights,      biases,      base_layout, fanin_rescale=True,  direction='forward')
    g_fw_wide_fi   = build_pyg_data(wide_weights, wide_biases, wide_layout, fanin_rescale=True,  direction='forward')
    g_fw_base_nofi = build_pyg_data(weights,      biases,      base_layout, fanin_rescale=False, direction='forward')
    g_fw_wide_nofi = build_pyg_data(wide_weights, wide_biases, wide_layout, fanin_rescale=False, direction='forward')

    if forward_only:
        x_fw_base_fi_mean = get_node_features(fw_mean, g_fw_base_fi)
        x_fw_wide_fi_mean = get_node_features(fw_mean, g_fw_wide_fi)
        x_fw_base_nofi    = get_node_features(fw_mean, g_fw_base_nofi)
        x_fw_wide_nofi    = get_node_features(fw_mean, g_fw_wide_nofi)
        x_fw_base_fi_sum  = get_node_features(fw_sum,  g_fw_base_fi)
        x_fw_wide_fi_sum  = get_node_features(fw_sum,  g_fw_wide_fi)
        return {
            "fw/fanin+mean":   compare_node_feats(x_fw_base_fi_mean, x_fw_wide_fi_mean, base_layout, multipliers),
            "fw/nofanin+mean": compare_node_feats(x_fw_base_nofi,    x_fw_wide_nofi,    base_layout, multipliers),
            "fw/fanin+sum":    compare_node_feats(x_fw_base_fi_sum,  x_fw_wide_fi_sum,  base_layout, multipliers),
            "fw/graph/fanin+mean+sum-ro":
                (get_graph_output(fw_mean, g_fw_base_fi)
                 - get_graph_output(fw_mean, g_fw_wide_fi)).abs().max().item(),
            "fw/graph/fanin+mean+lw":
                (get_graph_output(fw_mean_lw, g_fw_base_fi)
                 - get_graph_output(fw_mean_lw, g_fw_wide_fi)).abs().max().item(),
            "fw/graph/fanin+sum+lw":
                (get_graph_output(fw_sum_lw, g_fw_base_fi)
                 - get_graph_output(fw_sum_lw, g_fw_wide_fi)).abs().max().item(),
            "fw/graph/nofanin+mean+lw":
                (get_graph_output(fw_mean_lw, g_fw_base_nofi)
                 - get_graph_output(fw_mean_lw, g_fw_wide_nofi)).abs().max().item(),
        }

    # Build bidirectional graphs (adds bw_edge_index = flip(edge_index))
    g_bd_base_fi   = build_pyg_data(weights,      biases,      base_layout, fanin_rescale=True,  direction='bidirectional')
    g_bd_wide_fi   = build_pyg_data(wide_weights, wide_biases, wide_layout, fanin_rescale=True,  direction='bidirectional')
    g_bd_base_nofi = build_pyg_data(weights,      biases,      base_layout, fanin_rescale=False, direction='bidirectional')
    g_bd_wide_nofi = build_pyg_data(wide_weights, wide_biases, wide_layout, fanin_rescale=False, direction='bidirectional')

    # Forward node-level conditions
    x_fw_base_fi_mean  = get_node_features(fw_mean, g_fw_base_fi)
    x_fw_wide_fi_mean  = get_node_features(fw_mean, g_fw_wide_fi)
    x_fw_base_nofi     = get_node_features(fw_mean, g_fw_base_nofi)
    x_fw_wide_nofi     = get_node_features(fw_mean, g_fw_wide_nofi)
    x_fw_base_fi_sum   = get_node_features(fw_sum,  g_fw_base_fi)
    x_fw_wide_fi_sum   = get_node_features(fw_sum,  g_fw_wide_fi)

    # Bidirectional node-level conditions
    x_bd_base_fi_mean  = get_node_features(bidir_mean, g_bd_base_fi)
    x_bd_wide_fi_mean  = get_node_features(bidir_mean, g_bd_wide_fi)
    x_bd_base_nofi     = get_node_features(bidir_mean, g_bd_base_nofi)
    x_bd_wide_nofi     = get_node_features(bidir_mean, g_bd_wide_nofi)
    x_bd_base_fi_sum   = get_node_features(bidir_sum,  g_bd_base_fi)
    x_bd_wide_fi_sum   = get_node_features(bidir_sum,  g_bd_wide_fi)

    # Graph-level output: original sum-pooling readout (fan-in + mean aggregation)
    out_fw_base_sum_ro  = get_graph_output(fw_mean,    g_fw_base_fi)
    out_fw_wide_sum_ro  = get_graph_output(fw_mean,    g_fw_wide_fi)
    out_bd_base_sum_ro  = get_graph_output(bidir_mean, g_bd_base_fi)
    out_bd_wide_sum_ro  = get_graph_output(bidir_mean, g_bd_wide_fi)

    # Graph-level output: layer-wise mean readout
    out_fw_base_lw_mean  = get_graph_output(fw_mean_lw,    g_fw_base_fi)
    out_fw_wide_lw_mean  = get_graph_output(fw_mean_lw,    g_fw_wide_fi)
    out_fw_base_lw_sum   = get_graph_output(fw_sum_lw,     g_fw_base_fi)
    out_fw_wide_lw_sum   = get_graph_output(fw_sum_lw,     g_fw_wide_fi)
    out_fw_base_lw_nofi  = get_graph_output(fw_mean_lw,    g_fw_base_nofi)
    out_fw_wide_lw_nofi  = get_graph_output(fw_mean_lw,    g_fw_wide_nofi)
    out_bd_base_lw_mean  = get_graph_output(bidir_mean_lw, g_bd_base_fi)
    out_bd_wide_lw_mean  = get_graph_output(bidir_mean_lw, g_bd_wide_fi)
    out_bd_base_lw_sum   = get_graph_output(bidir_sum_lw,  g_bd_base_fi)
    out_bd_wide_lw_sum   = get_graph_output(bidir_sum_lw,  g_bd_wide_fi)
    out_bd_base_lw_nofi  = get_graph_output(bidir_mean_lw, g_bd_base_nofi)
    out_bd_wide_lw_nofi  = get_graph_output(bidir_mean_lw, g_bd_wide_nofi)

    return {
        # Node-level
        "fw/fanin+mean":    compare_node_feats(x_fw_base_fi_mean, x_fw_wide_fi_mean, base_layout, multipliers),
        "fw/nofanin+mean":  compare_node_feats(x_fw_base_nofi,    x_fw_wide_nofi,    base_layout, multipliers),
        "fw/fanin+sum":     compare_node_feats(x_fw_base_fi_sum,  x_fw_wide_fi_sum,  base_layout, multipliers),
        "bd/fanin+mean":    compare_node_feats(x_bd_base_fi_mean, x_bd_wide_fi_mean, base_layout, multipliers),
        "bd/nofanin+mean":  compare_node_feats(x_bd_base_nofi,    x_bd_wide_nofi,    base_layout, multipliers),
        "bd/fanin+sum":     compare_node_feats(x_bd_base_fi_sum,  x_bd_wide_fi_sum,  base_layout, multipliers),
        # Graph-level: fan-in + mean + sum-readout (should FAIL — sum scales with k)
        "fw/graph/fanin+mean+sum-ro":  (out_fw_base_sum_ro - out_fw_wide_sum_ro).abs().max().item(),
        "bd/graph/fanin+mean+sum-ro":  (out_bd_base_sum_ro - out_bd_wide_sum_ro).abs().max().item(),
        # Graph-level: layer-wise mean readout
        "fw/graph/fanin+mean+lw":      (out_fw_base_lw_mean - out_fw_wide_lw_mean).abs().max().item(),
        "fw/graph/nofanin+mean+lw":    (out_fw_base_lw_nofi - out_fw_wide_lw_nofi).abs().max().item(),
        "fw/graph/fanin+sum+lw":       (out_fw_base_lw_sum  - out_fw_wide_lw_sum).abs().max().item(),
        "bd/graph/fanin+mean+lw":      (out_bd_base_lw_mean - out_bd_wide_lw_mean).abs().max().item(),
        "bd/graph/nofanin+mean+lw":    (out_bd_base_lw_nofi - out_bd_wide_lw_nofi).abs().max().item(),
        "bd/graph/fanin+sum+lw":       (out_bd_base_lw_sum  - out_bd_wide_lw_sum).abs().max().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _copy_gnn_backbone(src_model, dst_model):
    """Copy all parameters except readout from src to dst."""
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    for key in dst_sd:
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst_model.load_state_dict(dst_sd)
    dst_model.eval()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))) \
                  / "mnist_inrs" / "w32_canon"
    p.add_argument("--data-dir", type=Path, default=default_dir)
    p.add_argument("--n-inrs",   type=int,   default=5,    help="Number of INR files to test")
    p.add_argument("--k-max",    type=int,   default=4,    help="Max width multiplier")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--tol",      type=float, default=1e-5, help="Pass threshold for max |error|")
    p.add_argument("--symmetry", type=str,   default="sign", choices=["sign", "scale", "permutation"],
                   help="Symmetry type for ScaleGMN (sign=SIREN, scale=ReLU)")
    p.add_argument("--template-conf", type=Path, default=TEMPLATE_CONF,
                   help="Architecture template config. Pass a matrix-product config "
                        "(mpgmn_*/mpsgmn_*) to check that MSG function instead.")
    p.add_argument("--forward-only", action="store_true",
                   help="Skip the bidirectional conditions. Use for mpsgmn_* (the "
                        "matrix-product ScaleGMN is forward-only in every experiment). "
                        "The forward numbers are unchanged by this flag.")
    return p.parse_args()


def main():
    global TEMPLATE_CONF
    args = parse_args()
    TEMPLATE_CONF = args.template_conf if args.template_conf.is_absolute() \
        else (PROJECT_ROOT / args.template_conf)
    torch.manual_seed(args.seed)

    files = sorted(args.data_dir.rglob("*.pth"))[: args.n_inrs]
    if not files:
        print(f"[ERROR] No .pth files found under {args.data_dir}")
        sys.exit(1)

    base_layout = [2, 32, 32, 1]

    sym = args.symmetry

    # Node-level models (original readout, only used in equivariant/node mode).
    # The bidirectional ones are built even under --forward-only, where they go unused:
    # they consume RNG before the layer-wise-readout models below, so building them
    # unconditionally keeps the reported forward numbers identical in both modes.
    fw_mean    = build_model(base_layout, aggregator="mean", direction='forward', symmetry=sym)
    fw_sum     = build_model(base_layout, aggregator="add",  direction='forward', symmetry=sym)
    bidir_mean = build_model(base_layout, aggregator="mean", direction='bidirectional', symmetry=sym)
    bidir_sum  = build_model(base_layout, aggregator="add",  direction='bidirectional', symmetry=sym)

    # Share weights within each direction pair so aggregator is the only variable
    fw_sum.load_state_dict(fw_mean.state_dict())
    fw_sum.eval()
    bidir_sum.load_state_dict(bidir_mean.state_dict())
    bidir_sum.eval()

    # Graph-level models (with layer-wise mean readout)
    fw_mean_lw    = build_model_with_layerwise_readout(base_layout, aggregator="mean", direction='forward', symmetry=sym)
    fw_sum_lw     = build_model_with_layerwise_readout(base_layout, aggregator="add",  direction='forward', symmetry=sym)
    bidir_mean_lw = build_model_with_layerwise_readout(base_layout, aggregator="mean", direction='bidirectional', symmetry=sym)
    bidir_sum_lw  = build_model_with_layerwise_readout(base_layout, aggregator="add",  direction='bidirectional', symmetry=sym)

    # Share GNN backbone weights (copy from node-level models) so only readout differs
    # The layerwise-readout models have a wrapped forward, so we load into the underlying gnn
    # We need to be careful: the readout params are different, so we only copy non-readout params
    _copy_gnn_backbone(fw_mean, fw_mean_lw)
    _copy_gnn_backbone(fw_mean, fw_sum_lw)
    _copy_gnn_backbone(bidir_mean, bidir_mean_lw)
    _copy_gnn_backbone(bidir_mean, bidir_sum_lw)

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    print(f"Testing {len(files)} INR files from {args.data_dir}")
    print(f"Template config: {TEMPLATE_CONF.name}  "
          f"(message_fn_type={build_conf(base_layout, 'mean', 'forward', sym)['scalegmn_args']['gnn_args'].get('message_fn_type')})")
    print(f"Base layout: {base_layout}  k_max={args.k_max}  tol={args.tol}  symmetry={sym}"
          f"{'  direction=forward only' if args.forward_only else ''}")
    print()

    # ---- Part 1: Node-level checks ----
    print("="*80)
    print("Part 1: Node-level representations (before readout)")
    print("="*80)

    col_w = 38
    header = (f"{'File':<{col_w}}  "
              f"{'fw/fi+mean':>12}  {'fw/nofi+mean':>13}  {'fw/fi+sum':>11}  ")
    if not args.forward_only:
        header += f"{'bd/fi+mean':>12}  {'bd/nofi+mean':>13}  {'bd/fi+sum':>11}  "
    header += "k"
    print(header)
    print("-" * len(header))

    node_conditions = [
        ("fw/fanin+mean",   "forward    | fan-in + mean",    True),
        ("fw/nofanin+mean", "forward    | no fan-in + mean", False),
        ("fw/fanin+sum",    "forward    | fan-in + sum",     False),
        ("bd/fanin+mean",   "bidirect.  | fan-in + mean",    True),
        ("bd/nofanin+mean", "bidirect.  | no fan-in + mean", False),
        ("bd/fanin+sum",    "bidirect.  | fan-in + sum",     False),
    ]
    # Per direction: the full condition, then each of its three components switched
    # off in turn (fan-in rescaling, mean aggregation, layer-wise mean readout).
    graph_conditions = [
        ("fw/graph/fanin+mean+lw",     "forward    | fan-in + mean + lw-readout",    True),
        ("fw/graph/nofanin+mean+lw",   "forward    | no fan-in + mean + lw-readout", False),
        ("fw/graph/fanin+sum+lw",      "forward    | fan-in + sum  + lw-readout",    False),
        ("fw/graph/fanin+mean+sum-ro", "forward    | fan-in + mean + sum-readout",   False),
        ("bd/graph/fanin+mean+lw",     "bidirect.  | fan-in + mean + lw-readout",    True),
        ("bd/graph/nofanin+mean+lw",   "bidirect.  | no fan-in + mean + lw-readout", False),
        ("bd/graph/fanin+sum+lw",      "bidirect.  | fan-in + sum  + lw-readout",    False),
        ("bd/graph/fanin+mean+sum-ro", "bidirect.  | fan-in + mean + sum-readout",   False),
    ]
    if args.forward_only:
        keep = lambda cs: [c for c in cs if not c[0].startswith("bd/")]  # noqa: E731
        node_conditions, graph_conditions = keep(node_conditions), keep(graph_conditions)

    all_errs = {k: [] for k, _, _ in node_conditions + graph_conditions}

    for pth in files:
        k1 = int(torch.randint(2, args.k_max + 1, (), generator=rng).item())
        k2 = int(torch.randint(2, args.k_max + 1, (), generator=rng).item())
        mults = [1, k1, k2, 1]

        errs = check_file(pth, mults, fw_mean, fw_sum, bidir_mean, bidir_sum,
                          fw_mean_lw, fw_sum_lw, bidir_mean_lw, bidir_sum_lw,
                          tol=args.tol, forward_only=args.forward_only)
        for k, v in errs.items():
            all_errs[k].append(v)

        name = str(pth.relative_to(args.data_dir))
        row = (f"{name:<{col_w}}  "
               f"{errs['fw/fanin+mean']:>12.2e}  "
               f"{errs['fw/nofanin+mean']:>13.2e}  "
               f"{errs['fw/fanin+sum']:>11.2e}  ")
        if not args.forward_only:
            row += (f"{errs['bd/fanin+mean']:>12.2e}  "
                    f"{errs['bd/nofanin+mean']:>13.2e}  "
                    f"{errs['bd/fanin+sum']:>11.2e}  ")
        print(row + f"[{k1},{k2}]")

    print("-" * len(header))
    print("\nSummary — max |error| over all files:")
    print()
    print("  Node-level (before readout):")

    all_correct = True
    for key, label, expect_pass in node_conditions:
        max_err = max(all_errs[key])
        passed  = max_err < args.tol
        correct = passed == expect_pass
        if not correct:
            all_correct = False
        marker  = "✓" if correct else "✗"
        tag     = "PASS" if passed else "FAIL"
        expect  = "should PASS" if expect_pass else "should FAIL"
        print(f"    {marker} [{tag}]  {label}  ({expect})  max_err={max_err:.2e}")

    # ---- Part 2: Graph-level checks ----
    print()
    print("  Graph-level output:")

    for key, label, expect_pass in graph_conditions:
        max_err = max(all_errs[key])
        passed  = max_err < args.tol
        correct = passed == expect_pass
        if not correct:
            all_correct = False
        marker  = "✓" if correct else "✗"
        tag     = "PASS" if passed else "FAIL"
        expect  = "should PASS" if expect_pass else "should FAIL"
        print(f"    {marker} [{tag}]  {label}  ({expect})  max_err={max_err:.2e}")

    print()
    if all_correct:
        print("[OK] All conditions behaved as expected.")
    else:
        print("[FAIL] Some conditions did not behave as expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()

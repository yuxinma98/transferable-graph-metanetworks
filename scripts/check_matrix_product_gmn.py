"""
Checks for the matrix-product GMN (`gnn_args.message_fn_type: matrix_product`).

The matrix-product GMN is the duplication-compatible plain GMN with the MSG
function constrained to a multiplication by the raw scalar weight:

    msg_{v_j^(l-1) -> v_i^(l)} = Wbar^(l)_ij * (h_{v_j^(l-1)} A_in),

so that, with mean aggregation, the forward aggregate is a matrix product

    Z^(l),in = (1 / n_{l-1}) Wbar^(l) H^(l-1) A_in = W^(l) H^(l-1) A_in,

and the backward aggregate (bidirectional variant) uses (Wbar^(l+1))^T.

Three tests:

Test 1 — matrix-product identity
    The aggregate produced by PyG message passing equals the closed-form
    (1 / n_{l-1}) Wbar^(l) H^(l-1) A_in (and 0 for the input layer, which has no
    incoming weight matrix).

Test 2 — equivalence classes, forward vs bidirectional
    Graph-level output is compared between a base INR and five families of
    functionally equivalent widenings (all with k_0 = k_L = 1). All satisfy the
    forward (row-sum) condition

        (row)  W^(l)up_{ij} 1_{k_{l-1}} = W^(l)_{ij} 1_{k_l}
                                            eq:widened-general-duplication-equivalence

    and they differ in whether they also satisfy the backward (column-sum)
    condition required by the bidirectional variant (paper, "For layer r
    contributing backward messages to layer r-1"):

        (col)  1_{k_l}^T W^(l)up_{ij} = (k_l / k_{l-1}) W^(l)_{ij} 1^T

      uniform        W^(l)up = W^(l) (x) (1/k_{l-1}) 1 1^T   row+col  eq:widened-duplication
      row-stoch      W^(l)up = W^(l) (x) P, P 1 = 1          row      eq:widened-row-stochastic
      doubly-stoch   as above, also P^T 1 = (k_l/k_{l-1}) 1  row+col
      general-bidir  blockwise, non-Kronecker                row+col
      general        blockwise, non-Kronecker                row

    The FORWARD variant preserves all five (the row condition is exactly what a
    matrix product by a duplicated vector sees). The BIDIRECTIONAL variant
    preserves exactly the three that also satisfy (col) -- including the
    non-Kronecker `general-bidir` family, so what it respects is the full
    row+col class, not merely the doubly-stochastic Kronecker special case.

    Each family is checked to be a genuine functional equivalence of the
    underlying MLP, to actually satisfy the conditions it claims, and (except for
    `uniform`) to be measurably far from the uniform widening, so that no
    equivalence test passes vacuously.

Test 3 — spectral continuity
    Using the Hadamard family of Section "Matrix-product GMN" of the paper:
    Wbar^(2) = H_n, all other weights/biases zero, so the normalized spectral
    norm ||theta_n|| = 1/sqrt(n) -> 0. A model that is Lipschitz w.r.t. the
    normalized spectral norm must satisfy f(theta_n) -> f(0). The matrix-product
    GMN does; the plain (edgewise-MLP) GMN need not, and empirically does not.

Both matrix-product families are checked by the same three tests, selected with
`--family`: the plain GMN (`gmn`, `symmetry: permutation`, MSG constraint on top of the
concat-MLP model) and the matrix-product ScaleGMN (`scalegmn`, `symmetry: scale`, the same
MSG constraint on top of ScaleGMN's node states and EquivariantNet update,
`message_fn_type: matrix_product_scale`). The contrast model in Tests 2 and 3 is always
the *edgewise-MLP* model of the same family.

Usage
-----
    python scripts/check_matrix_product_gmn.py
    python scripts/check_matrix_product_gmn.py --family scalegmn
    python scripts/check_matrix_product_gmn.py --n-inrs 10 --k-max 4
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

from src.scalegmn.models import ScaleGMN                                  # noqa: E402
from src.utils.helpers import overwrite_conf                              # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from models.layerwise_mean_readout import install_layerwise_mean_readout   # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context       # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from check_duplication_equiv import (                                     # noqa: E402
    build_pyg_data, get_graph_output, infer_layout, state_dict_to_wb, widen_wb,
)

CONFIGS = PROJECT_ROOT / "configs" / "mnist_cls"

#: --family -> (matrix-product config, edgewise-MLP contrast config, contrast label).
#: `gmn` is the plain GMN of paper sec:matrix-product-gmn (symmetry: permutation);
#: `scalegmn` is the matrix-product ScaleGMN (symmetry: scale) -- same MSG constraint,
#: ScaleGMN node states and EquivariantNet update.
FAMILY_CONFS = {
    "gmn": (CONFIGS / "mpgmn_sizegen_sp.yml",
            CONFIGS / "gmn_sizegen_sp.yml", "plain GMN"),
    "scalegmn": (CONFIGS / "mpsgmn_sizegen_sp_v3.yml",
                 CONFIGS / "scalegmn_sizegen_sp_v3.yml", "plain ScaleGMN"),
}

#: Families with no bidirectional model. There is no bidirectional matrix-product
#: ScaleGMN: the `mpsgmn_*` configs are forward-only in every experiment, so Test 2 has
#: no `mp-bidirectional` column for them and the column condition is exercised on the
#: `gmn` family, which is where bidirectional is actually run.
FORWARD_ONLY_FAMILIES = {"scalegmn"}

D_HID = 32
NUM_LAYERS = 2


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_conf(conf_path, layer_layout, direction):
    """Load a config and shrink it to a fast, self-consistent test architecture."""
    with open(conf_path) as f:
        conf = yaml.safe_load(f)
    a = conf["scalegmn_args"]
    a["layer_layout"] = layer_layout
    a["direction"] = direction
    a["num_layers"] = NUM_LAYERS
    a["d_hid"] = D_HID
    # duplication-equivariant condition (as set by train_sizegen.py --duplication-equiv)
    a["gnn_args"]["aggregator"] = "mean"
    a["gnn_args"]["msg_equiv_on_hidden"] = True
    return overwrite_conf(conf, {})   # propagates d_hid to d_node/d_edge/d_rho/d_k


def build_model(conf_path, layer_layout, direction, layerwise_readout=True):
    conf = build_conf(conf_path, layer_layout, direction)
    # no-op unless the config selects a matrix-product ScaleGMN MSG (the plain-GMN
    # matrix-product layer lives in the subtree and needs no swap)
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    if layerwise_readout:
        install_layerwise_mean_readout(model, conf["scalegmn_args"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 2: families of functionally equivalent widenings
# ---------------------------------------------------------------------------

FAMILIES = ("uniform", "row-stoch", "doubly-stoch", "general-bidir", "general")


def _row_stochastic(k_out, k_in):
    """Random P in R^{k_out x k_in} with P 1 = 1 (row sums one)."""
    p = torch.randn(k_out, k_in)
    return p + (1.0 - p.sum(-1, keepdim=True)) / k_in


def _equal_row_col_sums(shape, c):
    """Random real matrices with all row sums AND all column sums equal to c.

    `shape` is (..., k, k); `c` broadcasts against the leading dims. Obtained by
    projecting a random matrix onto that affine subspace (J = 1 1^T):
        B = R - (1/k) R J - (1/k) J R + (1/k^2) J R J + (c/k) J.
    Entries may be negative; c = 0 is allowed (so it also covers zero base weights,
    which a "W_ij times a doubly stochastic matrix" parameterization would miss).
    """
    k = shape[-1]
    assert shape[-2] == k, "equal row/column sums require a square block"
    r = torch.randn(*shape)
    j = torch.ones(k, k)
    return r - r @ j / k - j @ r / k + j @ r @ j / k ** 2 + c[..., None, None] * j / k


def _doubly_stochastic(k):
    """Random P in R^{k x k} with P 1 = 1 and P^T 1 = 1 (entries may be negative)."""
    return _equal_row_col_sums((k, k), torch.ones(()))


def _layer_P(family, k_out, k_in):
    """Kronecker factor P^(l) for one layer, or None for the blockwise family.

    For the boundary layers (k_in == 1 or k_out == 1) the constraints force P, so
    every family agrees there: P = 1_{k_out} when k_in == 1, and P = 1^T / k_in
    when k_out == 1 (the latter only for row/doubly-stochastic + uniform).
    """
    if family == "uniform" or k_in == 1 or k_out == 1:
        if family == "row-stoch" and k_out == 1:
            return _row_stochastic(k_out, k_in)      # row sum 1, but not uniform
        return torch.ones(k_out, k_in) / k_in
    if family == "row-stoch":
        return _row_stochastic(k_out, k_in)
    if family == "doubly-stoch":
        assert k_out == k_in, "doubly-stochastic widening needs k_l == k_{l-1}"
        return _doubly_stochastic(k_in)
    raise ValueError(family)


def widen_family(weights, biases, multipliers, family):
    """Widen an MLP with one of the four equivalence families.

    Weights are in ScaleGMN's [n_in, n_out, 1] convention. Copies of base neuron j
    occupy contiguous indices [j*k, (j+1)*k), matching `widen_wb` and
    `compare_node_feats` in check_duplication_equiv.py.
    """
    assert multipliers[0] == 1 and multipliers[-1] == 1
    wide_weights, wide_biases = [], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        n_in, n_out, _ = W.shape
        k_in, k_out = multipliers[l], multipliers[l + 1]
        W = W.squeeze(-1)                                        # [n_in, n_out]

        if family == "general":
            # blocks[j, i] in R^{k_out x k_in}, free apart from all row sums == W_ij
            blocks = torch.randn(n_in, n_out, k_out, k_in)
            blocks = blocks + (W[:, :, None, None] - blocks.sum(-1, keepdim=True)) / k_in
        elif family == "general-bidir":
            # blocks[j, i] free apart from all row sums == W_ij (forward condition,
            # eq:widened-general-duplication-equivalence) AND all column sums
            # == (k_out / k_in) W_ij (backward condition, paper eq. after
            # "For layer r contributing backward messages to layer r-1"). Not of
            # Kronecker form: each (i, j) block is an independent random matrix.
            if k_in == k_out and k_in > 1:
                blocks = _equal_row_col_sums((n_in, n_out, k_out, k_in), W)
            else:
                # Boundary layers (k_in == 1 or k_out == 1): the two conditions
                # together force the uniform block, W_ij 1 / k_in. (For unequal
                # k_in != k_out the uniform block is still a valid member of the
                # family, just not the most general one -- not hit by main().)
                blocks = W[:, :, None, None] * torch.ones(k_out, k_in) / k_in
        else:
            p = _layer_P(family, k_out, k_in)
            blocks = W[:, :, None, None] * p                     # W (x) P

        # [j, a, i, b] -> [N_in, N_out]
        W_up = blocks.permute(0, 3, 1, 2).reshape(n_in * k_in, n_out * k_out)
        wide_weights.append(W_up.unsqueeze(-1))
        wide_biases.append(torch.kron(b.squeeze(-1), torch.ones(k_out)).unsqueeze(-1))

    return tuple(wide_weights), tuple(wide_biases)


def check_family_conditions(weights, wide_weights, multipliers):
    """Which of the two blockwise conditions the widened weights actually satisfy.

    Returns (row_err, col_err, nonuniformity) as max abs deviations over all layers:
      row_err  -- forward condition  W^(l)up_{ij} 1_{k_in}  == W^(l)_{ij} 1_{k_out}
      col_err  -- backward condition 1_{k_out}^T W^(l)up_{ij} == (k_out/k_in) W^(l)_{ij} 1^T
      nonuniformity -- distance from the uniform widening, so a family cannot pass
                       the equivalence test vacuously by collapsing to uniform.
    """
    row_err = col_err = nonuniformity = 0.0
    for l, W in enumerate(weights):
        n_in, n_out, _ = W.shape
        k_in, k_out = multipliers[l], multipliers[l + 1]
        W = W.squeeze(-1)
        # invert the [j, a, i, b] flattening of widen_family -> blocks [j, i, b, a]
        blocks = wide_weights[l].squeeze(-1) \
            .reshape(n_in, k_in, n_out, k_out).permute(0, 2, 3, 1)
        row_err = max(row_err, (blocks.sum(-1) - W[:, :, None]).abs().max().item())
        col_err = max(col_err,
                      (blocks.sum(-2) - (k_out / k_in) * W[:, :, None]).abs().max().item())
        uniform = W[:, :, None, None] * torch.ones(k_out, k_in) / k_in
        nonuniformity = max(nonuniformity, (blocks - uniform).abs().max().item())
    return row_err, col_err, nonuniformity


def mlp_forward(weights, biases, x):
    """Evaluate the INR MLP (ReLU hidden layers, linear output) on x: [B, n_0]."""
    h = x
    for l, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W.squeeze(-1) + b.squeeze(-1)
        if l < len(weights) - 1:
            h = torch.relu(h)
    return h


# ---------------------------------------------------------------------------
# Test 1: matrix-product identity
# ---------------------------------------------------------------------------

def check_matrix_product_identity(model, data, layer_layout):
    """Compare the PyG aggregate of the first GNN layer with the closed form."""
    batch = torch_geometric.data.Batch.from_data_list([data])
    with torch.no_grad():
        batch = model.construct_graph(batch)
        pos_embed = None
        if model.node_pos_embed:
            batch, pos_embed = model.positional_embeddings(batch)

        layer = model.gnn.fw_layers[0]
        aggr = layer.propagate(edge_index=batch.edge_index, x=batch.x,
                               edge_attr=batch.edge_attr, mask_hidden=None,
                               mask_first_layer=None, pos_embed=pos_embed,
                               sign_mask=None)

        # Closed form: Z^(l) = (1 / n_{l-1}) Wbar^(l) H^(l-1) A_in
        num_nodes = sum(layer_layout)
        dense_w = torch.zeros(num_nodes, num_nodes)
        dense_w[batch.edge_index[0], batch.edge_index[1]] = batch.edge_attr.squeeze(-1)
        hA = layer.message_fn(batch.x)                      # [N, d_hid]

        expected = torch.zeros_like(aggr)
        offsets = torch.cumsum(torch.tensor([0] + list(layer_layout)), dim=0)
        for l in range(1, len(layer_layout)):
            rows = slice(offsets[l], offsets[l + 1])        # destination nodes, layer l
            cols = slice(offsets[l - 1], offsets[l])        # source nodes, layer l-1
            w_bar = dense_w[cols, rows].t()                 # [n_l, n_{l-1}]
            expected[rows] = (w_bar @ hA[cols]) / layer_layout[l - 1]

    err = (aggr - expected).abs().max().item()
    in_layer_err = aggr[: layer_layout[0]].abs().max().item()
    return err, in_layer_err


# ---------------------------------------------------------------------------
# Test 3: spectral continuity (Hadamard family)
# ---------------------------------------------------------------------------

def hadamard(n):
    """Sylvester Hadamard matrix, n a power of two."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    assert h.shape[0] == n, f"{n} is not a power of two"
    return h


def hadamard_wb(n, d_in=2, d_out=1):
    """theta_n: Wbar^(2) = H_n (i.e. W^(2) = H_n / n), everything else zero.

    Weights in ScaleGMN's [n_in, n_out, 1] convention; the graph builder applies
    the fan-in rescaling Wbar^(l) = n_{l-1} W^(l).
    """
    weights = (
        torch.zeros(d_in, n, 1),
        (hadamard(n) / n).unsqueeze(-1),
        torch.zeros(n, d_out, 1),
    )
    biases = (torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(d_out, 1))
    return weights, biases


def zero_wb(n, d_in=2, d_out=1):
    weights = (torch.zeros(d_in, n, 1), torch.zeros(n, n, 1), torch.zeros(n, d_out, 1))
    biases = (torch.zeros(n, 1), torch.zeros(n, 1), torch.zeros(d_out, 1))
    return weights, biases


def context_wb(n, d_in=2, d_out=1, hadamard_layer_2=True):
    """The Hadamard perturbation on top of a fixed, width-independent context.

    Needed because the all-zero context of :func:`hadamard_wb` is *annihilated* by a
    scale-equivariant model: ScaleGMN's message and update are degree-one homogeneous,
    so with zero hidden node features (all biases zero) and zero W^(1) every hidden
    state stays exactly zero and f(theta_n) == f(0) == 0 for every n, for the
    matrix-product and the edgewise-MLP ScaleGMN alike -- a vacuous pass.

    Here only layer 2 is perturbed, layers 1 and 3 and all biases hold fixed values
    whose fan-in-rescaled edge features Wbar^(l) are O(1) at every width, so the
    normalized-spectral-norm distance between the two graphs is still exactly that of
    layer 2 alone:

        sqrt(n_1/n_2) ||W^(2)||_2 = ||H_n||_2 / n = 1 / sqrt(n)  ->  0

    and a model Lipschitz w.r.t. the normalized spectral norm must have
    |f(theta_n^ctx) - f(theta_n^ctx, W^(2) = 0)| -> 0.
    """
    w1 = torch.empty(d_in, n)
    w1[0] = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)   # two node classes
    w1[1] = 0.5
    w2 = hadamard(n) / n if hadamard_layer_2 else torch.zeros(n, n)
    w3 = torch.ones(n, d_out) / n                              # Wbar^(3) = 1: O(1)
    weights = (w1.unsqueeze(-1), w2.unsqueeze(-1), w3.unsqueeze(-1))
    biases = (0.5 * torch.ones(n, 1), -0.25 * torch.ones(n, 1), 0.1 * torch.ones(d_out, 1))
    return weights, biases


#: Perturbed / reference weight pairs for Test 3. `hadamard` is the family of the paper
#: (zero context); `hadamard+context` adds the fixed nonzero context above.
CONTINUITY_FAMILIES = {
    "hadamard": (hadamard_wb, zero_wb),
    "hadamard+context": (context_wb,
                         lambda n, **kw: context_wb(n, hadamard_layer_2=False, **kw)),
}


def check_spectral_continuity(conf_path, widths, direction, continuity_family="hadamard"):
    """|f(theta_n) - f(theta_n, W^(2) = 0)| as n grows, for one fixed model per width.

    The model's learned parameters are width-independent, so one model built at
    the smallest width can be reused: only its layer_layout-derived readout sizes
    matter, and those depend on the number of layers, not the widths.
    """
    perturbed, reference = CONTINUITY_FAMILIES[continuity_family]
    base_layout = [2, widths[0], widths[0], 1]
    model = build_model(conf_path, base_layout, direction)
    diffs = {}
    for n in widths:
        layout = [2, n, n, 1]
        g_had = build_pyg_data(*perturbed(n), layout, fanin_rescale=True, direction=direction)
        g_ref = build_pyg_data(*reference(n), layout, fanin_rescale=True, direction=direction)
        out_had = get_graph_output(model, g_had)
        out_ref = get_graph_output(model, g_ref)
        diffs[n] = (out_had - out_ref).abs().max().item()
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))) \
                  / "mnist_inrs" / "w16_sp"
    p.add_argument("--family", choices=sorted(FAMILY_CONFS), default="gmn",
                   help="Which matrix-product family to check: the plain GMN "
                        "(symmetry: permutation) or the matrix-product ScaleGMN "
                        "(symmetry: scale)")
    p.add_argument("--data-dir", type=Path, default=default_dir)
    p.add_argument("--n-inrs", type=int, default=5, help="Number of INR files to test")
    p.add_argument("--k-max", type=int, default=4, help="Max width multiplier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5, help="Pass threshold for max |error|")
    p.add_argument("--include-bidir-scalegmn", action="store_true",
                   help="Also build and score a bidirectional matrix-product ScaleGMN and a "
                        "bidirectional plain ScaleGMN in Test 2, via "
                        "MatrixProductScale_GNN_layer_aggr (models/matrix_product_scale_layer.py), "
                        "which exists specifically so this contrast can be completed even though no "
                        "training experiment uses a bidirectional mpsgmn. No-op for --family gmn, "
                        "which already builds both directions.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    mp_conf, base_conf, base_label = FAMILY_CONFS[args.family]
    print(f"Family: {args.family}   matrix-product: {mp_conf.name}   "
          f"contrast ({base_label}): {base_conf.name}")

    files = sorted(args.data_dir.rglob("*.pth"))[: args.n_inrs]
    if not files:
        print(f"[ERROR] No .pth files found under {args.data_dir}")
        sys.exit(1)

    sd = torch.load(files[0], map_location="cpu", weights_only=True)
    base_layout = infer_layout(state_dict_to_wb(sd)[0])
    print(f"Base layout: {base_layout}   ({len(files)} INRs from {args.data_dir})")

    fw = build_model(mp_conf, base_layout, "forward")
    forward_only = args.family in FORWARD_ONLY_FAMILIES and not args.include_bidir_scalegmn
    bd = None if forward_only else build_model(mp_conf, base_layout, "bidirectional")

    # -- Test 1 -------------------------------------------------------------
    print("\n=== Test 1: matrix-product identity (forward aggregate) ===")
    max_err = max_in_err = 0.0
    for f in files:
        weights, biases = state_dict_to_wb(torch.load(f, map_location="cpu", weights_only=True))
        layout = infer_layout(weights)
        data = build_pyg_data(weights, biases, layout, fanin_rescale=True, direction="forward")
        err, in_err = check_matrix_product_identity(fw, data, layout)
        max_err = max(max_err, err)
        max_in_err = max(max_in_err, in_err)
    t1_ok = max_err < args.tol and max_in_err < args.tol
    print(f"  aggregate vs (1/n_l-1) Wbar H A : max |err| = {max_err:.3e}  "
          f"{'PASS' if max_err < args.tol else 'FAIL'}")
    print(f"  input-layer aggregate == 0      : max |err| = {max_in_err:.3e}  "
          f"{'PASS' if max_in_err < args.tol else 'FAIL'}")

    # -- Test 2 -------------------------------------------------------------
    print("\n=== Test 2: equivalence classes, forward vs bidirectional ===")
    k_hidden = [1] + [args.k_max] * (len(base_layout) - 2) + [1]
    print(f"  width multipliers: {k_hidden}")
    # the edgewise-MLP model of the same family, dup-equiv, for contrast
    base_fw = build_model(base_conf, base_layout, "forward")
    base_bd = None if forward_only else build_model(base_conf, base_layout, "bidirectional")
    variants = (("mp-forward", fw), ("base-forward", base_fw)) if forward_only else \
               (("mp-forward", fw), ("mp-bidirectional", bd),
                ("base-forward", base_fw), ("base-bidirectional", base_bd))
    labels = {"mp-forward": "mp-forward", "mp-bidirectional": "mp-bidirectional",
              "base-forward": base_label, "base-bidirectional": f"{base_label} (bd)"}
    errs = {(fam, d): 0.0 for fam in FAMILIES for d, _ in variants}
    mlp_errs = {fam: 0.0 for fam in FAMILIES}
    cond = {fam: [0.0, 0.0, float("inf")] for fam in FAMILIES}   # row_err, col_err, min nonunif
    coords = torch.rand(64, base_layout[0]) * 2 - 1
    for f in files:
        weights, biases = state_dict_to_wb(torch.load(f, map_location="cpu", weights_only=True))
        layout = infer_layout(weights)
        wide_layout = [n * k for n, k in zip(layout, k_hidden)]
        y_base = mlp_forward(weights, biases, coords)
        for fam in FAMILIES:
            wide_weights, wide_biases = widen_family(weights, biases, k_hidden, fam)
            if fam == "uniform":   # self-check: must agree with check_duplication_equiv
                ref_w, _ = widen_wb(weights, biases, k_hidden)
                assert all(torch.allclose(a, b) for a, b in zip(wide_weights, ref_w)), \
                    "uniform widening disagrees with widen_wb (node ordering mismatch)"
            # the widened MLP must compute the same function
            y_wide = mlp_forward(wide_weights, wide_biases, coords)
            mlp_errs[fam] = max(mlp_errs[fam], (y_base - y_wide).abs().max().item())
            # ... and the widening must really be the family it claims to be
            r_err, c_err, nonunif = check_family_conditions(weights, wide_weights, k_hidden)
            cond[fam][0] = max(cond[fam][0], r_err)
            cond[fam][1] = max(cond[fam][1], c_err)
            cond[fam][2] = min(cond[fam][2], nonunif)
            for name, model in variants:
                direction = "forward" if name.endswith("forward") else "bidirectional"
                g_base = build_pyg_data(weights, biases, layout, True, direction)
                g_wide = build_pyg_data(wide_weights, wide_biases, wide_layout, True, direction)
                d = (get_graph_output(model, g_base)
                     - get_graph_output(model, g_wide)).abs().max().item()
                errs[(fam, name)] = max(errs[(fam, name)], d)

    # Which blockwise conditions each family is built to satisfy: the forward
    # (row-sum) one, and the backward (column-sum) one of paper eq. 1576-1579.
    expect_col_cond = {"uniform": True, "row-stoch": False, "doubly-stoch": True,
                       "general-bidir": True, "general": False}
    print(f"  {'family':14s} {'MLP equiv':>10s} {'row cond':>10s} {'col cond':>10s} {'nonunif':>9s}")
    t2_ok = True
    for fam in FAMILIES:
        r_err, c_err, nonunif = cond[fam]
        t2_ok = t2_ok and mlp_errs[fam] < 1e-4          # genuine functional equivalence
        t2_ok = t2_ok and r_err < args.tol              # every family satisfies the row condition
        t2_ok = t2_ok and ((c_err < args.tol) == expect_col_cond[fam])
        t2_ok = t2_ok and (fam == "uniform" or nonunif > 1e-2)   # not vacuously uniform
        print(f"  {fam:14s} {mlp_errs[fam]:10.2e} {r_err:10.2e} {c_err:10.2e} {nonunif:9.2e}")

    # matrix-product forward preserves everything satisfying the row condition;
    # matrix-product bidirectional needs the column condition too; the edgewise-MLP
    # plain GMN only uniform duplication.
    expect_pass = {}
    for fam in FAMILIES:
        expect_pass[(fam, "mp-forward")] = True
        expect_pass[(fam, "mp-bidirectional")] = expect_col_cond[fam]
        expect_pass[(fam, "base-forward")] = (fam == "uniform")
        expect_pass[(fam, "base-bidirectional")] = (fam == "uniform")
    print()
    print(f"  {'family':14s}  " + "  ".join(f"{labels[n]:>18s}" for n, _ in variants))
    for fam in FAMILIES:
        cells = []
        for name, _ in variants:
            e = errs[(fam, name)]
            passed = e < args.tol
            t2_ok = t2_ok and (passed == expect_pass[(fam, name)])
            cells.append(f"{e:.2e} {'PASS' if passed else 'FAIL'}")
        print(f"  {fam:14s}  " + "  ".join(f"{c:>18s}" for c in cells))
    print("  (expected: mp-forward PASS on all five; mp-bidirectional PASS exactly on the")
    print("   families satisfying the column condition -- uniform / doubly-stoch /")
    print(f"   general-bidir; edgewise-MLP {base_label} PASS on uniform duplication only)")
    if forward_only:
        print(f"   No mp-bidirectional column: the {args.family} matrix-product model is")
        print("   forward-only in every experiment.")

    # -- Test 3 -------------------------------------------------------------
    print("\n=== Test 3: spectral continuity on the Hadamard family ===")
    widths = [16, 32, 64, 128, 256]
    scored = {}
    for cf in CONTINUITY_FAMILIES:
        mp_diffs = check_spectral_continuity(mp_conf, widths, "forward", cf)
        gmn_diffs = check_spectral_continuity(base_conf, widths, "forward", cf)
        print(f"  {cf}: |f(theta_n) - f(theta_n, W^(2) = 0)|   "
              f"(the layer-2 distance is 1/sqrt(n) -> 0)")
        print(f"    {'n':>5}  {'1/sqrt(n)':>10}  {'matrix-product':>15}  {base_label:>15}")
        for n in widths:
            print(f"    {n:5d}  {n ** -0.5:10.4f}  {mp_diffs[n]:15.3e}  {gmn_diffs[n]:15.3e}")
        # The matrix-product model must decay (Lipschitz in the normalized spectral norm).
        vacuous = mp_diffs[widths[0]] < args.tol and gmn_diffs[widths[0]] < args.tol
        decays = mp_diffs[widths[-1]] < 0.25 * mp_diffs[widths[0]]
        if vacuous:
            # scale-equivariant models annihilate the zero-context family: no evidence
            # either way, so this variant is reported but not scored
            print("    vacuous: f(theta_n) == f(theta_n, W^(2) = 0) for both models at "
                  "every n (degree-one homogeneity + zero context)")
        else:
            print(f"    matrix-product decays with n: {'PASS' if decays else 'FAIL'}   "
                  f"({base_label} ratio n={widths[-1]}/n={widths[0]}: "
                  f"{gmn_diffs[widths[-1]] / max(gmn_diffs[widths[0]], 1e-12):.2f})")
            scored[cf] = decays
        print()
    # every non-vacuous variant must decay, and at least one must be non-vacuous
    t3_ok = bool(scored) and all(scored.values())
    if not scored:
        print("  every continuity family was vacuous -- nothing was tested: FAIL")

    print("\n" + "=" * 60)
    all_ok = t1_ok and t2_ok and t3_ok
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""
Checks for the **conv** matrix-product GMN (``gnn_args.message_fn_type:
matrix_product_conv``, :mod:`src.models.conv_matrix_product_layer`).

The CNN analog of ``scripts/check_matrix_product_gmn.py``.  On a CNN graph an edge
carries a whole zero-padded kernel in R^9 rather than a scalar, so the MSG function
is constrained to be **bilinear** in the kernel and the source node state, with one
learned map per kernel offset::

    msg_{u -> v} = sum_s a_{u,v}[s] * (h_u A_s),      A in R^{9 x d_in_v x d_hid}

which with ``aggregator: mean`` makes the forward aggregate a sum of 9 matrix
products, one per spatial offset::

    Z^(l) = (1 / n_{l-1}) sum_s Wbar_s^(l) H^(l-1) A_s,   Wbar_s^(l) in R^{n_l x n_{l-1}}

so the input weights enter the metanetwork only through matrix multiplication.

Three tests, mirroring the MLP script:

Test 1 — conv matrix-product identity
    The aggregate produced by PyG message passing equals the closed form above,
    and is 0 on the input layer (no incoming weight matrix).

Test 2 — equivalence classes, forward vs bidirectional
    Base network: the zoo CNN ``[1] -> conv3x3/s2 x3 -> GAP -> dense -> [10]``.
    Channel multipliers ``k_1 = k_2 = k_3 = k``, widened with the five families of
    :data:`src.data.zoo_cnn.FAMILIES`.  All satisfy the forward (row) condition

        (row)  sum_a B_{ij}[b, a, s] = W[i, j, s]

    which is exactly what preserves the function (a convolution is linear in its
    input channels), and they differ in whether they also satisfy the backward
    (column) condition

        (col)  sum_b B_{ij}[b, a, s] = (k_l / k_{l-1}) W[i, j, s].

    Expected, by the same algebra as the scalar case: the FORWARD variant preserves
    all five (both at node level and at graph level), the BIDIRECTIONAL variant
    exactly the three that also satisfy (col), and the edgewise-MLP plain GMN only
    uniform duplication.  Node-level equivalence is checked as well as graph-level,
    because it is the readout-independent statement.

    Each family is also checked to be a genuine functional equivalence of the CNN,
    to satisfy the conditions it claims, and (except for ``uniform``) to be
    measurably far from the uniform widening, so nothing passes vacuously.

Test 3 — spectral continuity
    Conv Hadamard family: the *second* conv layer gets ``Wbar^(2)[:, :, centre] =
    H_n`` (i.e. ``W^(2) = H_n / n`` on the centre kernel offset), every other
    weight and bias zero.  Its normalized spectral norm is
    ``sqrt(n_1/n_2) ||W^(2)||_2 = 1/sqrt(n) -> 0`` (a single-offset kernel, so this
    also bounds the conv operator norm), and a model Lipschitz w.r.t. that norm
    must satisfy ``f(theta_n) -> f(0)``.  The conv matrix-product GMN does; the
    plain (edgewise-MLP) GMN need not, and empirically does not.

    Reported for both readouts, but only ``layerwise_mean`` is scored.  With
    ``last_layer`` the comparison is vacuous for the matrix-product model at any
    depth: the dense layer's weights are zero on this family and the message is
    bilinear in the kernel, so its messages into the output nodes are *identically*
    zero and the difference is exactly 0 at every ``n``.  (At the default
    ``num_layers=2`` the plain GMN reads 0 there as well, for an unrelated reason —
    the perturbation enters at the second conv, two hops upstream of the output
    nodes, so two rounds of message passing cannot carry it that far.  With
    ``--num-layers 4`` it is nonzero and roughly flat in ``n``, as expected.)

Usage
-----
    python scripts/check_matrix_product_gmn_cnn.py
    python scripts/check_matrix_product_gmn_cnn.py --width 8 --k 2 3 --n-nets 3
    python scripts/check_matrix_product_gmn_cnn.py --only 1 3
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

from src.scalegmn.models import ScaleGMN                                    # noqa: E402
from src.utils.helpers import overwrite_conf                                # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from data.cnn_zoo_dataset import build_cnn_graph_data                       # noqa: E402
from data.zoo_cnn import (                                                  # noqa: E402
    FAMILIES, ZooCNN, check_family_conditions, layer_layout, state_dict_to_wb,
    widen_wb, zoo_cnn_forward,
)
from models.last_layer_readout import install_last_layer_readout             # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout      # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context          # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from check_duplication_equiv_cnn import (                                   # noqa: E402
    compare_node_feats, copy_backbone, graph_output, node_features,
)

CONFIGS = PROJECT_ROOT / "configs" / "cifar10_predgen"

#: --family -> (matrix-product config, edgewise-MLP contrast config, contrast label).
#: `gmn` is the conv matrix-product GMN (symmetry: permutation, message_fn_type:
#: matrix_product_conv); `scalegmn` is the conv matrix-product ScaleGMN (symmetry: scale,
#: message_fn_type: matrix_product_conv_scale) -- the same bilinear MSG constraint on top
#: of ScaleGMN's node states and EquivariantNet update.
FAMILY_CONFS = {
    "gmn": (CONFIGS / "mpgmn_sizegen_sp_v1.yml",
            CONFIGS / "gmn_sizegen_sp_v1.yml", "plain GMN"),
    "scalegmn": (CONFIGS / "mpsgmn_sizegen_sp_v1.yml",
                 CONFIGS / "scalegmn_sizegen_sp_v1.yml", "plain ScaleGMN"),
}

#: Families with no bidirectional model. There is no bidirectional conv matrix-product
#: ScaleGMN: the `mpsgmn_*` configs are forward-only in every experiment, so Test 2 has no
#: `mp-bidirectional` column for them and the column condition is exercised on the `gmn`
#: family, which is where bidirectional is actually run.
FORWARD_ONLY_FAMILIES = {"scalegmn"}

#: set from --family in main()
MP_CONF = FAMILY_CONFS["gmn"][0]
GMN_CONF = FAMILY_CONFS["gmn"][1]
BASE_LABEL = FAMILY_CONFS["gmn"][2]

D_HID = 32
NUM_LAYERS = 2   # overridable with --num-layers

#: index of the centre offset in the row-major flattening of a 3x3 kernel
CENTRE = 4

#: set from --num-layers; None keeps NUM_LAYERS
NUM_LAYERS_OVERRIDE = None


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_conf(conf_path, base_layout, direction):
    """Load a predgen config and shrink it to a fast, self-consistent test model.

    Always in the duplication-equivariant setting (``aggregator: mean``), which is
    the only one the matrix-product MSG is defined in.
    """
    with open(conf_path) as f:
        conf = yaml.safe_load(f)
    conf = overwrite_conf(conf, {})
    a = conf["scalegmn_args"]
    a["d_hid"] = D_HID
    a["num_layers"] = NUM_LAYERS_OVERRIDE or NUM_LAYERS
    a["layer_layout"] = base_layout
    a["direction"] = direction
    a["readout_range"] = "full_graph"          # our readouts install into this slot
    a["gnn_args"]["aggregator"] = "mean"
    return overwrite_conf(conf, {})            # re-propagate d_hid (leaves d_edge alone)


def build_model(conf_path, base_layout, direction, readout="last_layer"):
    conf = build_conf(conf_path, base_layout, direction)
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    if readout == "layerwise_mean":
        install_layerwise_mean_readout(model, conf["scalegmn_args"])
    elif readout == "last_layer":
        install_last_layer_readout(model, conf["scalegmn_args"])
    else:
        raise ValueError(readout)
    model.eval()
    return model


def graphs(weights, biases, direction="forward"):
    """The graph the dup-equiv conditions feed the model (fan-in rescaled)."""
    return build_cnn_graph_data(weights, biases, direction=direction,
                                fanin_rescale=True, score=0.0,
                                node_pos_embed=True, edge_pos_embed=False,
                                equiv_on_hidden=True, get_first_layer_mask=False)


# ---------------------------------------------------------------------------
# Test 1: conv matrix-product identity
# ---------------------------------------------------------------------------

def check_conv_matrix_product_identity(model, data, layout):
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

        # Closed form: Z^(l) = (1 / n_{l-1}) sum_s Wbar_s^(l) H^(l-1) A_s
        num_nodes = sum(layout)
        d_e = batch.edge_attr.shape[-1]
        dense_w = torch.zeros(num_nodes, num_nodes, d_e)
        dense_w[batch.edge_index[0], batch.edge_index[1]] = batch.edge_attr
        A = layer.message_fn.A                                  # [9, d_in_v, d_hid]

        expected = torch.zeros_like(aggr)
        offsets = torch.cumsum(torch.tensor([0] + list(layout)), dim=0)
        for l in range(1, len(layout)):
            rows = slice(offsets[l], offsets[l + 1])            # destinations, layer l
            cols = slice(offsets[l - 1], offsets[l])            # sources, layer l-1
            w_bar = dense_w[cols, rows].permute(1, 0, 2)        # [n_l, n_{l-1}, 9]
            expected[rows] = torch.einsum(
                "vus,ud,sdk->vk", w_bar, batch.x[cols], A) / layout[l - 1]

    err = (aggr - expected).abs().max().item()
    in_layer_err = aggr[: layout[0]].abs().max().item()
    return err, in_layer_err


# ---------------------------------------------------------------------------
# Test 3: spectral continuity (conv Hadamard family)
# ---------------------------------------------------------------------------

def hadamard(n):
    """Sylvester Hadamard matrix, n a power of two."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    assert h.shape[0] == n, f"{n} is not a power of two"
    return h


def hadamard_cnn_wb(n, k=3, in_channels=1, num_classes=10, num_conv_layers=3):
    """theta_n: the second conv's centre offset is H_n / n, everything else zero.

    Shapes follow :mod:`src.data.zoo_cnn` (``[out, in, kh, kw]`` per conv, then
    ``[out, in]`` for the dense head).  ``build_cnn_graph_data(fanin_rescale=True)``
    multiplies each layer by its fan-in ``n_{l-1}``, so the *graph* sees
    ``Wbar^(2)[:, :, centre] = n * (H_n / n) = H_n``.
    """
    w2 = torch.zeros(n, n, k, k)
    w2.view(n, n, k * k)[:, :, CENTRE] = hadamard(n) / n
    weights = [torch.zeros(n, in_channels, k, k), w2] \
        + [torch.zeros(n, n, k, k) for _ in range(num_conv_layers - 2)] \
        + [torch.zeros(num_classes, n)]
    biases = [torch.zeros(n) for _ in range(num_conv_layers)] + [torch.zeros(num_classes)]
    return weights, biases


def zero_cnn_wb(n, **kwargs):
    weights, biases = hadamard_cnn_wb(n, **kwargs)
    return [torch.zeros_like(w) for w in weights], biases


def context_cnn_wb(n, k=3, in_channels=1, num_classes=10, num_conv_layers=3,
                   hadamard_layer_2=True):
    """The conv Hadamard perturbation on top of a fixed, width-independent context.

    Needed because the all-zero context of :func:`hadamard_cnn_wb` is *annihilated* by a
    scale-equivariant model: its message and update are degree-one homogeneous, so with
    all biases zero (hence zero hidden node features) and zero ``W^(1)`` every hidden
    state stays exactly zero and ``f(theta_n) == f(0) == 0`` at every ``n``, for the
    matrix-product and the edgewise-MLP ScaleGMN alike -- a vacuous pass.

    Only the second conv is perturbed; the other layers and all biases hold fixed values
    whose fan-in-rescaled edge features ``Wbar^(l) = n_{l-1} W^(l)`` are O(1) at every
    width, so the normalized-spectral-norm distance between the two graphs is still that
    of the second conv alone, ``sqrt(n_1/n_2) ||W^(2)||_2 = 1/sqrt(n) -> 0``.
    """
    w1 = torch.zeros(n, in_channels, k, k)
    w1.view(n, in_channels, k * k)[:, :, CENTRE] = \
        torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0)[:, None]     # two node classes
    w2 = torch.zeros(n, n, k, k)
    if hadamard_layer_2:
        w2.view(n, n, k * k)[:, :, CENTRE] = hadamard(n) / n
    w_mid = []
    for _ in range(num_conv_layers - 2):
        w = torch.zeros(n, n, k, k)
        w.view(n, n, k * k)[:, :, CENTRE] = 1.0 / n                   # Wbar centre = 1
        w_mid.append(w)
    dense = (1.0 + torch.arange(num_classes)[:, None] / num_classes) / n
    weights = [w1, w2] + w_mid + [dense]
    biases = [0.5 * torch.ones(n), -0.25 * torch.ones(n)] \
        + [0.25 * torch.ones(n) for _ in range(num_conv_layers - 2)] \
        + [0.1 * torch.ones(num_classes)]
    return weights, biases


#: Perturbed / reference weight pairs for Test 3. `hadamard` is the family of the paper
#: (zero context); `hadamard+context` adds the fixed nonzero context above.
CONTINUITY_FAMILIES = {
    "hadamard": (hadamard_cnn_wb, zero_cnn_wb),
    "hadamard+context": (context_cnn_wb,
                         lambda n, **kw: context_cnn_wb(n, hadamard_layer_2=False, **kw)),
}


def normalized_spec_norms(weights, layout):
    """``sqrt(n_{l-1}/n_l) * ||W^(l)||_2``, convs flattened to ``[c_out, c_in kh kw]``.

    The study's definition (see ``BatchedZooCNNTrainer.spectral_norm_components``).
    """
    out = []
    for l, w in enumerate(weights):
        flat = w.reshape(w.shape[0], -1)
        out.append((layout[l] / layout[l + 1]) ** 0.5
                   * torch.linalg.matrix_norm(flat, ord=2).item())
    return out


def check_spectral_continuity(conf_path, widths, readout, continuity_family="hadamard"):
    """``|f(theta_n) - f(theta_n, W^(2) = 0)|`` as n grows, for a single fixed model.

    The learned parameters are width-independent and ``node2type`` has the same 14
    values at every width, so one model built at the smallest width can be reused
    across the whole sweep.
    """
    perturbed, reference = CONTINUITY_FAMILIES[continuity_family]
    model = build_model(conf_path, layer_layout(widths[0]), "forward", readout)
    diffs, norms = {}, {}
    for n in widths:
        w_had, b_had = perturbed(n)
        w_ref, b_ref = reference(n)
        norms[n] = normalized_spec_norms(
            [a - b for a, b in zip(w_had, w_ref)], layer_layout(n))
        out_had = graph_output(model, graphs(w_had, b_had))
        out_ref = graph_output(model, graphs(w_ref, b_ref))
        diffs[n] = (out_had - out_ref).abs().max().item()
    return diffs, norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILY_CONFS), default="gmn",
                   help="Which conv matrix-product family to check: on top of the plain "
                        "GMN (symmetry: permutation) or of ScaleGMN (symmetry: scale)")
    p.add_argument("--width", type=int, default=8, help="Base channel count")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3],
                   help="Channel multipliers for the equivalence test")
    p.add_argument("--n-nets", type=int, default=3, help="Random CNNs per k")
    p.add_argument("--spectral-widths", type=int, nargs="+",
                   default=[16, 32, 64, 128], help="Powers of two for the Hadamard family")
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                   help="GNN message-passing rounds (Test 3 needs >= 3 for the "
                        "difference to reach the output nodes)")
    p.add_argument("--only", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--readout", choices=["last_layer", "layerwise_mean"], default="last_layer",
                   help="Readout Test 2 scores at graph level.")
    p.add_argument("--include-bidir-scalegmn", action="store_true",
                   help="Also build and score a bidirectional matrix-product ScaleGMN and a "
                        "bidirectional plain ScaleGMN in Test 2, via "
                        "MatrixProductScale_GNN_layer_aggr (models/matrix_product_scale_layer.py), "
                        "which exists specifically so this contrast can be completed even though no "
                        "training experiment uses a bidirectional mpsgmn. No-op for --family gmn, "
                        "which already builds both directions.")
    return p.parse_args()


def test1(args, base_layout):
    print("\n=== Test 1: conv matrix-product identity (forward aggregate) ===")
    model = build_model(MP_CONF, base_layout, "forward")
    max_err = max_in_err = 0.0
    for _ in range(args.n_nets):
        weights, biases = state_dict_to_wb(ZooCNN(width=args.width).export_state_dict())
        err, in_err = check_conv_matrix_product_identity(
            model, graphs(weights, biases), base_layout)
        max_err = max(max_err, err)
        max_in_err = max(max_in_err, in_err)
    print(f"  aggregate vs (1/n_l-1) sum_s Wbar_s H A_s : max |err| = {max_err:.3e}  "
          f"{'PASS' if max_err < args.tol else 'FAIL'}")
    print(f"  input-layer aggregate == 0                : max |err| = {max_in_err:.3e}  "
          f"{'PASS' if max_in_err < args.tol else 'FAIL'}")
    return max_err < args.tol and max_in_err < args.tol


def test2(args, base_layout):
    print("\n=== Test 2: equivalence classes, forward vs bidirectional ===")
    k_all = [1] + [args.k[0]] * (len(base_layout) - 2) + [1]
    print(f"  base layout={base_layout}  multipliers per k: {k_all} (k from {args.k})")

    include_bd = args.family not in FORWARD_ONLY_FAMILIES or args.include_bidir_scalegmn
    variants = [("mp-forward", build_model(MP_CONF, base_layout, "forward", args.readout))]
    if include_bd:
        variants.append(
            ("mp-bidirectional", build_model(MP_CONF, base_layout, "bidirectional", args.readout)))
    variants.append(("gmn-forward", build_model(GMN_CONF, base_layout, "forward", args.readout)))
    if include_bd:
        variants.append(
            ("gmn-bidirectional",
             build_model(GMN_CONF, base_layout, "bidirectional", args.readout)))
    variants = tuple(variants)
    node_errs = {(fam, name): 0.0 for fam in FAMILIES for name, _ in variants}
    graph_errs = {(fam, name): 0.0 for fam in FAMILIES for name, _ in variants}
    cnn_errs = {fam: 0.0 for fam in FAMILIES}
    cond = {fam: [0.0, 0.0, float("inf")] for fam in FAMILIES}   # row, col, min nonunif

    x = torch.randn(8, 1, 32, 32)
    for k in args.k:
        mults = [1] + [k] * (len(base_layout) - 2) + [1]
        for _ in range(args.n_nets):
            weights, biases = state_dict_to_wb(ZooCNN(width=args.width).export_state_dict())
            y_base = zoo_cnn_forward(weights, biases, x)
            for fam in FAMILIES:
                wide_w, wide_b = widen_wb(weights, biases, mults, fam)
                # the widened CNN must compute the same function ...
                y_wide = zoo_cnn_forward(wide_w, wide_b, x)
                cnn_errs[fam] = max(cnn_errs[fam],
                                    (y_base - y_wide).abs().max().item())
                # ... and be the family it claims to be
                r_err, c_err, nonunif = check_family_conditions(weights, wide_w, mults)
                cond[fam][0] = max(cond[fam][0], r_err)
                cond[fam][1] = max(cond[fam][1], c_err)
                cond[fam][2] = min(cond[fam][2], nonunif)

                for name, model in variants:
                    direction = ("bidirectional" if name.endswith("bidirectional")
                                 else "forward")
                    g_base = graphs(weights, biases, direction)
                    g_wide = graphs(wide_w, wide_b, direction)
                    node_errs[(fam, name)] = max(
                        node_errs[(fam, name)],
                        compare_node_feats(node_features(model, g_base),
                                           node_features(model, g_wide),
                                           base_layout, mults))
                    graph_errs[(fam, name)] = max(
                        graph_errs[(fam, name)],
                        (graph_output(model, g_base)
                         - graph_output(model, g_wide)).abs().max().item())

    expect_col = {"uniform": True, "row-stoch": False, "doubly-stoch": True,
                  "general-bidir": True, "general": False}
    ok = True
    print(f"  {'family':14s} {'CNN equiv':>10s} {'row cond':>10s} {'col cond':>10s} "
          f"{'nonunif':>9s}")
    for fam in FAMILIES:
        r_err, c_err, nonunif = cond[fam]
        ok = ok and cnn_errs[fam] < 1e-4                    # genuine functional equivalence
        ok = ok and r_err < args.tol                        # all families satisfy (row)
        ok = ok and ((c_err < args.tol) == expect_col[fam])
        ok = ok and (fam == "uniform" or nonunif > 1e-2)    # not vacuously uniform
        print(f"  {fam:14s} {cnn_errs[fam]:10.2e} {r_err:10.2e} {c_err:10.2e} "
              f"{nonunif:9.2e}")

    expect_pass = {}
    for fam in FAMILIES:
        expect_pass[(fam, "mp-forward")] = True
        expect_pass[(fam, "mp-bidirectional")] = expect_col[fam]
        expect_pass[(fam, "gmn-forward")] = (fam == "uniform")
        expect_pass[(fam, "gmn-bidirectional")] = (fam == "uniform")

    # The contrast column is scored only for the plain-GMN family. On CONV graphs the
    # edgewise ScaleGMN message is degree-one homogeneous in the kernel, which makes it
    # invariant to *first order* in the widening perturbation whenever the column
    # condition holds: its errors there sit at 1e-6..1e-4 (init- and k-dependent),
    # straddling the fixed tolerance. So it is reported, not asserted -- the substantive
    # claim, that it is not exactly equivalent, is carried by the row-only families.
    score_contrast = args.family == "gmn"
    for level, errs in (("node", node_errs), (f"graph ({args.readout} readout)", graph_errs)):
        print(f"\n  {level}-level max |error|")
        gmn_labels = {"gmn-forward": BASE_LABEL, "gmn-bidirectional": f"{BASE_LABEL} (bd)"}
        print(f"  {'family':14s}  " + "  ".join(
            f"{gmn_labels.get(n, n):>18s}" for n, _ in variants))
        for fam in FAMILIES:
            cells = []
            for name, _ in variants:
                e = errs[(fam, name)]
                passed = e < args.tol
                if score_contrast or name not in ("gmn-forward", "gmn-bidirectional"):
                    ok = ok and (passed == expect_pass[(fam, name)])
                cells.append(f"{e:.2e} {'PASS' if passed else 'FAIL'}")
            print(f"  {fam:14s}  " + "  ".join(f"{c:>18s}" for c in cells))
    print("  (expected: mp-forward PASS on all five; mp-bidirectional PASS exactly on the")
    print("   families satisfying the column condition -- uniform / doubly-stoch /")
    print(f"   general-bidir; edgewise-MLP {BASE_LABEL} PASS on uniform duplication only)")
    if args.family in FORWARD_ONLY_FAMILIES and not include_bd:
        print(f"  (no mp-bidirectional column: the conv matrix-product {args.family} model")
        print("   is forward-only in every experiment)")
    elif args.family in FORWARD_ONLY_FAMILIES:
        print(f"  (--include-bidir-scalegmn: the conv matrix-product {args.family} model is "
              "forward-only in\n   every training experiment, but its bidirectional MSG+AGGR path "
              "(models/matrix_product_scale_layer.py) exists specifically to complete this contrast)")
    if not score_contrast:
        print(f"  (the {BASE_LABEL} column is reported, not scored: its message is "
              "degree-one homogeneous\n   in the kernel, so on the column-condition "
              "families it is invariant to first order\n   in the perturbation and its "
              "errors straddle the tolerance; the row-only families\n   are what show it "
              "is not exactly equivalent)")
    return ok


def test3(args):
    print("\n=== Test 3: spectral continuity on the conv Hadamard family ===")
    widths = args.spectral_widths
    norms = None
    scored = {}
    for cf in CONTINUITY_FAMILIES:
        for readout in ("layerwise_mean", "last_layer"):
            mp_diffs, norms = check_spectral_continuity(MP_CONF, widths, readout, cf)
            gmn_diffs, _ = check_spectral_continuity(GMN_CONF, widths, readout, cf)
            print(f"\n  {cf}, readout = {readout}:   "
                  f"|f(theta_n) - f(theta_n, W^(2) = 0)|")
            print(f"    {'n':>5}  {'1/sqrt(n)':>10}  {'||dtheta_n||':>12}  "
                  f"{'matrix-product':>15}  {BASE_LABEL:>15}")
            for n in widths:
                print(f"    {n:5d}  {n ** -0.5:10.4f}  {max(norms[n]):12.4f}  "
                      f"{mp_diffs[n]:15.3e}  {gmn_diffs[n]:15.3e}")
            if readout != "layerwise_mean":
                print("    (not scored: with the last_layer readout the comparison is "
                      "vacuous for the\n     matrix-product model on the zero-context "
                      "family -- W^(L) = 0 there and the message\n     is bilinear in "
                      "the kernel, so its messages into the output nodes are identically"
                      "\n     zero -- and at the default num_layers=2 the perturbation "
                      "starts two hops upstream\n     of the output nodes and cannot "
                      "reach them at all.)")
                continue
            # the Lipschitz claim: decay with n for the matrix-product model
            if mp_diffs[widths[0]] < args.tol and gmn_diffs[widths[0]] < args.tol:
                # scale-equivariant models annihilate the zero-context family: no
                # evidence either way, so this variant is reported but not scored
                print("    vacuous: f(theta_n) == f(theta_n, W^(2) = 0) for both models "
                      "at every n\n     (degree-one homogeneity + zero context)")
                continue
            decays = mp_diffs[widths[-1]] < 0.25 * mp_diffs[widths[0]]
            ratio = gmn_diffs[widths[-1]] / max(gmn_diffs[widths[0]], 1e-12)
            print(f"    matrix-product decays with n: {'PASS' if decays else 'FAIL'}   "
                  f"({BASE_LABEL} ratio n={widths[-1]}/n={widths[0]}: {ratio:.2f})")
            scored[cf] = decays
    n0 = widths[0]
    print(f"\n  normalized spectral norms of the perturbation at n={n0}, per layer: "
          + "  ".join(f"{v:.4f}" for v in norms[n0])
          + f"\n  (confined to the second conv; ||dtheta_n|| = 1/sqrt(n) -> 0)")
    if not scored:
        print("  every continuity family was vacuous -- nothing was tested: FAIL")
    # every non-vacuous variant must decay, and at least one must be non-vacuous
    return bool(scored) and all(scored.values())


def main():
    global NUM_LAYERS_OVERRIDE, MP_CONF, GMN_CONF, BASE_LABEL
    args = parse_args()
    NUM_LAYERS_OVERRIDE = args.num_layers
    MP_CONF, GMN_CONF, BASE_LABEL = FAMILY_CONFS[args.family]
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    base_layout = layer_layout(args.width)
    print(f"Conv matrix-product checks, family = {args.family}")
    print(f"base layout={base_layout}  d_hid={D_HID}  num_layers={args.num_layers}  "
          f"tol={args.tol:.0e}")
    print(f"configs: {MP_CONF.name} (mp) vs {GMN_CONF.name} ({BASE_LABEL}), both dup-equiv")

    results = {}
    if 1 in args.only:
        results["Test 1 (identity)"] = test1(args, base_layout)
    if 2 in args.only:
        results["Test 2 (equivalence)"] = test2(args, base_layout)
    if 3 in args.only:
        results["Test 3 (spectral)"] = test3(args)

    print("\n" + "=" * 60)
    for name, ok in results.items():
        print(f"  {name:<24} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

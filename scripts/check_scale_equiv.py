"""
Check that the ScaleGMN family is equivariant / invariant to the neuron-wise positive
rescaling symmetry of ReLU MLPs -- the symmetry ScaleGMN is named after.

The symmetry
------------
A ReLU MLP with layout [d_0, n_1, ..., n_{L-1}, d_L] is unchanged as a function by

    W^(l)[i, j] -> (lam^(l)_j / lam^(l-1)_i) * W^(l)[i, j]
    b^(l)_j     -> lam^(l)_j * b^(l)_j
    lam^(0) = 1_{d_0},   lam^(L) = 1_{d_L},   lam^(l) > 0 elementwise

because ReLU is positively homogeneous. Test 0 below re-verifies that on the sampled
theta -> theta_lam pairs before anything else is measured.

What the model should do with it
-------------------------------
ScaleGMN's hidden node states are degree-one homogeneous: an EquivariantLayer computes
W(x) * g(x) with W bias-free and g the scale canonicalization x / (||x|| + eps), so
h_v -> lam_v h_v. I/O nodes carry lam = 1 and must come out unchanged. The graph output
is then invariant, because both readouts canonicalize the hidden states before pooling
and concatenate the (already invariant) I/O states raw.

Concretely, per graph object:
  * node feature of a hidden node = its bias                  -> scales by lam_v
  * node feature of an input node = the constant 1            -> invariant
  * edge feature of (u -> v)      = W[u, v] (x fan-in)        -> scales by lam_v / lam_u
  * MSG   w_e(e) (*) w_v(h_u)  =  (lam_v/lam_u)(lam_u) (...)  -> scales by lam_v
  * MSG   e * (h_u A)          =  (lam_v/lam_u)(lam_u) (...)  -> scales by lam_v
  * AGGR  mean / add                                          -> scales by lam_v
  * UPD   EquivariantNet(cat(h_v, z_v))                       -> scales by lam_v
so both the edgewise ScaleGMN and the matrix-product ScaleGMN are exactly equivariant in
the forward direction. Fan-in rescaling (a positive constant per layer) and mean vs. add
aggregation are irrelevant to the property, which the table checks explicitly.

Tests
-----
  0. functional equivalence of theta and theta_lam (sanity check on the transform)
  1. node level: max |h_v(theta_lam) - lam_v h_v(theta)|, hidden and I/O nodes separately
  2. graph level: max |f(theta_lam) - f(theta)| under the upstream invariant readout and
     under LayerWiseMeanReadout. `bidir` is the bidirectional model as *defined* -- with
     the reciprocal backward edge feature every entry point installs
     (src/models/bidir_reciprocal.py) -- and must PASS. `bd:raw` is the uncorrected
     upstream code path, kept as a must-FAIL regression guard; it is a bug, not a
     configuration of the model. The matrix-product ScaleGMN has forward rows only
     (FORWARD_ONLY): there is no bidirectional variant of it in any experiment.
  3. the same, over widening lam ranges, to show the error is float32 noise and not a
     small-perturbation artefact
  4. the reciprocal feature, isolated: ScaleGMN_GNN_bidir.forward initialises
     bw_edge_attr = batch.edge_attr, i.e. the *forward* weights, so a backward message
     into u scales by lam_v^2 / lam_u instead of lam_u and the property is lost. The
     backward direction needs the reciprocal edge feature 1/W[u, v] ~ lam_u/lam_v, which
     is what upstream's `reciprocal: True` flag (present in every ScaleGMN config,
     "only for bidirectional") asks for -- but the flag is only assigned, at
     models.py:305, and never read, and only one upstream dataset
     (cifar10_dataset.py:509) even builds the field, which models.py:418 then discards.
     Upstream's own INR experiments use symmetry: sign, where 1/lam = lam makes the
     omission invisible; symmetry: scale is what exposes it.
     `install_bidir_reciprocal` (src/models/bidir_reciprocal.py) wires it up and recovers
     exact equivariance, which pins the cause to that line. This test scores the
     production function itself, not a copy of it.

The permutation-symmetry models (plain GMN, matrix-product GMN) are must-FAIL contrast
columns: their messages concatenate / consume the raw edge feature with no homogeneity
structure, so nothing cancels.

Usage
-----
    python scripts/check_scale_equiv.py
    python scripts/check_scale_equiv.py --models scalegmn mpsgmn --n-inrs 10
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
from src.data.data_utils import (                                           # noqa: E402
    get_node_types, get_edge_types, nn_to_edge_index, to_pyg_batch,
)
from src.utils.helpers import overwrite_conf                                # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from models.layerwise_mean_readout import install_layerwise_mean_readout    # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context        # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal                # noqa: E402

#: label -> architecture config. The symmetry, message_fn_type and positional-encoding
#: settings are read off the config, not overridden: they *are* the model.
MODELS = {
    "scalegmn": "configs/mnist_cls/scalegmn_sizegen_sp_v3.yml",
    "mpsgmn":   "configs/mnist_cls/mpsgmn_sizegen_sp_v3.yml",
    "gmn":      "configs/mnist_cls/gmn_sizegen_sp_v3.yml",
    "mpgmn":    "configs/mnist_cls/mpgmn_sizegen_sp_v3.yml",
}

#: Models whose experiments only ever run the duplication-equivariant setting
#: (fan-in rescaled edges + mean aggregation); the others get both settings.
DUP_EQUIV_ONLY = {"mpsgmn", "mpgmn"}

#: Models with no bidirectional variant in any experiment. There is no bidirectional
#: matrix-product ScaleGMN, so it is checked in the forward direction only.
FORWARD_ONLY = {"mpsgmn"}


# ---------------------------------------------------------------------------
# The symmetry
# ---------------------------------------------------------------------------

def state_dict_to_wb(state_dict):
    """Extract (weights, biases) in ScaleGMN's [n_in, n_out, 1] convention."""
    weights = tuple(v.permute(1, 0).unsqueeze(-1) for k, v in state_dict.items() if "weight" in k)
    biases = tuple(v.unsqueeze(-1) for k, v in state_dict.items() if "bias" in k)
    return weights, biases


def infer_layout(weights):
    return [weights[0].shape[0]] + [w.shape[1] for w in weights]


def sample_lambdas(layout, log_range, generator):
    """One positive lam per neuron; lam = 1 on the input and output layers."""
    lambdas = [torch.ones(layout[0])]
    for n_l in layout[1:-1]:
        u = torch.rand(n_l, generator=generator) * 2 - 1          # U[-1, 1]
        lambdas.append(torch.exp(u * torch.log(torch.tensor(float(log_range)))))
    lambdas.append(torch.ones(layout[-1]))
    return lambdas


def rescale_wb(weights, biases, lambdas):
    """Apply the ReLU rescaling symmetry. Weights are [n_in, n_out, 1]."""
    new_w, new_b = [], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        lam_in, lam_out = lambdas[l], lambdas[l + 1]
        new_w.append(W * (lam_out.view(1, -1, 1) / lam_in.view(-1, 1, 1)))
        new_b.append(b * lam_out.view(-1, 1))
    return tuple(new_w), tuple(new_b)


def mlp_forward(weights, biases, x):
    """ReLU MLP forward, weights in [n_in, n_out, 1]."""
    for i, (W, b) in enumerate(zip(weights, biases)):
        x = x @ W.squeeze(-1) + b.squeeze(-1)
        if i < len(weights) - 1:
            x = torch.relu(x)
    return x


def node_lambdas(lambdas):
    """lam per graph node, in the [input | hidden | output] node order."""
    return torch.cat(lambdas).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def batch_to_graphs(weights, biases):
    """Dense node/edge feature matrices, mirroring BaseDataset.batch_to_graphs."""
    num_nodes = weights[0].shape[0] + sum(w.shape[1] for w in weights)
    node_features = torch.zeros(num_nodes, biases[0].shape[-1])
    edge_features = torch.zeros(num_nodes, num_nodes, weights[0].shape[-1])

    row_off, col_off = 0, weights[0].shape[0]
    for w in weights:
        n_in, n_out, _ = w.shape
        edge_features[row_off:row_off + n_in, col_off:col_off + n_out] = w
        row_off += n_in
        col_off += n_out

    row_off = weights[0].shape[0]
    node_features[:row_off] = 1.0          # input node state = 1
    for b in biases:
        node_features[row_off:row_off + b.shape[0]] = b
        row_off += b.shape[0]

    return node_features, edge_features


def build_pyg_data(weights, biases, layout, fanin_rescale, direction):
    if fanin_rescale:
        weights = tuple(w * w.shape[0] for w in weights)

    node_feats, edge_feats = batch_to_graphs(weights, biases)
    hidden_mask = torch.tensor(
        [False] * layout[0] + [True] * sum(layout[1:-1]) + [False] * layout[-1]
    ).unsqueeze(-1)

    data, _ = to_pyg_batch(
        node_feats, edge_feats, nn_to_edge_index(layout, device="cpu"),
        node2type=get_node_types(layout),
        edge2type=get_edge_types(layout),
        direction=direction,
        label=torch.tensor([0]),
        hidden_nodes=hidden_mask,
    )
    return data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_conf(template, layout, aggregator, direction):
    with open(PROJECT_ROOT / template) as f:
        conf = yaml.safe_load(f)
    conf = overwrite_conf(conf, {})
    args = conf["scalegmn_args"]
    args["d_hid"] = 32                       # small, to keep the check fast
    args["num_layers"] = 2
    args["layer_layout"] = layout
    args["direction"] = direction
    args["readout_range"] = "full_graph"
    args["gnn_args"]["aggregator"] = aggregator
    return overwrite_conf(conf, {})          # re-propagate d_hid


def build_model(template, layout, aggregator, direction, readout):
    conf = build_conf(template, layout, aggregator, direction)
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    # stashed so patch_bw_reciprocal() can re-read the settings the installer gates on
    model.conf_gnn_args_for_check = conf["scalegmn_args"]
    if readout == "layerwise_mean":
        install_layerwise_mean_readout(model, conf["scalegmn_args"])
    elif readout != "upstream":
        raise ValueError(readout)
    model.eval()
    return model


def copy_backbone(src_model, dst_model):
    """Copy every shape-compatible parameter (i.e. all but the readout)."""
    src_sd, dst_sd = src_model.state_dict(), dst_model.state_dict()
    for key in dst_sd:
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst_model.load_state_dict(dst_sd)
    dst_model.eval()


def patch_bw_reciprocal(model):
    """Install the production repair (`src/models/bidir_reciprocal.py`) on `model`.

    That module is what the training / eval entry points install, so Test 4 scores the
    shipped fix rather than a copy of it. It reciprocates the **raw scalar** edge feature
    and pushes it through the same bias-free projection and the same product edge
    positional encoding as the forward feature; reciprocating the already-projected
    d_edge vector instead is badly conditioned (~2e-5, above tolerance).

    The install is a no-op unless the config asks for it, so force the gate here: this
    test deliberately measures both the patched and unpatched bidirectional models.
    """
    conf = dict(model.conf_gnn_args_for_check)
    assert install_bidir_reciprocal(model, conf), "the repair did not install"
    return model


def node_features(model, data):
    batch = torch_geometric.data.Batch.from_data_list([data])
    model.gnn.equivariant = True
    with torch.no_grad():
        x, _ = model(batch)
    model.gnn.equivariant = False
    return x


def graph_output(model, data):
    batch = torch_geometric.data.Batch.from_data_list([data])
    with torch.no_grad():
        return model(batch)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def max_err(got, want):
    """max |got - want| / max(1, max |want|): a mixed absolute/relative error.

    Purely relative would be wrong here -- a random readout can put the graph output
    near zero, and dividing by it turns float32 noise into a large-looking ratio.
    Purely absolute would be wrong too -- lam up to 4 per layer inflates the hidden
    states, so the same float32 noise shows up as a bigger number. Flooring the
    denominator at 1 gives a relative error where the reference is large and an
    absolute one where it is small. The reference magnitudes are printed with each
    table so the numbers can be read either way.
    """
    return (got - want).abs().max().item() / max(1.0, want.abs().max().item())


def node_errors(x_base, x_lam, lam_nodes, hidden_mask):
    """Equivariance error on hidden nodes and invariance error on I/O nodes."""
    target = lam_nodes * x_base
    h, io = hidden_mask.squeeze(-1), ~hidden_mask.squeeze(-1)
    return max_err(x_lam[h], target[h]), max_err(x_lam[io], target[io])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = Path(os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))) \
        / "mnist_inrs" / "w24_sp"
    p.add_argument("--data-dir", type=Path, default=default_dir)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--n-inrs", type=int, default=5)
    p.add_argument("--lam-range", type=float, default=4.0,
                   help="lam ~ LogUniform[1/r, r] for the main table")
    p.add_argument("--lam-sweep", type=float, nargs="+", default=[2.0, 4.0, 16.0, 64.0],
                   help="lam ranges for Test 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    files = sorted(args.data_dir.rglob("*.pth"))[: args.n_inrs]
    if not files:
        print(f"[ERROR] No .pth files found under {args.data_dir}")
        sys.exit(1)

    thetas = [state_dict_to_wb(torch.load(f, map_location="cpu", weights_only=True))
              for f in files]
    layout = infer_layout(thetas[0][0])
    for w, _ in thetas:
        assert infer_layout(w) == layout, "all INRs must share one layout"

    hidden_mask = torch.tensor(
        [False] * layout[0] + [True] * sum(layout[1:-1]) + [False] * layout[-1]
    ).unsqueeze(-1)

    print("Scale equivariance of the ScaleGMN family (ReLU MLP / INR graphs)")
    print(f"{len(files)} INRs from {args.data_dir}  layout={layout}  "
          f"lam ~ LogUniform[1/{args.lam_range:g}, {args.lam_range:g}]  "
          f"tol={args.tol:.0e}  float32")
    print()

    rng = torch.Generator()
    rng.manual_seed(args.seed)
    lambdas = [sample_lambdas(layout, args.lam_range, rng) for _ in thetas]
    rescaled = [rescale_wb(w, b, lam) for (w, b), lam in zip(thetas, lambdas)]

    # ---- Test 0: the transform preserves the represented function ----------
    print("=" * 88)
    print("Test 0: theta_lam represents the same function as theta")
    print("=" * 88)
    x = torch.rand(256, layout[0], generator=rng) * 2 - 1
    f_err = max(max_err(mlp_forward(rw, rb, x), mlp_forward(w, b, x))
                for (w, b), (rw, rb) in zip(thetas, rescaled))
    lam_lo = min(lam.min().item() for lams in lambdas for lam in lams)
    lam_hi = max(lam.max().item() for lams in lambdas for lam in lams)
    ok = f_err < args.tol
    print(f"  [{'PASS' if ok else 'FAIL'}]  max |f(theta_lam)(x) - f(theta)(x)| = "
          f"{f_err:.2e}   (lam in [{lam_lo:.3f}, {lam_hi:.3f}])")
    print()
    all_ok = ok

    # ---- Tests 1 & 2: node-level equivariance, graph-level invariance ------
    print("=" * 88)
    print("Tests 1 & 2: node-level equivariance and graph-level invariance")
    print("=" * 88)
    header = (f"{'model':<10} {'dir':<10} {'edges/aggr':<16} "
              f"{'node hidden':>12} {'node I/O':>10} {'graph upstm':>12} {'graph lw-mean':>14}"
              f"  verdict")
    print(header)
    print("-" * len(header))

    for name in args.models:
        template = MODELS[name]
        sym = build_conf(template, layout, "mean", "forward")["scalegmn_args"]["symmetry"]
        msg_fn = build_conf(template, layout, "mean", "forward")["scalegmn_args"]["gnn_args"] \
            .get("message_fn_type", "edgewise_mlp")
        expect_pass = sym == "scale"

        settings = [("fan-in + mean", True, "mean")]
        if name not in DUP_EQUIV_ONLY:
            settings.append(("raw + add", False, "add"))

        ref_h = ref_f = 0.0
        # `bidir` is the model as *defined*: a bidirectional scale model consumes the
        # reciprocal backward edge feature, which every entry point installs
        # (install_bidir_reciprocal). `bd:raw` is the uncorrected upstream code path, kept
        # as a must-FAIL regression guard -- it is a bug, not a configuration.
        variants = [("forward", False)]
        if name not in FORWARD_ONLY:
            if expect_pass:
                variants += [("bidirectional", True), ("bidirectional", False)]
            else:
                variants.append(("bidirectional", False))
        for direction, recip in variants:
            m_node = build_model(template, layout, "mean", direction, "upstream")
            m_lw = build_model(template, layout, "mean", direction, "layerwise_mean")
            copy_backbone(m_node, m_lw)
            if recip:
                patch_bw_reciprocal(m_node)
                patch_bw_reciprocal(m_lw)

            for label, fanin, aggr in settings:
                if aggr != "mean":
                    m_node_a = build_model(template, layout, aggr, direction, "upstream")
                    m_lw_a = build_model(template, layout, aggr, direction, "layerwise_mean")
                    copy_backbone(m_node, m_node_a)
                    copy_backbone(m_node, m_lw_a)
                    if recip:
                        patch_bw_reciprocal(m_node_a)
                        patch_bw_reciprocal(m_lw_a)
                else:
                    m_node_a, m_lw_a = m_node, m_lw

                e_hid = e_io = e_up = e_lw = 0.0
                for (w, b), (rw, rb), lams in zip(thetas, rescaled, lambdas):
                    g0 = build_pyg_data(w, b, layout, fanin, direction)
                    g1 = build_pyg_data(rw, rb, layout, fanin, direction)
                    lam_n = node_lambdas(lams)

                    h0, h1 = node_features(m_node_a, g0), node_features(m_node_a, g1)
                    eh, ei = node_errors(h0, h1, lam_n, hidden_mask)
                    e_hid, e_io = max(e_hid, eh), max(e_io, ei)
                    f_up, f_lw = graph_output(m_node_a, g0), graph_output(m_lw_a, g0)
                    ref_h = max(ref_h, (lam_n * h0).abs().max().item())
                    ref_f = max(ref_f, f_up.abs().max().item(), f_lw.abs().max().item())
                    e_up = max(e_up, max_err(graph_output(m_node_a, g1), f_up))
                    e_lw = max(e_lw, max_err(graph_output(m_lw_a, g1), f_lw))

                errs = (e_hid, e_io, e_up, e_lw)
                # I/O nodes are the lam = 1 special case of the same node-level check,
                # so they are reported separately rather than scored: for the scale
                # models they add nothing to the hidden column, and for the contrast
                # models only redundant evidence of failure.
                passed = max(e_hid, e_up, e_lw) < args.tol
                want = expect_pass and (direction == "forward" or recip)
                correct = passed == want
                all_ok = all_ok and correct
                dir_col = ("bd:raw" if direction == "bidirectional" and not recip
                           and expect_pass else direction[:5])
                print(f"{name:<10} {dir_col:<10} {label:<16} "
                      + " ".join(f"{e:>12.2e}" if i != 1 else f"{e:>10.2e}"
                                 for i, e in enumerate(errs))
                      + f"  {'OK ' if correct else 'BAD'} [{'PASS' if passed else 'FAIL'}]"
                        f" should {'PASS' if want else 'FAIL'}")
        print(f"{'':<10} ^ symmetry={sym}, message_fn_type={msg_fn}; "
              f"reference magnitudes max|lam_v h_v|={ref_h:.2e}, max|f|={ref_f:.2e}")

    # ---- Test 3: lam-range sweep (forward, scale models only) --------------
    scale_models = [n for n in args.models
                    if build_conf(MODELS[n], layout, "mean", "forward")["scalegmn_args"]["symmetry"] == "scale"]
    if scale_models:
        print()
        print("=" * 88)
        print("Test 3: the forward error does not grow with the size of the rescaling")
        print("=" * 88)
        hdr = f"{'model':<10} " + " ".join(f"{'r=' + f'{r:g}':>14}" for r in args.lam_sweep)
        print(hdr)
        print("-" * len(hdr))
        for name in scale_models:
            m = build_model(MODELS[name], layout, "mean", "forward", "layerwise_mean")
            row = []
            for r in args.lam_sweep:
                g = torch.Generator()
                g.manual_seed(args.seed + 1)
                worst = 0.0
                for w, b in thetas:
                    lams = sample_lambdas(layout, r, g)
                    rw, rb = rescale_wb(w, b, lams)
                    worst = max(worst, max_err(
                        graph_output(m, build_pyg_data(rw, rb, layout, True, "forward")),
                        graph_output(m, build_pyg_data(w, b, layout, True, "forward"))))
                row.append(worst)
                all_ok = all_ok and worst < args.tol
            print(f"{name:<10} " + " ".join(f"{v:>14.2e}" for v in row))
        print("  (max |f(theta_lam) - f(theta)|, graph level, layer-wise mean readout)")

    # ---- Test 4: bidirectional diagnostic ---------------------------------
    if scale_models:
        print()
        print("=" * 88)
        print("Test 4: the reciprocal backward edge feature -- upstream code vs definition")
        print("=" * 88)
        hdr = (f"{'model':<10} {'bw edge feature':<34} {'node hidden':>12} "
               f"{'graph lw-mean':>14}")
        print(hdr)
        print("-" * len(hdr))
        for name in [n for n in scale_models if n not in FORWARD_ONLY]:
            m_plain = build_model(MODELS[name], layout, "mean", "bidirectional", "upstream")
            m_plain_lw = build_model(MODELS[name], layout, "mean", "bidirectional", "layerwise_mean")
            copy_backbone(m_plain, m_plain_lw)
            m_recip = build_model(MODELS[name], layout, "mean", "bidirectional", "upstream")
            m_recip_lw = build_model(MODELS[name], layout, "mean", "bidirectional", "layerwise_mean")
            copy_backbone(m_plain, m_recip)
            copy_backbone(m_plain, m_recip_lw)
            patch_bw_reciprocal(m_recip)
            patch_bw_reciprocal(m_recip_lw)

            for label, mn, ml, want in (
                ("batch.edge_attr (upstream code path)", m_plain, m_plain_lw, False),
                ("proj(1/w) (*) pe  (as defined)", m_recip, m_recip_lw, True),
            ):
                e_hid = e_lw = 0.0
                for (w, b), (rw, rb), lams in zip(thetas, rescaled, lambdas):
                    g0 = build_pyg_data(w, b, layout, True, "bidirectional")
                    g1 = build_pyg_data(rw, rb, layout, True, "bidirectional")
                    eh, _ = node_errors(node_features(mn, g0), node_features(mn, g1),
                                        node_lambdas(lams), hidden_mask)
                    e_hid = max(e_hid, eh)
                    e_lw = max(e_lw, max_err(graph_output(ml, g1), graph_output(ml, g0)))
                passed = max(e_hid, e_lw) < args.tol
                correct = passed == want
                all_ok = all_ok and correct
                print(f"{name:<10} {label:<34} {e_hid:>12.2e} {e_lw:>14.2e}  "
                      f"{'OK ' if correct else 'BAD'} [{'PASS' if passed else 'FAIL'}]"
                      f" should {'PASS' if want else 'FAIL'}")
        print("  (the second row wires up what `reciprocal: True` asks for: the reciprocal of")
        print("   the RAW scalar weight, pushed through the same projection and edge PE)")

    print()
    if all_ok:
        print("[OK] All conditions behaved as expected.")
    else:
        print("[FAIL] Some conditions did not behave as expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()

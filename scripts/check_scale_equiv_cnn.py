"""
Check that the ScaleGMN family is equivariant / invariant to the per-channel positive
rescaling symmetry of the ReLU zoo CNN -- the CNN analogue of
``scripts/check_scale_equiv.py``.

The symmetry on a CNN
---------------------
For per-channel lam^(l) > 0 on the three conv layers (lam = 1 on the single input
channel and on the 10 output logits),

    conv:  W^(l)[i, j, :, :] -> (lam^(l)_i / lam^(l-1)_j) * W^(l)[i, j, :, :]
    dense: W^(L)[i, j]       -> (1 / lam^(L-1)_j) * W^(L)[i, j]
    bias:  b^(l)_i           -> lam^(l)_i * b^(l)_i

leaves the function unchanged: a convolution is linear in its input channels, ReLU is
positively homogeneous, and global average pooling is linear and per-channel, so the
1/lam of the dense head cancels the lam carried by the pooled features. Test 0 verifies
this on the sampled pairs.

On the graph, each edge (u -> v) carries the whole (zero-padded) R^9 kernel of one
(in-channel, out-channel) pair, so the edge feature scales by lam_v / lam_u exactly as a
scalar MLP weight does, and node features (biases) scale by lam_v. Both ScaleGMN MSG
functions therefore produce messages that scale by lam_v:

    edgewise         w_e(e) (*) w_v(h_u)                      -> lam_v
    conv matrix-prod sum_s a_{u,v}[s] * (h_u A_s)             -> lam_v

and the ScaleGMN node update, being degree-one homogeneous, keeps that. Both CNN
readouts are invariant: ``LastLayerReadout`` reads only the 10 output nodes (lam = 1) and
``LayerWiseMeanReadout`` canonicalizes hidden states before pooling.

Tests
-----
  0. functional equivalence of theta and theta_lam
  1. node level: max |h_v(theta_lam) - lam_v h_v(theta)|, hidden and I/O nodes separately
  2. graph level: max |f(theta_lam) - f(theta)| under LayerWiseMeanReadout and
     LastLayerReadout
  3. the same over widening lam ranges
  4. the reciprocal backward edge feature: the upstream code path against the definition

The permutation-symmetry models (plain GMN, conv matrix-product GMN) are must-FAIL
contrast columns. The conv matrix-product ScaleGMN is forward-only in every experiment
(FORWARD_ONLY), so it gets no bidirectional row at all.

A bidirectional ``symmetry: scale`` model is *defined* with the reciprocal backward edge
feature ``1/W[u, v]``; without it the backward messages carry ``lam_v^2/lam_u`` and there
is no equivariance to speak of. Upstream never wires its own ``reciprocal`` flag up
(``ScaleGMN_GNN_bidir.forward`` feeds the *forward* features to the backward layers), so
the ``bidirectional`` rows here install ``install_bidir_reciprocal``, exactly as
``scripts/train_sizegen_predgen.py`` does, and are scored as must-PASS. The uncorrected
variant is kept only as a regression guard (``bidir(raw)``, must-FAIL) and as Test 4's
audit of the upstream code path -- it is not a configuration of the model.

Conv graphs need one extra convention: 20.7% of the R^9 edge components are structurally
zero (the dense head's 1x1 kernels are zero-padded), so plain ``torch.reciprocal`` is
``inf`` there, but the 1/0 := 0 pseudo-inverse ``install_bidir_reciprocal`` uses is finite
and law-preserving (zero is a fixed point of the rescaling). See
``results/VERIFICATION_gmn_properties.md``.

Usage
-----
    python scripts/check_scale_equiv_cnn.py
    python scripts/check_scale_equiv_cnn.py --models scalegmn mpsgmn --width 16
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
    ZooCNN, layer_layout, state_dict_to_wb, zoo_cnn_forward,
)
from models.last_layer_readout import install_last_layer_readout            # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout    # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal              # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context        # noqa: E402

#: label -> architecture config. symmetry / message_fn_type are read off the config.
MODELS = {
    "scalegmn": "configs/cifar10_predgen/scalegmn_sizegen_sp_v1.yml",
    "mpsgmn":   "configs/cifar10_predgen/mpsgmn_sizegen_sp_v1.yml",
    "gmn":      "configs/cifar10_predgen/gmn_sizegen_sp_v1.yml",
    "mpgmn":    "configs/cifar10_predgen/mpgmn_sizegen_sp_v1.yml",
}

#: Models whose experiments only ever run the duplication-equivariant setting.
DUP_EQUIV_ONLY = {"mpsgmn", "mpgmn"}

#: Models with no bidirectional variant in any experiment: the conv matrix-product
#: ScaleGMN is forward-only by scope, so it gets no bidirectional row anywhere.
FORWARD_ONLY = {"mpsgmn"}


# ---------------------------------------------------------------------------
# The symmetry
# ---------------------------------------------------------------------------

def sample_lambdas(layout, log_range, generator):
    """One positive lam per channel; lam = 1 on the input channel and the logits."""
    lambdas = [torch.ones(layout[0])]
    for n_l in layout[1:-1]:
        u = torch.rand(n_l, generator=generator) * 2 - 1          # U[-1, 1]
        lambdas.append(torch.exp(u * torch.log(torch.tensor(float(log_range)))))
    lambdas.append(torch.ones(layout[-1]))
    return lambdas


def rescale_wb(weights, biases, lambdas):
    """Apply the per-channel rescaling. Weights are [out, in, kh, kw] / [out, in]."""
    new_w, new_b = [], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        lam_in, lam_out = lambdas[l], lambdas[l + 1]
        if W.ndim == 4:
            factor = lam_out.view(-1, 1, 1, 1) / lam_in.view(1, -1, 1, 1)
        else:
            factor = lam_out.view(-1, 1) / lam_in.view(1, -1)
        new_w.append(W * factor)
        new_b.append(b * lam_out)
    return new_w, new_b


def node_lambdas(lambdas):
    """lam per graph node, in the [input | hidden | output] node order."""
    return torch.cat(lambdas).unsqueeze(-1)


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
    args["readout_range"] = "full_graph"     # our readouts install into this slot
    args["gnn_args"]["aggregator"] = aggregator
    return overwrite_conf(conf, {})          # re-propagate d_hid


def build_model(template, layout, aggregator, direction, readout):
    conf = build_conf(template, layout, aggregator, direction)
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    if readout == "layerwise_mean":
        install_layerwise_mean_readout(model, conf["scalegmn_args"])
    elif readout == "last_layer":
        install_last_layer_readout(model, conf["scalegmn_args"])
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
    target = lam_nodes * x_base
    h, io = hidden_mask, ~hidden_mask
    return max_err(x_lam[h], target[h]), max_err(x_lam[io], target[io])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--width", type=int, default=16, help="Channel count")
    p.add_argument("--n-nets", type=int, default=3, help="Random CNNs to test")
    p.add_argument("--lam-range", type=float, default=4.0,
                   help="lam ~ LogUniform[1/r, r] for the main table")
    p.add_argument("--lam-sweep", type=float, nargs="+", default=[2.0, 4.0, 16.0, 64.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    return p.parse_args()


def graphs(weights, biases, fanin, direction):
    return build_cnn_graph_data(weights, biases, direction=direction,
                               fanin_rescale=fanin, score=0.0,
                               node_pos_embed=True, edge_pos_embed=False,
                               equiv_on_hidden=True, get_first_layer_mask=False)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    layout = layer_layout(args.width)
    hidden_mask = torch.tensor(
        [False] * layout[0] + [True] * sum(layout[1:-1]) + [False] * layout[-1]
    )

    thetas = [state_dict_to_wb(ZooCNN(width=args.width).export_state_dict())
              for _ in range(args.n_nets)]

    print("Scale equivariance of the ScaleGMN family (zoo-CNN graphs)")
    print(f"{args.n_nets} random CNNs  layout={layout}  "
          f"lam ~ LogUniform[1/{args.lam_range:g}, {args.lam_range:g}]  "
          f"tol={args.tol:.0e}  float32 (TF32 off)")
    print()

    rng = torch.Generator()
    rng.manual_seed(args.seed)
    lambdas = [sample_lambdas(layout, args.lam_range, rng) for _ in thetas]
    rescaled = [rescale_wb(w, b, lam) for (w, b), lam in zip(thetas, lambdas)]

    # ---- Test 0 -----------------------------------------------------------
    print("=" * 92)
    print("Test 0: theta_lam represents the same function as theta")
    print("=" * 92)
    x = torch.randn(8, layout[0], 32, 32, generator=rng)
    f_err = max(max_err(zoo_cnn_forward(rw, rb, x), zoo_cnn_forward(w, b, x))
                for (w, b), (rw, rb) in zip(thetas, rescaled))
    lam_lo = min(lam.min().item() for lams in lambdas for lam in lams)
    lam_hi = max(lam.max().item() for lams in lambdas for lam in lams)
    ok = f_err < args.tol
    print(f"  [{'PASS' if ok else 'FAIL'}]  max |f(theta_lam)(x) - f(theta)(x)| = "
          f"{f_err:.2e}   (lam in [{lam_lo:.3f}, {lam_hi:.3f}])")
    print()
    all_ok = ok

    # ---- Tests 1 & 2 ------------------------------------------------------
    print("=" * 92)
    print("Tests 1 & 2: node-level equivariance and graph-level invariance")
    print("=" * 92)
    header = (f"{'model':<10} {'dir':<6} {'edges/aggr':<16} "
              f"{'node hidden':>12} {'node I/O':>10} {'graph lw-mean':>14} "
              f"{'graph last-layer':>17}  verdict")
    print(header)
    print("-" * len(header))

    for name in args.models:
        template = MODELS[name]
        base_conf = build_conf(template, layout, "mean", "forward")["scalegmn_args"]
        sym = base_conf["symmetry"]
        msg_fn = base_conf["gnn_args"].get("message_fn_type", "edgewise_mlp")
        expect_pass = sym == "scale"

        settings = [("fan-in + mean", True, "mean")]
        if name not in DUP_EQUIV_ONLY:
            settings.append(("raw + add", False, "add"))

        ref_h = ref_f = 0.0
        # `bidir` is the model as defined -- with the reciprocal backward edge feature the
        # entry points install. `bidir(raw)` is the uncorrected upstream code path, kept as
        # a must-FAIL regression guard, not as a configuration of the model.
        variants = [("forward", False)]
        if name not in FORWARD_ONLY:
            if expect_pass:
                variants += [("bidirectional", True), ("bidirectional", False)]
            else:
                variants.append(("bidirectional", False))
        for direction, recip in variants:
            for label, fanin, aggr in settings:
                m_node = build_model(template, layout, aggr, direction, "upstream")
                m_lw = build_model(template, layout, aggr, direction, "layerwise_mean")
                m_ll = build_model(template, layout, aggr, direction, "last_layer")
                copy_backbone(m_node, m_lw)
                copy_backbone(m_node, m_ll)
                if recip:
                    conf = build_conf(template, layout, aggr,
                                      direction)["scalegmn_args"]
                    for m in (m_node, m_lw, m_ll):
                        assert install_bidir_reciprocal(m, dict(conf)), \
                            "the reciprocal backward feature did not install"

                e_hid = e_io = e_lw = e_ll = 0.0
                for (w, b), (rw, rb), lams in zip(thetas, rescaled, lambdas):
                    g0 = graphs(w, b, fanin, direction)
                    g1 = graphs(rw, rb, fanin, direction)
                    lam_n = node_lambdas(lams)

                    h0 = node_features(m_node, g0)
                    eh, ei = node_errors(h0, node_features(m_node, g1), lam_n, hidden_mask)
                    e_hid, e_io = max(e_hid, eh), max(e_io, ei)
                    f_lw, f_ll = graph_output(m_lw, g0), graph_output(m_ll, g0)
                    ref_h = max(ref_h, (lam_n * h0).abs().max().item())
                    ref_f = max(ref_f, f_lw.abs().max().item(), f_ll.abs().max().item())
                    e_lw = max(e_lw, max_err(graph_output(m_lw, g1), f_lw))
                    e_ll = max(e_ll, max_err(graph_output(m_ll, g1), f_ll))

                passed = max(e_hid, e_lw, e_ll) < args.tol
                want = expect_pass and (direction == "forward" or recip)
                correct = passed == want
                all_ok = all_ok and correct
                dir_tag = direction[:5] if direction == "forward" or recip else "bd:raw"
                print(f"{name:<10} {dir_tag:<6} {label:<16} "
                      f"{e_hid:>12.2e} {e_io:>10.2e} {e_lw:>14.2e} {e_ll:>17.2e}  "
                      f"{'OK ' if correct else 'BAD'} [{'PASS' if passed else 'FAIL'}]"
                      f" should {'PASS' if want else 'FAIL'}")
        print(f"{'':<10} ^ symmetry={sym}, message_fn_type={msg_fn}; "
              f"reference magnitudes max|lam_v h_v|={ref_h:.2e}, max|f|={ref_f:.2e}")

    # ---- Test 3 -----------------------------------------------------------
    scale_models = [n for n in args.models
                    if build_conf(MODELS[n], layout, "mean", "forward")["scalegmn_args"]["symmetry"] == "scale"]
    if scale_models:
        print()
        print("=" * 92)
        print("Test 3: the forward error does not grow with the size of the rescaling")
        print("=" * 92)
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
                        graph_output(m, graphs(rw, rb, True, "forward")),
                        graph_output(m, graphs(w, b, True, "forward"))))
                row.append(worst)
                all_ok = all_ok and worst < args.tol
            print(f"{name:<10} " + " ".join(f"{v:>14.2e}" for v in row))
        print("  (max |f(theta_lam) - f(theta)|, graph level, layer-wise mean readout)")

    # ---- Test 4 -----------------------------------------------------------
    if scale_models:
        print()
        print("=" * 92)
        print("Test 4: the reciprocal backward edge feature -- upstream code vs definition")
        print("=" * 92)
        g = graphs(*thetas[0], True, "bidirectional")
        n_zero = (g.edge_attr == 0).sum().item()
        print(f"  edge features are R^9 kernels; {n_zero}/{g.edge_attr.numel()} "
              f"({n_zero / g.edge_attr.numel():.1%}) components are structurally zero "
              f"(the dense\n  head's 1x1 kernels are zero-padded), so the reciprocal is "
              f"carried entirely by the 1/0 := 0\n  convention. That keeps the "
              f"lam_u/lam_v law componentwise, so the repair is at least well defined\n"
              f"  here -- this test measures whether it is also sufficient.")
        print()
        hdr = (f"{'model':<10} {'bw edge feature':<35} "
               f"{'node hidden':>12} {'graph lw-mean':>14} {'graph last-layer':>17}")
        print(hdr)
        print("-" * len(hdr))
        for name in [n for n in scale_models if n not in FORWARD_ONLY]:
            template = MODELS[name]
            for patched in (False, True):
                m_node = build_model(template, layout, "mean", "bidirectional", "upstream")
                m_lw = build_model(template, layout, "mean", "bidirectional", "layerwise_mean")
                m_ll = build_model(template, layout, "mean", "bidirectional", "last_layer")
                copy_backbone(m_node, m_lw)
                copy_backbone(m_node, m_ll)
                if patched:
                    conf = build_conf(template, layout, "mean", "bidirectional")["scalegmn_args"]
                    for m in (m_node, m_lw, m_ll):
                        assert install_bidir_reciprocal(m, dict(conf))

                e_hid = e_lw = e_ll = 0.0
                for (w, b), (rw, rb), lams in zip(thetas, rescaled, lambdas):
                    g0 = graphs(w, b, True, "bidirectional")
                    g1 = graphs(rw, rb, True, "bidirectional")
                    lam_n = node_lambdas(lams)
                    h0 = node_features(m_node, g0)
                    eh, _ = node_errors(h0, node_features(m_node, g1), lam_n, hidden_mask)
                    e_hid = max(e_hid, eh)
                    f_lw, f_ll = graph_output(m_lw, g0), graph_output(m_ll, g0)
                    e_lw = max(e_lw, max_err(graph_output(m_lw, g1), f_lw))
                    e_ll = max(e_ll, max_err(graph_output(m_ll, g1), f_ll))

                label = ("pinv(edge_attr), 1/0 := 0" if patched
                         else "batch.edge_attr (upstream code path)")
                print(f"{name:<10} {label:<35} "
                      f"{e_hid:>12.2e} {e_lw:>14.2e} {e_ll:>17.2e}")
        print("  (the second row of each pair is the model as defined, and is what Tests 1 & 2")
        print("   score; the first is the upstream code path, i.e. the bug. See")
        print("   results/VERIFICATION_gmn_properties.md for what the numbers mean.)")

    print()
    if all_ok:
        print("[OK] All conditions behaved as expected.")
    else:
        print("[FAIL] Some conditions did not behave as expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()

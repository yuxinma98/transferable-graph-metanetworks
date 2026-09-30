"""Reciprocal backward edge features for bidirectional `symmetry: scale` models.

Why this exists
---------------
The ReLU rescaling symmetry acts on an MLP as

    W^(l)[i, j] -> (lam_j / lam_i) W^(l)[i, j],   b^(l)_j -> lam_j b^(l)_j

with lam = 1 on the input/output nodes. ScaleGMN's hidden node states are degree-one
homogeneous (`EquivariantLayer` = bias-free `W(x) * x/(||x||+eps)`), so equivariance
means `h_v -> lam_v h_v`.

*Forward* messages get that for free: an edge `u -> v` carries `lam_v/lam_u` and `h_u`
carries `lam_u`, so the product is `lam_v`. A *backward* layer aggregates at `u` over the
same edge, where the available factors are `e_uv ~ lam_v/lam_u` and `h_v ~ lam_v`; the
only power product that yields `lam_u` is `e_uv^{-1} h_v`. The backward layers must
therefore consume the **reciprocal** edge feature `1/W[u, v]`.

Upstream ships the switch for this -- `reciprocal: True`, commented "only for
bidirectional", in every ScaleGMN config -- but never reads it: `models.py:305` only
assigns `self.reciprocal`, and `ScaleGMN_GNN_bidir.forward` (`models.py:418`) hardcodes
`bw_edge_attr = batch.edge_attr`, i.e. the forward features, so a backward message into
`u` scales by `lam_v^2/lam_u` and the property is lost. The omission is invisible in
upstream's own INR experiments because those are `symmetry: sign`, where `1/lam = lam`.

What this module does
---------------------
`install_bidir_reciprocal(model, conf)` reciprocates the **raw scalar** edge feature and
pushes it through the same bias-free `GraphInit.proj_weight` and the same product edge
positional encoding as the forward feature, so the backward feature entering round 0 is
`proj(1/w) (*) pe`. Rounds > 0 are threaded exactly as `models.py` threads its own
`bw_edge_attr` -- through `bw_update_edge_attr_fn`, whose `EdgeUpdate` is degree-one
homogeneous in the edge feature and hence law-preserving -- but starting from the
corrected value.

Reciprocating the *projected* `d_edge` vector instead is mathematically equivalent for a
bias-free projection composed with a diagonal PE only in the scalar case; numerically it
is badly conditioned and lands at ~2e-5 rather than ~3e-7. Reciprocate the raw scalar.

Exact zeros take the scalar pseudo-inverse convention `1/0 := 0`. That is not a fudge: the
rescaling sends `0 -> (lam_v/lam_u) * 0 = 0`, so zero is a fixed point and `recip(c w) =
c^{-1} recip(w)` holds for every `w` including 0. Any other filler (eps, clamping) would
break the law on those edges. Real fits do contain them -- one weight out of the 2000
w256_sp v3 test INRs is exactly 0.0 -- and the smallest nonzero magnitude seen across the
MNIST widths is 2.7e-10, whose reciprocal 3.7e9 is comfortably inside float32.

Why a wrapper and not a subtree edit: `src/scalegmn/` is a git subtree of upstream and is
not modified in this repo (`git subtree pull` would clobber or conflict with such edits).
This is the same instance-level installation pattern as
`models.layerwise_mean_readout.install_layerwise_mean_readout`, which wraps `gnn.forward`
and so composes with this module in either order.

Scope
-----
Installed on every entry point that can build a bidirectional model: the five INR ones
(`train_sizegen.py`, `train_sizegen_regression.py`, `eval_sizegen_duplicated.py`,
`eval_sizegen_regression_duplicated.py`, `train_scalegmn.py`) and both predgen ones
(`train_sizegen_predgen.py`, `eval_sizegen_predgen_duplicated.py`). It works unchanged on
CNN graphs: ~21% of the R^9 edge-feature components at c=16 are structurally zero (the
dense head's 1x1 kernels are zero-padded), so plain `torch.reciprocal` is `inf` there and
the repair was long assumed unrepresentable, but under the `1/0 := 0` convention it is
finite and law-preserving.

Verified by `scripts/check_scale_equiv.py` (Test 4): as shipped the bidirectional arm
fails at 5.9e-01 (node) / 1.1e-01 (graph); with this installed it is exactly equivariant
at ~3.5e-07 / ~1.6e-07, the same float32 floor as the forward arm.
`scripts/check_scale_equiv_cnn.py` (Test 4) measures ~1.5e-08 (node) / ~1.9e-07 (graph)
on conv graphs, against 2.0e-03 / 2.7e-02 for the shipped code path.
"""

import torch


def wants_bidir_reciprocal(conf):
    """True iff `conf` (a `scalegmn_args` dict) asks for reciprocal backward features.

    Mirrors the gate upstream computes at `models.py:305` and then ignores, plus the
    direction check (the flag is meaningless for `direction: forward`).
    """
    return bool(
        conf.get("direction") == "bidirectional"
        and conf.get("reciprocal", False)
        and conf.get("symmetry") != "permutation"
    )


def install_bidir_reciprocal(model, conf):
    """Feed `1/W[u, v]` to the backward layers of a bidirectional ScaleGMN.

    No-op returning False when `conf` does not ask for it (forward direction,
    `reciprocal: False`, or `symmetry: permutation`), so callers can install
    unconditionally. Returns True when the patch was applied.
    """
    if not wants_bidir_reciprocal(conf):
        return False

    gi = model.construct_graph
    pe = model.positional_embeddings_edge if model.edge_use_pos_embed else None
    if pe is not None:
        assert not (pe.sum_pos_enc or pe.equiv_net or pe.different_linear
                    or pe.final_linear_pos_embed), (
            "install_bidir_reciprocal assumes the plain product edge PE "
            "(edge_attr * pos_embed[edge2type]); this config uses a variant, so the "
            "backward feature would not follow the lam_u/lam_v law"
        )

    gnn = model.gnn
    assert hasattr(gnn, "bw_layers"), "not a bidirectional model"
    orig_model_forward = model.forward

    def model_forward(batch, *a, **kw):
        # Reciprocate the raw scalar weights, then apply exactly the transforms
        # BaseScaleGMN.forward applies to the forward edge features.
        # Exact zeros take the scalar pseudo-inverse 1/0 := 0 -- zero is a fixed point of
        # the rescaling, so this is the one convention that keeps the lam_u/lam_v law.
        e = batch.edge_attr
        bw = torch.where(e == 0, torch.zeros_like(e), torch.reciprocal(e))
        if not torch.isfinite(bw).all():
            raise RuntimeError(
                "reciprocal backward edge features are non-finite even after the 1/0 := 0 "
                "convention, so the forward edge features must themselves contain inf/nan"
            )
        if gi.project_edge_feats:
            bw = gi.proj_weight(bw)
        if pe is not None:
            bw = bw * pe.pos_embed[batch.edge2type]
        model._bw_edge_attr = bw
        return orig_model_forward(batch, *a, **kw)

    model.forward = model_forward

    # `models.py` threads its own (uncorrected) bw_edge_attr through the edge update
    # between rounds, so every bw layer -- not just the first -- must be redirected.
    edge_upd = getattr(gnn, "bw_update_edge_attr_fn", None)

    for i, layer in enumerate(gnn.bw_layers):
        orig_layer_forward = layer.forward

        def layer_forward(_orig=orig_layer_forward, _i=i, **kw):
            kw["edge_attr"] = model._bw_edge_attr
            out = _orig(**kw)
            if edge_upd is not None and _i < len(edge_upd):
                model._bw_edge_attr = edge_upd[_i](
                    kw["x"], kw["edge_index"], model._bw_edge_attr,
                    sign_mask=kw.get("sign_mask"))
            return out

        layer.forward = layer_forward

    return True

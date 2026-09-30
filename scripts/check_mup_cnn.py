"""
Verify the muP CNN implementation against the ``mup`` package. HARD GATE before
generating any zoo: a muP bug here is silent and would invalidate the whole muP arm.

Four checks, in increasing cost:

**V1 — shape inference.**  Instantiate the reference :class:`src.data.zoo_cnn.ZooCNN`
(with ``MuReadout`` as its head) at each width, run ``mup.set_base_shapes`` against the
base-width model, and assert every parameter's ``infshape.ninf()`` and
``infshape.width_mult()`` match the per-layer table in
``src/data/train_zoo_cnns``'s docstring.  Catches trap T1 (kernel dims must be finite,
or ``width_mult`` is read off the wrong dimension).

**V2 — coordinate check.**  Per-layer activation coordinate magnitudes must be flat in
width under muP, over widths 16..512 and ``t = 0..4`` Adam steps.  SP is the negative
control on the same axes and must visibly diverge.  Two variants, both plotted to
``results/figures/``:

* ``V2-ours`` runs :class:`src.data.train_zoo_cnns.BatchedZooCNNTrainer` itself, so it
  tests the init table *and* the lr table we actually ship.  This is the decisive one.
* ``V2-ref`` runs ``mup.coord_check.get_coord_data`` on the reference ``MuReadout``
  model, which checks that the architecture is muP-compatible at all (independent of
  our code) and gives the familiar plot to compare against.

The readout curve legitimately grows between ``t = 0`` and ``t = 1``: muP initializes
the readout at ``O(1/width)``, so the logits start at ``O(width^-1/2)`` and only become
``O(1)`` after the first update.  Flatness is required for ``t >= 1``.

**V3 — batched trainer vs reference.**  One model through ``nn.Conv2d`` +
``mup.set_base_shapes`` + ``mup.MuAdamW`` against our ``vmap`` trainer with ``N = 1``,
same seed and same data, for ``K`` steps; assert the parameters agree to ``--tol``.
Repeated at several widths (exercising the width factors) and several sampled
hyperparameter draws (exercising trap T4's per-model lr vector), plus the pinned
``WD_PROBE_HP`` draw so the weight-decay path is visible against the gate at all, plus an
``[eps]`` row at the inflated ``EPS_PROBE`` where the epsilon table is not numerically
negligible.  Three must-fail controls: ``[ctl-lr]`` drops the hidden convs' ``1/wm``,
``[ctl-wd]`` double-counts ``MuAdam``'s ``wd *= wm`` instead of cancelling it against
``lr /= wm`` (trap T2), and ``[ctl-eps]`` leaves ``eps`` fixed as the ``mup`` package
does (trap T7).  Both sides hold the same variables — our trainer carries ``MuReadout``'s
division in the forward pass as ``ZooCNN.readout_mult``, so ``self.params`` *is* the
reference parameterization and the comparison is direct.  The one thing the reference has
to be *told* is the per-layer ``eps``, which ``mup`` does not implement; it is restated
independently in :func:`eps_table` from V1's per-layer table and passed per param group.

**V4 — export round-trip.**  Reload an exported ``.pth`` through the plain functional
forward ``zoo_cnn_forward`` and assert it matches the trainer's own forward.  This is
what catches trap T3: the exported state dict must fold the readout multiplier in, so
that what the metanetwork reads is ``W_eff = W / wm``.

Usage
-----
    python scripts/check_mup_cnn.py                    # V1, V2-ours, V3, V4, identity
    python scripts/check_mup_cnn.py --coord-check      # + V2-ref (slow)
    python scripts/check_mup_cnn.py --only v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mup import MuAdamW, MuReadout, set_base_shapes  # noqa: E402

from data.train_zoo_cnns import (  # noqa: E402
    BatchedZooCNNTrainer, ZooHP, sample_hps,
)
from data.zoo_cnn import (  # noqa: E402
    NUM_CONV_LAYERS, PADDING, STRIDE, ZooCNN, state_dict_to_wb, zoo_cnn_forward,
)

BASE_WIDTH = 16


def ok(flag):
    return "PASS" if flag else "FAIL"


# ---------------------------------------------------------------------------
# V1 — shape inference
# ---------------------------------------------------------------------------

def expected_infshape(name, width, base_width):
    """``(ninf, width_mult)`` predicted by the per-layer table."""
    wm = width / base_width
    return {
        "convs.0.weight": (1, wm),   # [c, 1, 3, 3] — only dim 0 is infinite
        "convs.0.bias": (1, wm),
        "convs.1.weight": (2, wm),   # [c, c, 3, 3] — last inf dim is c_in
        "convs.1.bias": (1, wm),
        "convs.2.weight": (2, wm),
        "convs.2.bias": (1, wm),
        "fc.weight": (1, wm),        # [10, c] — only dim 1 is infinite
        "fc.bias": (0, 1.0),         # [10] — finite
    }[name]


def mup_reference_model(width, base_width, device="cpu", seed=0):
    """A ``ZooCNN`` with a ``MuReadout`` head and ``set_base_shapes`` applied."""
    torch.manual_seed(seed)
    model = ZooCNN(width=width, readout_cls=MuReadout).to(device)
    base = ZooCNN(width=base_width, readout_cls=MuReadout).to(device)
    delta = ZooCNN(width=base_width * 2, readout_cls=MuReadout).to(device)
    set_base_shapes(model, base, delta=delta)
    return model


def check_v1(widths, base_width=BASE_WIDTH):
    print("=" * 78)
    print("V1 — muP shape inference on the zoo CNN")
    print("=" * 78)
    all_ok = True
    for width in widths:
        model = mup_reference_model(width, base_width)
        rows = []
        for name, p in model.named_parameters():
            exp_ninf, exp_wm = expected_infshape(name, width, base_width)
            got_ninf = p.infshape.ninf()
            got_wm = p.infshape.width_mult()
            good = got_ninf == exp_ninf and abs(got_wm - exp_wm) < 1e-12
            all_ok &= good
            rows.append((name, tuple(p.shape), got_ninf, got_wm, exp_ninf, exp_wm, good))
        print(f"\nwidth {width} (wm = {width / base_width:g})")
        print(f"  {'param':<16} {'shape':<18} {'ninf':>4} {'wm':>7} | "
              f"{'exp':>4} {'exp wm':>7}  ")
        for name, shape, gn, gw, en, ew, good in rows:
            print(f"  {name:<16} {str(shape):<18} {gn:>4} {gw:>7.3f} | "
                  f"{en:>4} {ew:>7.3f}  {ok(good)}")
    print(f"\nV1: {ok(all_ok)}")
    return all_ok


# ---------------------------------------------------------------------------
# V2 — coordinate check
# ---------------------------------------------------------------------------

def _layer_activations(params, x, num_conv=NUM_CONV_LAYERS):
    """Mean ``|coordinate|`` of every layer's output for model 0 of a trainer.

    ``params`` must be *effective* weights, and the forward below is deliberately the
    plain multiplier-free one: this measures the function the exported ``.pth`` will
    compute, not an internal muP-scaled representation.
    """
    acts, h = {}, x
    for i in range(num_conv):
        h = F.relu(F.conv2d(h, params[f"convs.{i}.weight"][0],
                            params[f"convs.{i}.bias"][0],
                            stride=STRIDE, padding=PADDING))
        acts[f"conv{i + 1}"] = float(h.abs().mean())
    g = h.mean(dim=(-2, -1))
    acts["logits"] = float(
        (g @ params["fc.weight"][0].t() + params["fc.bias"][0]).abs().mean())
    return acts


@torch.no_grad()
def _coords_of_trainer(trainer, x):
    return _layer_activations(trainer.effective_weights(), x)


def check_v2_ours(widths, base_width=BASE_WIDTH, nsteps=20, lr=3e-3, batch_size=64,
                  device="cpu", out_dir=None, flat_tol=2.5, nseeds=3):
    """Coordinate check on ``BatchedZooCNNTrainer`` — our init table and our lr table.

    ``flat_tol`` is the largest activation-scale ratio between the widest and the
    narrowest width that still counts as flat.  It is a smoke bar, not a proof: the
    plots are what gate V5 reviews.

    The gate is applied at ``t = nsteps`` only, and ``nsteps`` is 20 rather than the
    usual 3-4 for one reason: muP initializes the readout at ``O(1/width)``, so the
    logits *start* at ``O(width^-1/2)`` (ratio ``~1/sqrt(32) = 0.18`` over these widths)
    and only become width-independent once the learned contribution dominates the
    initial one.  Judging the readout at ``t = 1`` would flag correct muP as broken.  The
    conv layers are flat from ``t = 0`` and every step is printed, so an early-step
    problem is still visible.
    """
    print("=" * 78)
    print("V2-ours — coordinate check on the shipped trainer")
    print("=" * 78)
    hp = ZooHP(lr=lr, init_mult=1.0, weight_decay=0.0, dropout=0.0,
               train_frac=1.0, epochs=1)

    # curves[arm][width_idx][t][module], averaged over seeds
    curves = {}
    for arm in ("mup", "sp"):
        per_width = []
        for w in widths:
            acc = None
            for s in range(nseeds):
                g = torch.Generator().manual_seed(s)
                x = torch.rand(batch_size, 1, 32, 32, generator=g).to(device)
                y = torch.randint(0, 10, (batch_size,), generator=g).to(device)
                tr = BatchedZooCNNTrainer(
                    [hp], w, base_width=base_width if arm == "mup" else None,
                    seed=s, device=device, batch_size=batch_size)
                steps = []
                for _ in range(nsteps + 1):
                    steps.append(_coords_of_trainer(tr, x))
                    tr.step(x.unsqueeze(0), y.unsqueeze(0))
                if acc is None:
                    acc = [{k: v / nseeds for k, v in d.items()} for d in steps]
                else:
                    for a, d in zip(acc, steps):
                        for k, v in d.items():
                            a[k] += v / nseeds
            per_width.append(acc)
        curves[arm] = per_width

    modules = list(curves["mup"][0][0])
    shown = sorted({0, 1, 2, 5, 10, nsteps} & set(range(nsteps + 1)))
    worst = {}
    for arm in ("mup", "sp"):
        print(f"\n{arm}: mean |activation| by width "
              f"({', '.join(str(w) for w in widths)})")
        arm_worst = 1.0
        for module in modules:
            print(f"  {module}")
            for t in shown:
                vals = [curves[arm][i][t][module] for i in range(len(widths))]
                ratio = vals[-1] / vals[0] if vals[0] > 0 else float("inf")
                if t == nsteps:
                    arm_worst = max(arm_worst, ratio, 1.0 / max(ratio, 1e-30))
                print(f"    t={t:<3d}" + " ".join(f"{v:9.4f}" for v in vals) +
                      f"   x{ratio:7.3f}" + ("   <- gated" if t == nsteps else ""))
        worst[arm] = arm_worst
        print(f"  worst width-ratio at t={nsteps}: x{arm_worst:.2f}")

    paths = _plot_coords(curves, widths, modules, nsteps, base_width, lr, out_dir,
                         nseeds)
    # muP must be flat; SP must NOT be — otherwise the "control" proves nothing
    good = worst["mup"] < flat_tol and worst["sp"] > flat_tol
    print(f"\nV2-ours: {ok(good)}  (muP x{worst['mup']:.2f} < {flat_tol} "
          f"< SP x{worst['sp']:.2f})")
    print(f"plots -> {', '.join(str(p) for p in paths)}")
    return good


def _plot_coords(curves, widths, modules, nsteps, base_width, lr, out_dir=None,
                 nseeds=1):
    """One figure per arm. Both share a y-range so the two are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    # a handful of steps, else the legend has more entries than the plot has lines
    ts = sorted({0, 1, 2, 5, 10, nsteps} & set(range(nsteps + 1)))
    everything = [curves[a][i][t][m] for a in curves for i in range(len(widths))
                  for t in ts for m in modules]
    ylim = (min(everything) / 2, max(everything) * 2)

    paths = []
    for arm in ("mup", "sp"):
        fig, axes = plt.subplots(1, len(modules), figsize=(3.1 * len(modules), 3.1),
                                 sharex=True, sharey=True)
        cmap = plt.get_cmap("viridis")
        for ax, module in zip(axes, modules):
            for j, t in enumerate(ts):
                ax.plot(widths, [curves[arm][i][t][module]
                                 for i in range(len(widths))],
                        marker="o", ms=3.5, color=cmap(j / max(1, len(ts) - 1)),
                        label=f"t={t}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_title(module)
            ax.set_xlabel("width")
            ax.grid(alpha=0.25, which="both", lw=0.4)
        axes[0].set_ylabel("mean |activation|")
        axes[-1].legend(fontsize=7, loc="lower left", framealpha=0.85)
        fig.suptitle(f"zoo CNN, {'muP' if arm == 'mup' else 'SP'} (our trainer), "
                     f"adam lr={lr}, base width {base_width}, "
                     f"mean of {nseeds} seed(s)")
        fig.tight_layout()
        path = out_dir / f"coord_check_cnn_ours_{arm}.png"
        fig.savefig(path, dpi=140, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def check_v2_ref(widths, base_width=BASE_WIDTH, nsteps=4, nseeds=3, lr=1e-3,
                 batch_size=64, out_dir=None, device="cpu"):
    """``mup.coord_check`` on the reference ``MuReadout`` model, muP vs SP."""
    from mup.coord_check import get_coord_data, plot_coord_data

    print("=" * 78)
    print("V2-ref — mup.coord_check on the reference model")
    print("=" * 78)
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # a fixed synthetic batch is enough: the coord check measures activation scale,
    # not learning (get_coord_data reuses the first batch by default)
    g = torch.Generator().manual_seed(0)
    data = [(torch.rand(batch_size, 1, 32, 32, generator=g),
             torch.randint(0, 10, (batch_size,), generator=g))
            for _ in range(nsteps + 1)]

    paths = []
    for mode in ("mup", "sp"):
        def make(width, mode=mode):
            def f():
                torch.manual_seed(0)
                if mode == "mup":
                    return mup_reference_model(width, base_width)
                return ZooCNN(width=width)
            return f

        models = {w: make(w) for w in widths}
        df = get_coord_data(models, data, mup=(mode == "mup"), lr=lr,
                            optimizer="adam", nseeds=nseeds, nsteps=nsteps,
                            lossfn="xent", cuda=(str(device) != "cpu"))
        path = out_dir / f"coord_check_cnn_ref_{mode}.png"
        plot_coord_data(df, legend="brief", save_to=str(path), suptitle=(
            f"zoo CNN, {'muP' if mode == 'mup' else 'SP'}, "
            f"adam lr={lr}, base width {base_width}"), face_color="white")
        paths.append(path)

        last = df[df["t"] == nsteps]
        print(f"\n{mode}: mean |activation| at t={nsteps}, by width")
        piv = last.groupby(["module", "width"])["l1"].mean().unstack()
        for module in piv.index:
            vals = piv.loc[module]
            print(f"  {str(module):<26} " +
                  " ".join(f"{v:9.4f}" for v in vals.values) +
                  f"   x{vals.iloc[-1] / vals.iloc[0]:.2f}")
    print(f"\nV2-ref plots -> {', '.join(str(p) for p in paths)}")
    return paths


# ---------------------------------------------------------------------------
# V3 — batched trainer vs mup reference
# ---------------------------------------------------------------------------

#: Adam epsilon at the base width.  Our trainer divides it by ``wm`` on every parameter
#: with an infinite dimension (trap T7); :func:`eps_table` restates that rule
#: independently of the trainer and hands it to the reference per param group, so V3
#: compares the two tables instead of assuming them equal.
EPS = 1e-8

#: A pinned V3 draw with an inflated ``eps``, used for the ``eps`` rows.  At the default
#: ``eps = 1e-8`` the epsilon term is ~1e-4 of ``sqrt(vhat)`` even at ``c = 256``, so
#: getting its width factor wrong changes nothing measurable and a control would not
#: fire.  ``eps = 1e-3`` puts it on the order of ``sqrt(vhat)`` (measured grad RMS
#: ~5e-5 at ``c = 512``), which is precisely the regime the scaling exists to avoid.
EPS_PROBE = 1e-3

#: A pinned V3 draw with ``wd`` at the top of ``WD_RANGE``, appended to the sampled
#: draws so the *weight-decay* path is always exercised.  Needed because the decay
#: term is ``eta*lambda*p`` while the adaptive term is ``O(eta)`` per entry: at a
#: typical sampled ``lambda`` (1e-6..1e-3) a wrong width factor on ``lambda`` moves the
#: parameters by ~1e-7..1e-4 of ``lr``, i.e. *under* the 1e-3 gate, so a decay bug would
#: pass unnoticed.  At ``lambda = 1e-1`` the same bug shows up at 3e-2..2e-1 of ``lr``.
WD_PROBE_HP = ZooHP(lr=1e-3, init_mult=1.0, weight_decay=1e-1, dropout=0.0,
                    train_frac=1.0, epochs=10)


def eps_table(width, base_width, mup_arm, eps=EPS):
    """``{parameter name: Adam eps}`` — the per-layer epsilon muP requires (trap T7).

    Stated here from :func:`expected_infshape`, i.e. from the same independently written
    per-layer table V1 asserts, *not* read off the trainer: ``eps`` is divided by ``wm``
    exactly on the parameters with an infinite dimension, whose gradient is
    ``Theta(1/c)``.  ``fc.bias`` is finite (``ninf 0``, gradient ``Theta(1)``) and keeps
    the base ``eps``.  Following Everett et al. (2024) Sec. 4.3; the ``mup`` package does
    not scale ``eps`` itself, which is why the reference has to be told.
    """
    out = {}
    for name, _ in ZooCNN(width=width).named_parameters():
        ninf, wm = expected_infshape(name, width, base_width)
        out[name] = eps / wm if (mup_arm and ninf >= 1) else eps
    return out


def build_reference(width, hp, base_width, mup_arm, device, seed, eps=EPS):
    """A single-model reference: ``ZooCNN`` + ``set_base_shapes`` + ``MuAdamW``.

    Initialized from our trainer's parameters so that V3 isolates the *optimizer* path,
    which is where trap T4 lives; the init itself is covered by V1 and by the muP init
    table.

    Adam's ``eps`` is passed per param group from :func:`eps_table`.  ``MuAdam`` copies
    every key other than ``params`` onto the groups it splits out, so the per-group
    ``eps`` survives its ``ninf``-based regrouping and reaches ``torch.optim.AdamW``.
    Nothing else needs compensating: our trainer keeps ``MuReadout``'s division in the
    forward pass, so both sides optimize the same variable.  (Storing
    ``W_eff = W_ref/wm`` instead would give ``g_ref = g_eff/wm`` and hence
    ``upd_ref = m_eff/(sqrt(v_eff) + wm*eps)``, i.e. yet another width factor on ``eps``,
    stacked on this one.)
    """
    if mup_arm:
        model = mup_reference_model(width, base_width, device=device, seed=seed)
    else:
        model = ZooCNN(width=width).to(device)
        set_base_shapes(model, ZooCNN(width=width).to(device), rescale_params=False)
    wm = (width / base_width) if mup_arm else 1.0

    table = eps_table(width, base_width, mup_arm, eps)
    groups = {}
    for name, p in model.named_parameters():
        groups.setdefault(table[name], []).append(p)
    opt = MuAdamW([{"params": ps, "eps": e} for e, ps in groups.items()],
                  lr=hp.lr, weight_decay=hp.weight_decay, betas=(0.9, 0.999), eps=eps)
    return model, opt, wm


def _v3_one(width, hp, base_width, mup_arm, steps, batch_size, device, seed,
            break_lr_table=False, break_wd=False, break_eps=False, eps=EPS):
    """Run both implementations and return ``(max, mean)`` absolute parameter gap."""
    trainer = BatchedZooCNNTrainer(
        [hp], width, base_width=base_width if mup_arm else None,
        seed=seed, device=device, batch_size=batch_size, eps=eps,
        # negative control: leave eps fixed, as the mup package does, while the
        # reference scales it (trap T7)
        eps_scaling=not break_eps)
    if break_lr_table:
        # negative control: "forget" the muP scaling on the hidden convs
        trainer.lr_factors["convs.1.weight"] = 1.0
        trainer.lr_factors["convs.2.weight"] = 1.0
    if break_wd:
        # negative control: scale lambda by wm *and* keep the base lr in the decay
        # term, i.e. double-count MuAdam's wd factor instead of cancelling it
        trainer.wd = trainer.wd * trainer.wm
    model, opt, wm = build_reference(width, hp, base_width, mup_arm, device, seed,
                                     eps=eps)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(trainer.params[name][0])

    g = torch.Generator().manual_seed(1234 + width)
    for _ in range(steps):
        x = torch.rand(batch_size, 1, 32, 32, generator=g).to(device)
        y = torch.randint(0, 10, (batch_size,), generator=g).to(device)
        trainer.step(x.unsqueeze(0), y.unsqueeze(0))
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    total, count, mx = 0.0, 0, 0.0
    for name, p in model.named_parameters():
        d = (trainer.params[name][0] - p.detach()).abs()
        mx = max(mx, d.max().item())
        total += d.sum().item()
        count += d.numel()
    return mx, total / count


def check_v3(widths, hps, steps=5, batch_size=8, base_width=BASE_WIDTH, tol=1e-3,
             device="cpu", seed=0):
    """Compare against ``mup.MuAdamW``, gating on the *mean* gap relative to ``lr``.

    The gate is ``mean |dp| < tol * lr`` rather than a bound on ``max |dp|``, because
    ``max`` is not a usable discriminator here.  Adam divides by ``sqrt(v) + eps``, so on
    an entry whose gradient happens to sit near ``eps`` — a nearly dead channel — a
    float32-level difference between our ``vmap``'d convolution and ``nn.Conv2d`` flips
    the update by ``O(1)`` and the parameter by ``O(lr)``.  That happens in the **SP**
    column too, where there are no muP factors at all, so it cannot indicate a muP bug.

    A genuinely wrong width factor, by contrast, perturbs *every* entry of the affected
    tensor by ``O(lr)``, which the mean catches with orders to spare.  The
    ``[control]`` row proves that: it drops the hidden convs' ``1/wm`` factor and must
    fail loudly.
    """
    print("=" * 78)
    print("V3 — batched vmap trainer == mup.MuAdamW reference (N=1)")
    print("=" * 78)
    all_ok = True
    # dropout off: the reference has none, and V3 tests the optimizer path. (Dropout is
    # covered by the base-width identity check, which reproduces it exactly.)
    hps = [ZooHP(lr=h.lr, init_mult=h.init_mult, weight_decay=h.weight_decay,
                 dropout=0.0, train_frac=h.train_frac, epochs=h.epochs) for h in hps]
    print(f"  {'arm':<9} {'width':>6} {'draw':>4} {'lr':>10} {'wd':>10} "
          f"{'mean|dp|/lr':>12} {'max|dp|/lr':>11}")
    for mup_arm in (True, False):
        for width in widths:
            for d, hp in enumerate(hps):
                mx, mean = _v3_one(width, hp, base_width, mup_arm, steps, batch_size,
                                   device, seed)
                good = mean < tol * hp.lr
                all_ok &= good
                print(f"  {'muP' if mup_arm else 'SP':<9} {width:>6} {d:>4} "
                      f"{hp.lr:>10.2e} {hp.weight_decay:>10.2e} "
                      f"{mean / hp.lr:>12.1e} {mx / hp.lr:>11.1e}  {ok(good)}")

    # the eps table, at an inflated eps where it actually moves the update (EPS_PROBE)
    width = widths[-1]
    mx, mean = _v3_one(width, hps[0], base_width, True, steps, batch_size, device, seed,
                       eps=EPS_PROBE)
    good = mean < tol * hps[0].lr
    all_ok &= good
    print(f"  {'[eps]':<9} {width:>6} {0:>4} {hps[0].lr:>10.2e} "
          f"{hps[0].weight_decay:>10.2e} {mean / hps[0].lr:>12.1e} "
          f"{mx / hps[0].lr:>11.1e}  {ok(good)} (eps={EPS_PROBE:g}, scaled both sides)")

    # negative controls: the same comparison with a deliberately wrong lr table, with
    # MuAdam's wd factor double-counted instead of cancelled, and with eps left fixed
    for label, hp, kw, why in (
            ("[ctl-lr]", hps[0], {"break_lr_table": True}, "wrong lr table must fail"),
            ("[ctl-wd]", WD_PROBE_HP, {"break_wd": True}, "wrong wd scaling must fail"),
            ("[ctl-eps]", hps[0], {"break_eps": True, "eps": EPS_PROBE},
             f"unscaled eps={EPS_PROBE:g} must fail")):
        mx, mean = _v3_one(width, hp, base_width, True, steps, batch_size, device, seed,
                           **kw)
        caught = mean >= tol * hp.lr
        all_ok &= caught
        print(f"  {label:<9} {width:>6} {0:>4} {hp.lr:>10.2e} "
              f"{hp.weight_decay:>10.2e} {mean / hp.lr:>12.1e} {mx / hp.lr:>11.1e}  "
              f"{ok(caught)} ({why})")

    print(f"\nV3: {ok(all_ok)}  (gate: mean |dp| < {tol:g} * lr)")
    return all_ok


# ---------------------------------------------------------------------------
# V4 — export round-trip
# ---------------------------------------------------------------------------

def check_v4(widths, n_models=4, base_width=BASE_WIDTH, steps=3, tol=1e-5,
             device="cpu", seed=0, tmp_dir=None):
    print("=" * 78)
    print("V4 — exported .pth reproduces the trainer's forward (trap T3)")
    print("=" * 78)
    import tempfile
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp(prefix="check_mup_cnn_"))
    all_ok = True
    hps = sample_hps(n_models, seed=7)
    print(f"  {'arm':<5} {'width':>6} {'max |f_trainer - f_pth|':>26}")
    for mup_arm in (True, False):
        for width in widths:
            trainer = BatchedZooCNNTrainer(
                hps, width, base_width=base_width if mup_arm else None,
                seed=seed, device=device, batch_size=8)
            g = torch.Generator().manual_seed(99)
            for _ in range(steps):
                x = torch.rand(n_models, 8, 1, 32, 32, generator=g).to(device)
                y = torch.randint(0, 10, (n_models, 8), generator=g).to(device)
                trainer.step(x, y)

            probe = torch.rand(5, 1, 32, 32, generator=g).to(device)
            with torch.no_grad():
                f_trainer = trainer.forward(probe, training=False, shared_x=True)

            max_err = 0.0
            for i, sd in enumerate(trainer.export_state_dicts()):
                path = tmp_dir / f"m{i}.pth"
                torch.save(sd, path)
                w, b = state_dict_to_wb(torch.load(path, weights_only=True))
                f_pth = zoo_cnn_forward([t.to(device) for t in w],
                                        [t.to(device) for t in b], probe)
                max_err = max(max_err, (f_trainer[i] - f_pth).abs().max().item())
            good = max_err < tol
            all_ok &= good
            print(f"  {'muP' if mup_arm else 'SP':<5} {width:>6} {max_err:>26.2e}  "
                  f"{ok(good)}")
    print(f"\nV4: {ok(all_ok)}  (tol {tol:g})")
    return all_ok


# ---------------------------------------------------------------------------

def check_base_width_identity(width=BASE_WIDTH, n_models=4, steps=5, device="cpu"):
    """muP at ``base_width == width`` must be bit-identical to SP.

    Not one of the plan's numbered checks, but it is the property the whole
    size-generalization comparison rests on: at the train width the muP arm and the SP
    arm must be the *same* zoo, so that the OOD comparison differs only OOD.

    Dropout is sampled inside ``vmap(..., randomness="different")``, which draws from
    the **global** RNG, so the two trainers only see the same masks if the global seed
    is reset before each corresponding step.  That is done here explicitly rather than
    dodged by forcing ``dropout = 0``: dropout is part of the HP sweep, and the run must
    be identical *with* it.
    """
    print("=" * 78)
    print(f"Base-width identity — muP(base={width}) == SP at width {width}")
    print("=" * 78)
    hps = sample_hps(n_models, seed=3)
    print(f"  dropout draws: {[h.dropout for h in hps]}")
    a = BatchedZooCNNTrainer(hps, width, base_width=width, seed=0, device=device,
                             batch_size=8)
    b = BatchedZooCNNTrainer(hps, width, base_width=None, seed=0, device=device,
                             batch_size=8)
    g = torch.Generator().manual_seed(5)
    for i in range(steps):
        x = torch.rand(n_models, 8, 1, 32, 32, generator=g).to(device)
        y = torch.randint(0, 10, (n_models, 8), generator=g).to(device)
        torch.manual_seed(1000 + i)
        a.step(x, y)
        torch.manual_seed(1000 + i)
        b.step(x, y)
    diff = max((a.params[n] - b.params[n]).abs().max().item() for n in a.params)
    print(f"  max param diff after {steps} steps: {diff:.2e}  {ok(diff == 0.0)}")
    return diff == 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None,
                   choices=["v1", "v2", "v2ref", "v3", "v4", "identity"])
    p.add_argument("--coord-check", action="store_true",
                   help="also run V2-ref (mup.coord_check; slow, writes plots)")
    p.add_argument("--widths", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--coord-widths", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--base-width", type=int, default=BASE_WIDTH)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--n-draws", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="V3 gate: mean |dp| < tol * lr (see check_v3)")
    p.add_argument("--device", default="cpu",
                   help="cpu keeps V3/V4 exactly reproducible; V2 uses cuda if present")
    args = p.parse_args()

    run = set(args.only) if args.only else {"v1", "v2", "v3", "v4", "identity"}
    if args.coord_check:
        run |= {"v2", "v2ref"}

    results = {}
    if "v1" in run:
        results["V1"] = check_v1(args.coord_widths, args.base_width)
    if "identity" in run:
        results["identity"] = check_base_width_identity(args.base_width,
                                                        device=args.device)
    if "v2" in run:
        results["V2-ours"] = check_v2_ours(args.coord_widths, args.base_width,
                                           device=args.device)
    if "v3" in run:
        hps = sample_hps(args.n_draws, seed=11) + [WD_PROBE_HP]
        results["V3"] = check_v3(args.widths, hps, steps=args.steps,
                                 base_width=args.base_width, tol=args.tol,
                                 device=args.device)
    if "v4" in run:
        results["V4"] = check_v4(args.widths, base_width=args.base_width,
                                 device=args.device)
    if "v2ref" in run:
        check_v2_ref(args.coord_widths, args.base_width, device=args.device)

    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"{k:<10} {ok(v)}")
    print("=" * 78)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

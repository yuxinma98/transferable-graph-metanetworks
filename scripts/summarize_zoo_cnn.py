"""Print the zoo CNN's architecture together with the per-parameter muP/SP scalings.

Everything is read off the shipped implementation rather than restated:

* shapes and the forward trace come from a real :class:`src.data.zoo_cnn.ZooCNN`
  forward pass;
* ``ninf`` and ``width_mult`` are inferred the way ``mup.set_base_shapes`` does it —
  compare the model's shapes against a base-width and a delta-width instance, take the
  dimensions that differ as infinite, and read ``width_mult`` off the **last** one
  (``--check-mup`` cross-checks that against ``p.infshape`` if ``mup`` is installed);
* the lr factors are the trainer's own ``lr_factors`` table, not a re-derivation;
* the init column's formula is checked against the standard deviation actually realized
  by :class:`src.data.train_zoo_cnns.BatchedZooCNNTrainer`.

So a change to ``train_zoo_cnns.py`` that this file does not know about shows up as a
FAIL in the init check or as a changed lr column, not as silently stale documentation.

Reference for the formulas: ``src/data/train_zoo_cnns.py``, cross-checked against the
``mup`` library by ``scripts/check_mup_cnn.py``.

Usage
-----
    python scripts/summarize_zoo_cnn.py                       # w16, muP(base 16) + SP
    python scripts/summarize_zoo_cnn.py --width 128           # the OOD end of Stage 3
    python scripts/summarize_zoo_cnn.py --lr 3e-3 --weight-decay 1e-2
    python scripts/summarize_zoo_cnn.py --check-mup           # also verify vs `mup`
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.train_zoo_cnns import BatchedZooCNNTrainer, ZooHP  # noqa: E402
from data.zoo_cnn import (  # noqa: E402
    IN_CHANNELS, KERNEL_SIZE, NUM_CLASSES, NUM_CONV_LAYERS, PADDING, STRIDE, ZooCNN,
    layer_layout,
)

BASE_WIDTH = 16
INPUT_HW = 32


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def architecture_trace(width, device="cpu"):
    """``[(label, output shape, #params)]`` from a real forward pass."""
    model = ZooCNN(width=width).to(device)
    rows, handles = [], []

    def hook(label, module):
        def fn(_m, _inp, out):
            n = sum(p.numel() for p in module.parameters())
            rows.append((label, tuple(out.shape), n))
        return fn

    for i, conv in enumerate(model.convs):
        handles.append(conv.register_forward_hook(hook(
            f"conv{i} (3x3, s{STRIDE}, p{PADDING})", conv)))
    handles.append(model.fc.register_forward_hook(hook("fc (dense)", model.fc)))

    x = torch.zeros(1, IN_CHANNELS, INPUT_HW, INPUT_HW, device=device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return model, rows


def print_architecture(width, rows):
    print("=" * 96)
    print(f"zoo CNN architecture — width c = {width}")
    print("=" * 96)
    print(f"  input                              [1, {IN_CHANNELS}, {INPUT_HW}, "
          f"{INPUT_HW}]")
    for i, (label, shape, n) in enumerate(rows):
        act = "ReLU (+ dropout when training)" if i < len(rows) - 1 else "-"
        if label.startswith("fc"):
            print(f"  {'global average pool':<34} [1, {width}]")
        print(f"  {label:<34} {str(list(shape)):<22} {n:>8,} params   {act}")
    total = sum(n for _, _, n in rows)
    print(f"  {'':<34} {'':<22} {total:>8,} total")
    print(f"\n  graph layer_layout        {layer_layout(width)}"
          f"   ({1 + NUM_CONV_LAYERS * width + NUM_CLASSES} nodes)")
    print(f"  node types (node2type)    {_num_node_types(width)}"
          f"  = 1 input + {NUM_CONV_LAYERS} hidden layers + {NUM_CLASSES} outputs, "
          f"width-independent")
    print("  edge features             one edge per input channel, carrying the whole "
          "3x3 kernel in R^9")


def _num_node_types(width):
    """Distinct ``node2type`` values, from upstream's ``get_node_types``."""
    try:
        from data.cnn_zoo_dataset import get_node_types  # noqa: PLC0415
        return int(get_node_types(layer_layout(width)).max().item()) + 1
    except Exception as exc:                                    # pragma: no cover
        return f"<unavailable: {type(exc).__name__}>"


# ---------------------------------------------------------------------------
# muP shape inference (the way `mup` does it, without importing it)
# ---------------------------------------------------------------------------

def infer_infshapes(width, base_width):
    """``{param: (ninf, width_mult, infinite_dims)}`` for every parameter.

    A dimension is *infinite* if it differs between the base-width and the delta-width
    reference models; ``width_mult`` is the ratio on the **last** infinite dimension.
    That last-dim rule is why the kernel size must be fixed across widths (trap T1):
    for a ``[c_out, c_in, 3, 3]`` conv weight it makes ``width_mult = c_in / c_in_base``.
    """
    shapes = {n: tuple(p.shape) for n, p in ZooCNN(width=width).named_parameters()}
    base = {n: tuple(p.shape) for n, p in ZooCNN(width=base_width).named_parameters()}
    delta = {n: tuple(p.shape) for n, p in ZooCNN(width=base_width * 2).named_parameters()}

    out = {}
    for name, shape in shapes.items():
        inf_dims = [d for d in range(len(shape)) if base[name][d] != delta[name][d]]
        wm = shape[inf_dims[-1]] / base[name][inf_dims[-1]] if inf_dims else 1.0
        out[name] = (len(inf_dims), wm, inf_dims)
    return out


def role_of(name, ninf):
    if name.endswith("bias"):
        return "fixed" if ninf == 0 else "vector"
    if name == "convs.0.weight":
        return "input"
    if name == "fc.weight":
        return "readout"
    return "hidden"


def check_against_mup(width, base_width, inferred):
    """Cross-check the inferred table against ``mup.set_base_shapes``."""
    try:
        from mup import MuReadout, set_base_shapes  # noqa: PLC0415
    except ImportError:
        print("  `mup` not installed — skipping cross-check")
        return None
    model = ZooCNN(width=width, readout_cls=MuReadout)
    set_base_shapes(model, ZooCNN(width=base_width, readout_cls=MuReadout),
                    delta=ZooCNN(width=base_width * 2, readout_cls=MuReadout))
    all_ok = True
    for name, p in model.named_parameters():
        ninf, wm, _ = inferred[name]
        good = p.infshape.ninf() == ninf and abs(p.infshape.width_mult() - wm) < 1e-12
        all_ok &= good
        if not good:
            print(f"  MISMATCH {name}: ours ({ninf}, {wm:g}) vs mup "
                  f"({p.infshape.ninf()}, {p.infshape.width_mult():g})")
    print(f"  cross-check vs mup.set_base_shapes: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Per-parameter init / lr / weight decay
# ---------------------------------------------------------------------------

def analytic_init(name, shape, alpha, wm, mup):
    """``(formula, std)`` of the initial distribution, **as stored** by the trainer.

    Mirrors :meth:`BatchedZooCNNTrainer._init_params`: torch's default
    Kaiming-uniform bound ``1/sqrt(fan_in)`` (fan_in ``= prod(shape[1:])``, so a conv
    counts ``c_in * kh * kw``), times the sampled ``init_mult``, times ``wm**0.5`` on
    the readout weight (``MuReadout._rescale_parameters``).  Biases are zero.

    The readout's stored std is therefore width-*independent*
    (``a/sqrt(c) * sqrt(c/c0) = a/sqrt(c0)``); the muP-relevant
    :math:`\\Theta(c^{-1})` lives in the effective weight ``W/wm`` — see
    :func:`effective_init_std`.
    """
    if name.endswith("bias"):
        return "0", 0.0
    fan_in = 1
    for d in shape[1:]:
        fan_in *= d
    bound = alpha / math.sqrt(fan_in)
    formula = f"a*U(+-1/sqrt({fan_in}))"
    if mup and name == "fc.weight":
        bound *= wm ** 0.5
        formula += " * wm^+1/2"
    return formula, bound / math.sqrt(3.0)


def effective_init_std(name, shape, alpha, wm, mup):
    """Init std of the weight the *function* uses, i.e. after ``readout_mult``."""
    std = analytic_init(name, shape, alpha, wm, mup)[1]
    return std / wm if (mup and name == "fc.weight") else std


def summarize_arm(width, base_width, hp, mup, n_models, device, tol=0.05):
    """Print the parameter table for one arm and return whether the init check passed."""
    arm = f"muP (base c0 = {base_width})" if mup else "SP"
    wm = (width / base_width) if mup else 1.0
    trainer = BatchedZooCNNTrainer([hp] * n_models, width,
                                   base_width=base_width if mup else None,
                                   seed=0, device=device, batch_size=8)
    inferred = infer_infshapes(width, base_width)

    print()
    print("=" * 96)
    print(f"{arm}   —   c = {width}, wm = c/c0 = {wm:g}   |   "
          f"lr eta = {hp.lr:g}, init_mult a = {hp.init_mult:g}, wd lambda = "
          f"{hp.weight_decay:g}")
    print("=" * 96)
    print(f"  forward: fc sees (GAP output) * readout_mult, "
          f"readout_mult = {trainer.template.readout_mult:g}"
          f"{'  (= 1/wm, muP)' if mup else '  (SP: no multiplier)'}")
    print()
    print(f"  {'parameter':<15} {'shape':<18} {'#par':>8} {'ninf':>4} {'role':<8} "
          f"{'init':<30} {'std':>9} {'lr x':>6} {'lr':>9} {'wd/step':>9} {'eps':>9}")
    print("  " + "-" * 106)

    all_ok = True
    for name, p in trainer.template.named_parameters():
        shape = tuple(p.shape)
        ninf, param_wm, _ = inferred[name]
        role = role_of(name, ninf)
        formula, std = analytic_init(name, shape, hp.init_mult, param_wm, mup)
        r = trainer.lr_factors[name]
        print(f"  {name:<15} {str(list(shape)):<18} {p.numel():>8,} {ninf:>4} "
              f"{role:<8} {formula:<30} {std:>9.3e} "
              f"{_fmt_factor(r):>6} {hp.lr * r:>9.2e} "
              f"{hp.lr * hp.weight_decay:>9.2e} {trainer.eps_values[name]:>9.2e}")

        # the init column is a claim about the shipped trainer — verify it
        t = trainer.params[name].detach()
        got = float(t.std()) if t.numel() > 1 else float(t.abs().max())
        ok = (got == 0.0) if std == 0.0 else abs(got - std) <= tol * std
        all_ok &= ok
        if not ok:
            print(f"  {'':<15} ^ init MISMATCH: formula {std:.4e} vs realized "
                  f"{got:.4e}")

    print("  " + "-" * 106)
    print(f"  init check vs BatchedZooCNNTrainer ({n_models} models): "
          f"{'PASS' if all_ok else 'FAIL'}  (rel tol {tol:g})")
    print(f"\n  update rule (Adam betas={trainer.betas}, base eps={trainer.eps:g}, "
          "AdamW-decoupled decay):")
    print("      p <- p  -  eta*(lr x) * mhat/(sqrt(vhat)+eps)  -  eta*lambda * p")
    print("    * the decay term uses the BASE lr eta for every parameter, and lambda is "
          "never")
    print("      scaled by width: mup's MuAdam does lr/=wm AND wd*=wm, which cancel "
          "under")
    print("      decoupled decay, so a sampled lambda means the same thing at every "
          "width.")
    print("    * biases are decayed too — nothing is excluded from weight decay.")
    if mup:
        eff = effective_init_std(
            "fc.weight", (NUM_CLASSES, width), hp.init_mult, wm, True)
        print("    * the lr x column is literally MuAdam's rule: divide by wm iff "
              "ninf == 2.")
        print("      The readout keeps the base lr; its width correction is the forward")
        print(f"      multiplier above, so the *effective* readout is W/wm with init "
              f"std {eff:.3e}")
        print("      (Theta(1/c), variance Theta(1/c^2)) while the stored std is "
              "width-independent.")
        print("      export_state_dicts() / spectral_norm_components() fold the "
              "multiplier in (trap T3).")
        print("    * the eps column is base eps / wm on every parameter with an "
              "infinite dimension")
        print("      (their gradient is Theta(1/c)); fc.bias is finite (ninf 0, "
              "gradient Theta(1)) and")
        print("      keeps the base eps. Everett et al. 2024 Sec 4.3; `mup` itself does "
              "not scale eps.")
    return all_ok


def _fmt_factor(r):
    if abs(r - 1.0) < 1e-12:
        return "1"
    return f"1/{1.0 / r:g}"


# ---------------------------------------------------------------------------
# What the scalings do as width grows
# ---------------------------------------------------------------------------

def print_width_scaling(widths, base_width, hp, device):
    """The three distinct weight roles' lr and init std across widths, muP vs SP."""
    print()
    print("=" * 96)
    print(f"scaling across widths (base c0 = {base_width}, eta = {hp.lr:g}, "
          f"a = {hp.init_mult:g})")
    print("=" * 96)
    names = ["convs.0.weight", "convs.1.weight", "fc.weight"]
    for mup in (True, False):
        print(f"\n  {'muP' if mup else 'SP'}")
        print(f"    {'c':>6} {'wm':>7} " + " ".join(
            f"{n.replace('.weight', '') + ' lr':>14}" for n in names) + "   |  "
            + " ".join(
            f"{n.replace('.weight', '') + ' eff std':>18}" for n in names))
        for w in widths:
            wm = (w / base_width) if mup else 1.0
            inferred = infer_infshapes(w, base_width)
            trainer = BatchedZooCNNTrainer([hp], w,
                                           base_width=base_width if mup else None,
                                           seed=0, device=device, batch_size=8)
            lrs, stds = [], []
            for n in names:
                shape = tuple(dict(ZooCNN(width=w).named_parameters())[n].shape)
                lrs.append(hp.lr * trainer.lr_factors[n])
                stds.append(effective_init_std(n, shape, hp.init_mult,
                                              inferred[n][1], mup))
            print(f"    {w:>6} {wm:>7.2f} " + " ".join(f"{v:>14.3e}" for v in lrs)
                  + "   |  " + " ".join(f"{v:>18.3e}" for v in stds))
    print("\n  std columns are *effective* (i.e. fc includes the 1/wm forward "
          "multiplier); the")
    print("  stored fc.weight std is a/sqrt(3*c0), width-independent.")
    print("\n  muP: only the hidden convs see a 1/wm lr; the readout's effective init "
          "std falls")
    print("       as 1/c (variance 1/c^2) while the hidden convs' falls as 1/sqrt(c)")
    print("       (variance 1/c) — that is the Theta(1)-activation condition.")
    print("  SP:  identical lrs at every width and no readout correction, so the "
          "logit scale")
    print("       and the update size both drift with c. This is the arm muP is the "
          "control for.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=BASE_WIDTH)
    p.add_argument("--base-width", type=int, default=BASE_WIDTH)
    p.add_argument("--lr", type=float, default=1e-3, help="base lr eta of the HP draw")
    p.add_argument("--init-mult", type=float, default=1.0, help="init multiplier a")
    p.add_argument("--weight-decay", type=float, default=1e-3, help="lambda")
    p.add_argument("--widths", type=int, nargs="+",
                   default=[16, 32, 48, 64, 96, 128],
                   help="widths for the scaling table (Stage 3 zoo widths)")
    p.add_argument("--n-models", type=int, default=64,
                   help="cohort size used for the empirical init check")
    p.add_argument("--arm", choices=["mup", "sp", "both"], default="both")
    p.add_argument("--check-mup", action="store_true",
                   help="cross-check the inferred infshapes against the mup package")
    p.add_argument("--no-scaling-table", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    hp = ZooHP(lr=args.lr, init_mult=args.init_mult, weight_decay=args.weight_decay,
               dropout=0.0, train_frac=1.0, epochs=10)

    _, rows = architecture_trace(args.width, args.device)
    print_architecture(args.width, rows)

    if args.check_mup:
        print()
        check_against_mup(args.width, args.base_width,
                          infer_infshapes(args.width, args.base_width))

    arms = ("mup", "sp") if args.arm == "both" else (args.arm,)
    all_ok = True
    for arm in arms:
        all_ok &= summarize_arm(args.width, args.base_width, hp, arm == "mup",
                                args.n_models, args.device)

    if not args.no_scaling_table:
        print_width_scaling(args.widths, args.base_width, hp, args.device)

    if args.width == args.base_width and args.arm == "both":
        print("\nNote: c == c0, so every muP factor is exactly 1.0 and the two tables "
              "above are\nthe same parameterization — that is the base-width identity "
              "the size-generalization\ncomparison rests on (asserted bit-exactly by "
              "scripts/check_mup_cnn.py).")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Wrapper to run ScaleGMN's inr_classification.py with our config.

ScaleGMN uses relative imports (from src.data import ..., from src.scalegmn import ...)
that assume CWD is the ScaleGMN repo root. This script sets cwd to src/scalegmn/
and resolves data paths to absolute before delegating — no patching of upstream code needed.

Usage
-----
python scripts/train_scalegmn.py --conf configs/mnist_cls/scalegmn_reproduce_sp.yml
python scripts/train_scalegmn.py --conf configs/mnist_cls/scalegmn_reproduce_sp.yml --wandb True
python scripts/train_scalegmn.py --conf configs/mnist_cls/scalegmn_reproduce_sp.yml --debug True

All extra arguments are forwarded directly to inr_classification.py.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"


# `inr_classification.py` builds its own ScaleGMN, so the reciprocal backward edge feature
# (see src/models/bidir_reciprocal.py) cannot be installed by this wrapper directly. Instead
# of running the script as __main__, run this bootstrap, which rebinds the name
# `src.scalegmn.models.ScaleGMN` to a factory that installs the patch after construction --
# `inr_classification.py` does `from src.scalegmn.models import ScaleGMN` at import time, so
# it picks up the rebound name -- and then executes the script unchanged via runpy. A no-op
# for `direction: forward` and `symmetry: permutation` configs.
BOOTSTRAP = """
import runpy, sys
sys.argv = ['inr_classification.py'] + sys.argv[1:]
import src.scalegmn.models as _M
from models.bidir_reciprocal import install_bidir_reciprocal
_orig_ScaleGMN = _M.ScaleGMN

def _ScaleGMN(conf, *a, **kw):
    net = _orig_ScaleGMN(conf, *a, **kw)
    if install_bidir_reciprocal(net, conf):
        print('Installed reciprocal backward edge features (bidirectional scale equivariance).')
    return net

_M.ScaleGMN = _ScaleGMN
runpy.run_path('inr_classification.py', run_name='__main__')
"""


def resolve_data_paths(conf: dict, project_root: Path) -> dict:
    """Make dataset_path and split_path absolute so they work from any CWD."""
    data = conf.get("data", {})
    for key in ("dataset_path", "split_path"):
        if key in data:
            expanded = os.path.expandvars(data[key])
            if not Path(expanded).is_absolute():
                expanded = str((project_root / expanded).resolve())
            data[key] = expanded
    return conf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ScaleGMN inr_classification.py from the project root",
        add_help=False,  # pass --help through to inr_classification.py
    )
    parser.add_argument("--conf", required=True, help="Path to YAML config (relative to project root)")
    parser.add_argument(
        "--bidir-reciprocal", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
        help="Honour the config's `reciprocal` flag by feeding 1/W to the backward layers "
             "(required for scale equivariance in the bidirectional direction). Set False to "
             "reproduce runs from before the fix.",
    )
    args, extra = parser.parse_known_args()

    # Load and patch the config
    conf_path = (PROJECT_ROOT / args.conf).resolve()
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    conf = resolve_data_paths(conf, PROJECT_ROOT)

    # Write patched config to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", dir=SCALEGMN_ROOT, delete=False
    ) as tmp:
        yaml.dump(conf, tmp, default_flow_style=False)
        tmp_conf_path = tmp.name

    try:
        entry = "inr_classification.py" if not args.bidir_reciprocal else "-c"
        cmd = [sys.executable, entry]
        if args.bidir_reciprocal:
            cmd.append(BOOTSTRAP)
        cmd += ["--conf", tmp_conf_path] + extra
        shown = cmd if not args.bidir_reciprocal else [
            sys.executable, "-c", "<bidir-reciprocal bootstrap>", "--conf", tmp_conf_path, *extra]
        print(f"Running from {SCALEGMN_ROOT}:\n  {' '.join(shown)}\n")
        result = subprocess.run(cmd, cwd=SCALEGMN_ROOT)
        sys.exit(result.returncode)
    finally:
        os.unlink(tmp_conf_path)


if __name__ == "__main__":
    main()

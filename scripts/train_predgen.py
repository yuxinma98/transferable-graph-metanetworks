"""
Wrapper to run ScaleGMN's predicting_generalization.py with our configs.

Same reason as ``scripts/train_scalegmn.py``: ScaleGMN's imports assume CWD is
``src/scalegmn/``.  This script expands ``$ANYDIM_DATA_ROOT`` and resolves every
data path to an absolute one, writes a temp config next to the upstream script and
delegates via ``subprocess.run(..., cwd=src/scalegmn/)`` — no patching of the git
subtree.

Used for Experiment 1 (reproduce): accuracy prediction on the small CNN zoo at its
native width w16.  Size generalization across widths uses
``scripts/train_sizegen_predgen.py`` instead.

Usage
-----
python scripts/train_predgen.py --conf configs/cifar10_predgen/scalegmn_predgen_sp.yml --wandb True
python scripts/train_predgen.py --conf configs/cifar10_predgen/gmn_predgen_sp.yml --debug True

All extra arguments are forwarded to predicting_generalization.py (e.g.
``--optimization.optimizer_args.lr 5e-4``, ``--gpu_ids 0``, ``--wandb_args.name foo``).
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

#: every data key that names a file or directory on disk
PATH_KEYS = ("dataset_path", "data_path", "metrics_path", "layout_path",
             "idcs_file", "split_path")


def resolve_data_paths(conf: dict, project_root: Path) -> dict:
    """Expand ``$ANYDIM_DATA_ROOT`` and make every data path absolute."""
    data = conf.get("data", {})
    for key in PATH_KEYS:
        if data.get(key) in (None, "None"):
            continue
        expanded = os.path.expandvars(data[key])
        if "$" in expanded:
            raise ValueError(
                f"data.{key} = {data[key]!r} still contains an unexpanded variable "
                f"after expansion ({expanded!r}). Did you source .env?")
        if not Path(expanded).is_absolute():
            expanded = str((project_root / expanded).resolve())
        data[key] = expanded
    return conf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ScaleGMN predicting_generalization.py from the project root",
        add_help=False,  # pass --help through
    )
    parser.add_argument("--conf", required=True,
                        help="Path to YAML config (relative to project root)")
    args, extra = parser.parse_known_args()

    conf_path = (PROJECT_ROOT / args.conf).resolve()
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    conf = resolve_data_paths(conf, PROJECT_ROOT)

    missing = [conf["data"][k] for k in PATH_KEYS
               if conf["data"].get(k) not in (None, "None")
               and not Path(conf["data"][k]).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing zoo files:\n  " + "\n  ".join(missing)
            + "\nRun: bash scripts/download_cnn_zoo.sh")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", dir=SCALEGMN_ROOT, delete=False
    ) as tmp:
        yaml.dump(conf, tmp, default_flow_style=False)
        tmp_conf_path = tmp.name

    try:
        cmd = [sys.executable, "predicting_generalization.py",
               "--conf", tmp_conf_path] + extra
        print(f"Running from {SCALEGMN_ROOT}:\n  {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=SCALEGMN_ROOT)
        sys.exit(result.returncode)
    finally:
        os.unlink(tmp_conf_path)


if __name__ == "__main__":
    main()

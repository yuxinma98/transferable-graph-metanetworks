#!/bin/bash
# One-off migration: rename the `relu` parameterization marker to `sp` on disk.
#
# The code/config rename (w{N}_relu -> w{N}_sp) happened in the naming cleanup;
# this brings the data directories under $ANYDIM_DATA_ROOT in line with it.
#
# Two steps, both required:
#   1. mv  w{N}_relu[...]  ->  w{N}_sp[...]           (also _d3, _dup variants)
#   2. rewrite the ABSOLUTE paths stored inside every *_splits.json — the split
#      JSONs embed resolved paths and LabeledINRDataset.get_path() uses them
#      verbatim, so renaming the directories alone breaks every dataloader.
#
# Dry-run by default. Pass --apply to actually make the changes.
#
# Usage:
#   bash scripts/migrate_relu_to_sp.sh            # preview
#   bash scripts/migrate_relu_to_sp.sh --apply    # execute
set -euo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"
DATA_ROOT="${ANYDIM_DATA_ROOT:-$PROJECT_ROOT/data}"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    echo "Set ANYDIM_DATA_ROOT (or create .env) and re-run." >&2
    exit 1
fi

echo "Data root: $DATA_ROOT"
[ $APPLY -eq 0 ] && echo "Mode: DRY RUN (re-run with --apply to execute)" || echo "Mode: APPLY"
echo

# ---------------------------------------------------------------- step 1: dirs
echo "=== Step 1: directory renames ==="
n_dirs=0
n_conflict=0
for prefix in mnist fmnist; do
    inr_root="$DATA_ROOT/${prefix}_inrs"
    [ -d "$inr_root" ] || continue
    for src in "$inr_root"/w*_relu*; do
        [ -d "$src" ] || continue          # no glob match -> literal, skip
        dst="${src/_relu/_sp}"
        if [ -e "$dst" ]; then
            echo "  CONFLICT (skipping): $(basename "$src") -> $(basename "$dst") already exists"
            n_conflict=$((n_conflict + 1))
            continue
        fi
        echo "  $(basename "$src")  ->  $(basename "$dst")"
        [ $APPLY -eq 1 ] && mv "$src" "$dst"
        n_dirs=$((n_dirs + 1))
    done
done
echo "  $n_dirs directory rename(s), $n_conflict conflict(s)."
echo

# --------------------------------------------------- step 2: split JSON paths
# Substitutes only the width-directory component (/w<N>_relu -> /w<N>_sp) so an
# unrelated 'relu' elsewhere in a path is left alone.
echo "=== Step 2: split JSON path rewrites ==="
APPLY=$APPLY DATA_ROOT="$DATA_ROOT" python3 - <<'PY'
import os
import re
from pathlib import Path

apply = os.environ["APPLY"] == "1"
data_root = Path(os.environ["DATA_ROOT"])
pattern = re.compile(r"(/w\d+)_relu")

n_files = n_subs = 0
for prefix in ("mnist", "fmnist"):
    inr_root = data_root / f"{prefix}_inrs"
    if not inr_root.is_dir():
        continue
    for jsn in sorted(inr_root.glob("**/*_splits.json")):
        text = jsn.read_text()
        new_text, n = pattern.subn(r"\1_sp", text)
        if not n:
            continue
        rel = jsn.relative_to(data_root)
        print(f"  {rel}: {n} path(s)")
        if apply:
            jsn.write_text(new_text)
        n_files += 1
        n_subs += n
print(f"  {n_subs} path(s) across {n_files} file(s).")
PY
echo

# ----------------------------------------------------------------- verify
if [ $APPLY -eq 1 ]; then
    echo "=== Verification ==="
    # `|| true`: grep exits 1 when it finds nothing, which is the success case here.
    stale_dirs=$(find "$DATA_ROOT" -maxdepth 2 -type d -name 'w*_relu*' | wc -l | tr -d ' ')
    stale_json=$({ grep -rl '/w[0-9]*_relu' --include='*_splits.json' "$DATA_ROOT" 2>/dev/null || true; } | wc -l | tr -d ' ')
    echo "  stale directories: $stale_dirs"
    echo "  stale split JSONs: $stale_json"
    if [ "$stale_dirs" = "0" ] && [ "$stale_json" = "0" ]; then
        echo "  clean."
    else
        echo "  WARNING: stale references remain (see conflicts above)." >&2
        exit 1
    fi
fi

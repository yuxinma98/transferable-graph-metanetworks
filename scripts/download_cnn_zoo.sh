#!/usr/bin/env bash
# Download the Unterthiner et al. "small CNN zoo" (CIFAR-10 grayscale) plus NFN's
# fixed shuffle order, into $ANYDIM_DATA_ROOT/cifar10_zoo/.
#
# This is the dataset ScaleGMN's predicting_generalization.py consumes (Experiment 1
# of the CNN accuracy-prediction study).  ScaleGMN's own README points at
# storage.cloud.google.com, which is the browser-only interactive endpoint; the
# script-usable one is storage.googleapis.com.
#
# Produces:
#   $ANYDIM_DATA_ROOT/cifar10_zoo/weights.npy        ~ all 30k models x 4970 params, float32
#   $ANYDIM_DATA_ROOT/cifar10_zoo/metrics.csv.gz     per-model HPs + train/test metrics per step
#   $ANYDIM_DATA_ROOT/cifar10_zoo/layout.csv         start_idx/end_idx/shape per variable
#   $ANYDIM_DATA_ROOT/cifar10_zoo/cifar10_split.csv  NFN's shuffle order (idcs_file)
#
# Usage:
#   bash scripts/download_cnn_zoo.sh            # cifar10 (default)
#   bash scripts/download_cnn_zoo.sh svhn       # svhn_cropped instead
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .env ]; then source .env; fi
DATA_ROOT="${ANYDIM_DATA_ROOT:-data}"

ZOO="${1:-cifar10}"
case "$ZOO" in
  cifar10)  ARCHIVE="cifar10.tar.xz";      TARDIR="cifar10";      SPLIT="cifar10_split.csv"; OUTDIR="cifar10_zoo" ;;
  svhn)     ARCHIVE="svhn_cropped.tar.xz"; TARDIR="svhn_cropped"; SPLIT="svhn_split.csv";    OUTDIR="svhn_zoo" ;;
  *) echo "unknown zoo '$ZOO' (expected cifar10 or svhn)" >&2; exit 1 ;;
esac

BASE_URL="https://storage.googleapis.com/gresearch/smallcnnzoo-dataset"
SPLIT_URL="https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits"

DEST="$DATA_ROOT/$OUTDIR"
CACHE="$DATA_ROOT/downloads"
mkdir -p "$DEST" "$CACHE"

echo "zoo=$ZOO  dest=$DEST"

# --- archive ---------------------------------------------------------------
if [ -f "$DEST/weights.npy" ] && [ -f "$DEST/metrics.csv.gz" ] && [ -f "$DEST/layout.csv" ]; then
  echo "[skip] weights.npy / metrics.csv.gz / layout.csv already present"
else
  if [ ! -f "$CACHE/$ARCHIVE" ]; then
    echo "[get ] $BASE_URL/$ARCHIVE"
    wget -q --show-progress --progress=bar:force:noscroll \
        -O "$CACHE/$ARCHIVE.part" "$BASE_URL/$ARCHIVE"
    mv "$CACHE/$ARCHIVE.part" "$CACHE/$ARCHIVE"
  else
    echo "[skip] $CACHE/$ARCHIVE already downloaded"
  fi
  # the tarball contains a top-level <TARDIR>/ directory; strip it so the files
  # land directly in $DEST (which is named *_zoo, not *_).
  echo "[tar ] extracting into $DEST"
  tar -xf "$CACHE/$ARCHIVE" -C "$DEST" --strip-components=1 "$TARDIR"
fi

# --- NFN shuffle order -----------------------------------------------------
if [ -f "$DEST/$SPLIT" ]; then
  echo "[skip] $SPLIT already present"
else
  echo "[get ] $SPLIT_URL/$SPLIT"
  wget -q --show-progress -O "$DEST/$SPLIT" "$SPLIT_URL/$SPLIT"
fi

# --- report ----------------------------------------------------------------
echo
ls -la "$DEST"
echo
python - "$DEST" "$SPLIT" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

dest, split = Path(sys.argv[1]), sys.argv[2]
w = np.load(dest / "weights.npy", mmap_mode="r")
layout = pd.read_csv(dest / "layout.csv")
idcs = pd.read_csv(dest / split, header=None).values.flatten()
print(f"weights.npy   {w.shape} {w.dtype}")
print(f"layout.csv    {len(layout)} variables, total params = {layout['end_idx'].max()}")
print(layout.to_string(index=False))
print(f"{split}  {idcs.shape[0]} indices, max={idcs.max()}")
m = pd.read_csv(dest / "metrics.csv.gz", compression="gzip")
final = m[m["step"] == 86]
print(f"metrics.csv.gz {len(m)} rows, {len(final)} at step 86")
print("activations:", final["config.activation"].value_counts().to_dict())
relu = final[final["config.activation"] == "relu"]
print(f"relu models: {len(relu)}  test_accuracy "
      f"min={relu['test_accuracy'].min():.4f} median={relu['test_accuracy'].median():.4f} "
      f"max={relu['test_accuracy'].max():.4f}")
PY

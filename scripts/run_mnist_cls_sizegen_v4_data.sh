#!/bin/bash
# Size Generalization v4 data: add large test widths (w384-w1024) on top of v3.
#
# Only the ~1k INRs that are actually evaluated get fitted at each new width
# (`generate_inrs.py --only-from-split`), instead of all 70k images — at w1024 the
# full dataset would be ~300 GB for a 1k-image test set.
#
# Prerequisite: scripts/extend_sizegen_splits.py has written
# mnist_sizegen_splits_v4.json into every w{N}_{sp,mup24} directory. That file is
# the manifest of which images to fit, and it guarantees the test sets are
# disjoint across all widths (and from train/val).
#
# SP needs a per-width LR sweep (the stable LR shrinks with width under standard
# parameterization: 0.01 at w24 down to 0.002 at w192/w256). muP24 does not — its
# optimizer already rescales per-layer LRs by base_width/width, so lr=0.01 is
# reused at every width, exactly as for w24-w256.
#
# The sweep selects the LR with the lowest failure rate (breaking ties on PSNR) —
# the same rule that produced the w24-w256 fits, so one recipe covers all 13 widths.
# A failed fit is a broken input the metanetwork still has to classify, which matters
# more downstream than PSNR differences among the fits that succeeded.
#
# Caveat worth knowing when reading the results: even at its best LR, SP fit quality
# declines past w256 (33.4 dB) — w768 and w1024 reach only ~29.5 dB, and no candidate
# LR gets w1024's failure rate under 2.6%. Part of any accuracy drop at those widths
# is the INRs getting worse, not the metanetwork failing to generalize. muP24 (fixed
# lr=0.01) is the controlled comparison here.
#
# Usage: bash scripts/run_mnist_cls_sizegen_v4_data.sh [GPU_ID]
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
PYTHON=${PYTHON:-python}
SPLIT=mnist_sizegen_splits_v4.json
WIDTHS=(384 512 768 1024)
LOG_DIR=/tmp/sizegen_v4_data
mkdir -p $LOG_DIR
LR_FILE=$LOG_DIR/sp_lrs.txt

# Batch size for the parallel bmm fitter, shrinking with width to bound memory.
batch_size_for() {
    case $1 in
        384|512) echo 512 ;;
        *)       echo 256 ;;
    esac
}

echo "=== sizegen v4 data generation on GPU $GPU — started $(date) ==="

# ---------------------------------------------------------------
# SP: per-width LR sweep, then fit the v4 test subset
# ---------------------------------------------------------------
for W in "${WIDTHS[@]}"; do
    BS=$(batch_size_for $W)

    LR=$(grep -E "^${W} " $LR_FILE 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -n "$LR" ]; then
        echo "--- SP w${W}: reusing swept lr=$LR ---"
    else
        echo "--- SP w${W}: LR sweep (candidates 0.0005 0.001 0.002 0.005) ---"
        $PYTHON scripts/sweep_inr_lr.py --width $W \
            --lr-candidates 0.0005 0.001 0.002 0.005 \
            --batch-size $BS --output-file $LR_FILE \
            2>&1 | tee ${LOG_DIR}/sweep_sp_w${W}.log
        LR=$(grep -E "^${W} " $LR_FILE | tail -1 | awk '{print $2}')
    fi
    [ -n "$LR" ] || { echo "FATAL: no LR selected for w${W}"; exit 1; }

    echo "--- SP w${W}: fitting v4 test subset (lr=$LR, batch=$BS) ---"
    $PYTHON scripts/generate_inrs.py --widths $W --lr $LR --batch-size $BS \
        --only-from-split ${ANYDIM_DATA_ROOT}/mnist_inrs/w${W}_sp/${SPLIT} \
        --skip-existing \
        2>&1 | tee ${LOG_DIR}/gen_sp_w${W}.log
done

# ---------------------------------------------------------------
# muP24: fixed lr=0.01 at every width (base_width=24)
# ---------------------------------------------------------------
for W in "${WIDTHS[@]}"; do
    BS=$(batch_size_for $W)
    echo "--- muP24 w${W}: fitting v4 test subset (lr=0.01, batch=$BS) ---"
    $PYTHON scripts/generate_inrs.py --widths $W --lr 0.01 --batch-size $BS \
        --mup --base-width 24 --output-suffix mup24 \
        --only-from-split ${ANYDIM_DATA_ROOT}/mnist_inrs/w${W}_mup24/${SPLIT} \
        --skip-existing \
        2>&1 | tee ${LOG_DIR}/gen_mup24_w${W}.log
done

# ---------------------------------------------------------------
# Fit quality of the new widths (PSNR / failure rate)
# ---------------------------------------------------------------
echo "=== Fit quality ==="
$PYTHON scripts/verify_inr_quality.py --widths "${WIDTHS[@]}" --init-type sp \
    2>&1 | tee ${LOG_DIR}/quality_sp.log
$PYTHON scripts/verify_inr_quality.py --widths "${WIDTHS[@]}" --init-type mup24 \
    2>&1 | tee ${LOG_DIR}/quality_mup24.log

echo ""
echo "Selected SP LRs:"; cat $LR_FILE
echo "=== done $(date) ==="

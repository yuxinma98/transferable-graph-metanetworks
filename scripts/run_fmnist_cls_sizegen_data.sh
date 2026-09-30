#!/bin/bash
# Fashion-MNIST size-generalization data: train w32, test w32-w1024.
#
# Mirrors the MNIST v3/v4 recipe (train w24, test w24-w1024) with w32 as the
# training width, and produces the two parameterizations that experiment compares:
#
#   w{N}_sp/     standard parameterization, per-width swept fitting LR
#   w{N}_mup32/  muP with base_width=32 (= the training width, so w32 muP is
#                mathematically identical to w32 SP), one LR swept at w32 and
#                reused at every width
#
# Order matters: the split JSON is written FIRST and acts as the fitting manifest.
# Only w32 gets all 70k images fitted (it supplies train + val); every larger width
# fits only the ~1k INRs its disjoint test set names — fitting all 70k at w1024
# would be ~290 GB to support a 1k-image test set.
#
# LR protocol (identical selection rule everywhere: lowest failure rate at
# MSE > 0.01, ties broken on higher mean PSNR):
#   * SP    — swept per width, since the stable LR shrinks as width grows.
#   * muP32 — swept once at w32 and reused, because get_mup_optimizer already
#             rescales per-layer LRs by base_width/width. Sweeping it per width
#             would assume away the very property being tested.
# Every sweep runs with --auto-extend, so the grid is widened (halved below /
# doubled above) until the winner is an interior point and no width is left with
# an LR at the edge of its search.
#
# Resumable: selected LRs are cached in $LOG_DIR/{sp,mup32}_lrs.txt and fits use
# --skip-existing, so re-running picks up where it stopped.
#
# Usage: bash scripts/run_fmnist_cls_sizegen_data.sh [GPU_ID]
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

GPU=${1:-4}
export CUDA_VISIBLE_DEVICES=$GPU
PYTHON=${PYTHON:-python}

TRAIN_W=32
# Widths the split reserves images for. w1536 is reserved but deliberately NOT
# fitted (2026-08-19): w1024 = 32x the training width is far enough, and its SP
# fits are already the quality floor. Keep it in this list — the split partitions
# images by cursor, so dropping a width here reshuffles every other width's test
# set and would invalidate the fits already on disk. To add w1536 later, just fit
# it; w1536_{sp,mup32}/ already hold its (unfitted) 1k-image split.
TEST_WIDTHS=(48 64 80 96 128 192 256 384 512 768 1024 1536)
# Widths actually swept / fitted / reported.
FIT_WIDTHS=($TRAIN_W 48 64 80 96 128 192 256 384 512 768 1024)
SPLIT=fmnist_sizegen_splits_v2.json
INR=$ANYDIM_DATA_ROOT/fmnist_inrs
LOG_DIR=/tmp/fmnist_sizegen_data
mkdir -p $LOG_DIR
SP_LR_FILE=$LOG_DIR/sp_lrs.txt
MUP_LR_FILE=$LOG_DIR/mup32_lrs.txt

# Batch size for the parallel bmm fitter, shrinking with width to bound memory
# (measured: 12.5 GiB peak at w1536/bs256; throughput is compute-bound, so a
# larger batch buys nothing).
batch_size_for() {
    case $1 in
        384|512) echo 384 ;;
        768|1024|1536) echo 256 ;;
        *) echo 512 ;;
    esac
}

# Starting SP grid, centred on the LR that MNIST selected at a comparable width.
# --auto-extend takes over if the winner lands on an edge.
lr_grid_for() {
    case $1 in
        1024|1536) echo "0.000125 0.00025 0.0005 0.001" ;;
        384|512|768) echo "0.00025 0.0005 0.001 0.002" ;;
        128|192|256) echo "0.001 0.002 0.005 0.01" ;;
        *) echo "0.0025 0.005 0.01 0.02" ;;
    esac
}

echo "=== FMNIST sizegen data (train w${TRAIN_W}, fitting w${FIT_WIDTHS[*]}) on GPU $GPU — $(date) ==="

# ---------------------------------------------------------------------------
# 1. Splits first — they are the manifest of which INRs to fit.
#    Deterministic given --seed, and identical for both init types, so each
#    image keeps one role (train / val / test at exactly one width).
# ---------------------------------------------------------------------------
for SUF in sp mup32; do
    if [ -f $INR/w${TRAIN_W}_${SUF}/${SPLIT} ]; then
        echo "--- splits (_${SUF}): ${SPLIT} already present, keeping it ---"
        continue
    fi
    echo "--- splits (_${SUF}): 56k train / 1k val / 1k test per width ---"
    $PYTHON scripts/generate_sizegen_splits.py --dataset fmnist --init-type $SUF \
        --from-dataset --create-dirs \
        --train-width $TRAIN_W --test-widths "${TEST_WIDTHS[@]}" \
        --val-size 1000 --test-size 1000 --seed 20260819 \
        --split-name $SPLIT \
        2>&1 | tee ${LOG_DIR}/splits_${SUF}.log
done

# ---------------------------------------------------------------------------
# 2. muP32 LR: swept once at the base width, reused everywhere.
# ---------------------------------------------------------------------------
MUP_LR=$(grep -E "^${TRAIN_W} " $MUP_LR_FILE 2>/dev/null | tail -1 | awk '{print $2}')
if [ -n "$MUP_LR" ]; then
    echo "--- muP32 w${TRAIN_W}: reusing swept lr=$MUP_LR ---"
else
    echo "--- muP32 w${TRAIN_W}: LR sweep (base_width=32) ---"
    $PYTHON scripts/sweep_inr_lr.py --dataset fmnist --width $TRAIN_W \
        --mup --base-width 32 --lr-candidates $(lr_grid_for $TRAIN_W) \
        --auto-extend --batch-size $(batch_size_for $TRAIN_W) \
        --output-file $MUP_LR_FILE \
        2>&1 | tee ${LOG_DIR}/sweep_mup32_w${TRAIN_W}.log
    MUP_LR=$(grep -E "^${TRAIN_W} " $MUP_LR_FILE | tail -1 | awk '{print $2}')
fi
[ -n "$MUP_LR" ] || { echo "FATAL: no muP32 LR selected"; exit 1; }

# ---------------------------------------------------------------------------
# 3. Per-width SP LR sweep, then fit. w32 fits all 70k images (it is the only
#    width that needs train + val); larger widths fit only their test subset.
# ---------------------------------------------------------------------------
for W in "${FIT_WIDTHS[@]}"; do
    BS=$(batch_size_for $W)

    LR=$(grep -E "^${W} " $SP_LR_FILE 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -n "$LR" ]; then
        echo "--- SP w${W}: reusing swept lr=$LR ---"
    else
        echo "--- SP w${W}: LR sweep (start grid: $(lr_grid_for $W)) ---"
        $PYTHON scripts/sweep_inr_lr.py --dataset fmnist --width $W \
            --lr-candidates $(lr_grid_for $W) --auto-extend \
            --batch-size $BS --output-file $SP_LR_FILE \
            2>&1 | tee ${LOG_DIR}/sweep_sp_w${W}.log
        LR=$(grep -E "^${W} " $SP_LR_FILE | tail -1 | awk '{print $2}')
    fi
    [ -n "$LR" ] || { echo "FATAL: no SP LR selected for w${W}"; exit 1; }

    if [ "$W" = "$TRAIN_W" ]; then
        echo "--- SP w${W}: fitting ALL 70k images (train width; lr=$LR, batch=$BS) ---"
        $PYTHON scripts/generate_inrs.py --dataset fmnist --widths $W --lr $LR \
            --batch-size $BS --skip-existing \
            2>&1 | tee ${LOG_DIR}/gen_sp_w${W}.log
    else
        echo "--- SP w${W}: fitting the 1k test INRs (lr=$LR, batch=$BS) ---"
        $PYTHON scripts/generate_inrs.py --dataset fmnist --widths $W --lr $LR \
            --batch-size $BS --skip-existing \
            --only-from-split $INR/w${W}_sp/${SPLIT} --only-split-keys test \
            2>&1 | tee ${LOG_DIR}/gen_sp_w${W}.log
    fi
done

# ---------------------------------------------------------------------------
# 4. muP32 fits at the single swept LR.
# ---------------------------------------------------------------------------
for W in "${FIT_WIDTHS[@]}"; do
    BS=$(batch_size_for $W)
    if [ "$W" = "$TRAIN_W" ]; then
        echo "--- muP32 w${W}: fitting ALL 70k images (train width; lr=$MUP_LR, batch=$BS) ---"
        $PYTHON scripts/generate_inrs.py --dataset fmnist --widths $W --lr $MUP_LR \
            --mup --base-width 32 --output-suffix mup32 \
            --batch-size $BS --skip-existing \
            2>&1 | tee ${LOG_DIR}/gen_mup32_w${W}.log
    else
        echo "--- muP32 w${W}: fitting the 1k test INRs (lr=$MUP_LR, batch=$BS) ---"
        $PYTHON scripts/generate_inrs.py --dataset fmnist --widths $W --lr $MUP_LR \
            --mup --base-width 32 --output-suffix mup32 \
            --batch-size $BS --skip-existing \
            --only-from-split $INR/w${W}_mup32/${SPLIT} --only-split-keys test \
            2>&1 | tee ${LOG_DIR}/gen_mup32_w${W}.log
    fi
done

# ---------------------------------------------------------------------------
# 5. Fit quality (PSNR / failure rate) of everything that was just written.
# ---------------------------------------------------------------------------
echo "=== Fit quality ==="
$PYTHON scripts/verify_inr_quality.py --dataset fmnist --widths "${FIT_WIDTHS[@]}" \
    --init-type sp 2>&1 | tee ${LOG_DIR}/quality_sp.log
$PYTHON scripts/verify_inr_quality.py --dataset fmnist --widths "${FIT_WIDTHS[@]}" \
    --init-type mup32 2>&1 | tee ${LOG_DIR}/quality_mup32.log

echo ""
echo "Selected SP LRs (width lr):"; cat $SP_LR_FILE
echo "muP32 LR (reused at every width): $MUP_LR"
echo "=== done $(date) ==="

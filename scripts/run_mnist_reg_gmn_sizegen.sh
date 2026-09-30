#!/bin/bash
# Full regression sizegen experiment for plain GMN: LR sweeps + full training
# Train on w16, test on w24-w96. Non-overlapping images.
# 4 conditions: {SP, muP} x {baseline, dup-equiv}
# Task: predict 28x28 pixel values from INR weights (MSE regression).
#
# Uses GPUs 0-5 (parallel sweeps), then GPUs 0-3 (full training).
# Designed to run unattended (~6h total).
#
# Usage: bash scripts/run_mnist_reg_gmn_sizegen.sh
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600  # ~10 epochs
GPUS=(0 1 2 3 4 5)
LOG_DIR=/tmp/gmn_reg_sizegen

mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

# Run a sweep for one condition (7 LRs, 6 at a time then 1 more)
run_sweep() {
    local CONF=$1
    local DUP_EQUIV=$2
    local PREFIX=$3

    echo "--- $PREFIX LR sweep ---"
    PIDS=()
    for i in 0 1 2 3 4 5; do
        local CMD="$PYTHON scripts/train_sizegen_regression.py --conf $CONF --wandb True --max_steps $MAX_STEPS --lr ${LRS[$i]} --run_name ${PREFIX}-lr-${LRS[$i]}"
        [ -n "$DUP_EQUIV" ] && CMD="$CMD --duplication-equiv True"
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} $CMD 2>&1 | tee ${LOG_DIR}/${PREFIX}_${i}.log &
        PIDS+=($!)
        echo "  lr=${LRS[$i]} on GPU ${GPUS[$i]}"
    done
    wait_all "${PIDS[@]}"
    # 7th LR
    local CMD="$PYTHON scripts/train_sizegen_regression.py --conf $CONF --wandb True --max_steps $MAX_STEPS --lr ${LRS[6]} --run_name ${PREFIX}-lr-${LRS[6]}"
    [ -n "$DUP_EQUIV" ] && CMD="$CMD --duplication-equiv True"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $CMD 2>&1 | tee ${LOG_DIR}/${PREFIX}_6.log
    echo "$PREFIX sweep done."
    echo ""
}

# Parse best LR from a sweep's logs
parse_best_lr() {
    local PREFIX=$1
    local best_mse=999
    local best_lr=""
    for i in 0 1 2 3 4 5 6; do
        local logfile="${LOG_DIR}/${PREFIX}_${i}.log"
        local mse=$(grep -a "Best val MSE:" "$logfile" 2>/dev/null | tail -1 | grep -oP '(?<=Best val MSE: )\d+\.\d+')
        if [ -n "$mse" ]; then
            local is_better=$(awk "BEGIN {print ($mse < $best_mse) ? 1 : 0}")
            if [ "$is_better" -eq 1 ]; then
                best_mse=$mse
                best_lr=${LRS[$i]}
            fi
        fi
    done
    echo "$best_lr"
}

echo "=== Phase 1: LR Sweeps (4 conditions × 7 LRs = 28 runs) ==="

run_sweep "configs/mnist_cls/gmn_sizegen_regression_sp.yml" "" "gmn-reg-sp-base"
run_sweep "configs/mnist_cls/gmn_sizegen_regression_sp.yml" "True" "gmn-reg-sp-deq"
run_sweep "configs/mnist_cls/gmn_sizegen_regression_mup.yml" "" "gmn-reg-mup-base"
run_sweep "configs/mnist_cls/gmn_sizegen_regression_mup.yml" "True" "gmn-reg-mup-deq"

echo "=== Phase 1 Complete ==="

SP_BASE_LR=$(parse_best_lr "gmn-reg-sp-base")
SP_DEQ_LR=$(parse_best_lr "gmn-reg-sp-deq")
MUP_BASE_LR=$(parse_best_lr "gmn-reg-mup-base")
MUP_DEQ_LR=$(parse_best_lr "gmn-reg-mup-deq")

echo "Best LRs:"
echo "  SP baseline:   $SP_BASE_LR"
echo "  SP dup-equiv:  $SP_DEQ_LR"
echo "  muP baseline:  $MUP_BASE_LR"
echo "  muP dup-equiv: $MUP_DEQ_LR"
echo ""

echo "=== Phase 2: Full Training (200 epochs, patience=50) ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/gmn_sizegen_regression_sp.yml \
    --wandb True --lr $SP_BASE_LR \
    --run_name "gmn-reg-sp-baseline-full" \
    2>&1 | tee ${LOG_DIR}/full_sp_base.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/gmn_sizegen_regression_sp.yml \
    --duplication-equiv True \
    --wandb True --lr $SP_DEQ_LR \
    --run_name "gmn-reg-sp-dupequiv-full" \
    2>&1 | tee ${LOG_DIR}/full_sp_deq.log &
PID2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/gmn_sizegen_regression_mup.yml \
    --wandb True --lr $MUP_BASE_LR \
    --run_name "gmn-reg-mup-baseline-full" \
    2>&1 | tee ${LOG_DIR}/full_mup_base.log &
PID3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/gmn_sizegen_regression_mup.yml \
    --duplication-equiv True \
    --wandb True --lr $MUP_DEQ_LR \
    --run_name "gmn-reg-mup-dupequiv-full" \
    2>&1 | tee ${LOG_DIR}/full_mup_deq.log &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4

echo "=== All done! ==="
echo "Logs in: ${LOG_DIR}/"

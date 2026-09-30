#!/bin/bash
# Full classification sizegen experiment for the matrix-product GMN: LR sweep + full training
# Train on w16, test on w24-w96. Non-overlapping images.
# 2 conditions: {SP, muP}. The matrix-product GMN is always run with --duplication-equiv True
# (fan-in rescaled edges + mean aggregation + layer-wise mean readout) -- the matrix-product
# MSG function is defined on top of that variant.
#
# Uses GPUs 0-3 (parallel sweeps), then GPUs 0-1 (full training).
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600  # ~10 epochs
GPUS=(0 1 2 3)
LOG_DIR=/tmp/mpgmn_sizegen_sweep

mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

# Run a sweep for one condition (7 LRs, 4 at a time on GPUs 0-3)
run_sweep() {
    local CONF=$1
    local PREFIX=$2

    echo "--- Sweep: $PREFIX ---"
    for batch_start in 0 4; do
        PIDS=()
        for offset in 0 1 2 3; do
            local idx=$((batch_start + offset))
            if [ $idx -ge ${#LRS[@]} ]; then
                continue
            fi
            local gpu=${GPUS[$offset]}
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/train_sizegen.py \
                --conf "$CONF" \
                --duplication-equiv True \
                --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
                --run_name "mpgmn-sweep-${PREFIX}-${LRS[$idx]}" \
                2>&1 | tee "${LOG_DIR}/sweep_${PREFIX}_${idx}.log" &
            PIDS+=($!)
        done
        wait_all "${PIDS[@]}"
    done
    echo "--- Sweep $PREFIX done ---"
}

# Extract best LR from sweep logs for a condition
get_best_lr() {
    local PREFIX=$1
    local BEST_ACC=0
    local BEST_LR="2.5e-4"  # fallback

    for i in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${i}.log"
        if [ ! -f "$LOG" ]; then continue; fi
        local ACC=$(grep "Best val acc:" "$LOG" | tail -1 | awk '{print $NF}')
        if [ -z "$ACC" ]; then continue; fi
        local BETTER=$(python3 -c "print(1 if $ACC > $BEST_ACC else 0)")
        if [ "$BETTER" = "1" ]; then
            BEST_ACC=$ACC
            BEST_LR=${LRS[$i]}
        fi
    done
    echo "$BEST_LR"
}

echo "================================================================"
echo "=== Matrix-Product GMN Size Generalization (GPUs 0-3)        ==="
echo "=== Started: $(date)                                         ==="
echo "================================================================"
echo ""

echo "=== Phase 1: LR Sweeps (2 conditions x 7 LRs = 14 runs) ==="

run_sweep "configs/mnist_cls/mpgmn_sizegen_sp.yml" "sp"
run_sweep "configs/mnist_cls/mpgmn_sizegen_mup.yml" "mup"

echo ""
echo "=== Phase 1 Complete ==="

LR_SP=$(get_best_lr "sp")
LR_MUP=$(get_best_lr "mup")

echo "Best LRs from sweep:"
echo "  SP:   $LR_SP"
echo "  muP:  $LR_MUP"
echo ""

echo "=== Phase 2: Full Training (200 epochs, patience=50) ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/mpgmn_sizegen_sp.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SP \
    --run_name "mpgmn-sizegen-sp-deq" \
    2>&1 | tee ${LOG_DIR}/full_sp.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/mpgmn_sizegen_mup.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_MUP \
    --run_name "mpgmn-sizegen-mup-deq" \
    2>&1 | tee ${LOG_DIR}/full_mup.log &
PID2=$!

wait $PID1 $PID2
echo "Both full training runs done."

echo ""
echo "================================================================"
echo "=== All done: $(date) ==="
echo "=== Logs in $LOG_DIR ==="
echo "=== Check wandb project 'mnist_cls' for results ==="
echo "================================================================"

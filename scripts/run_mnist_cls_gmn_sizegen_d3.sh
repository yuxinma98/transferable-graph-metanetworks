#!/bin/bash
# Full classification sizegen experiment for plain GMN: LR sweeps + full training
# Train on w16, test on w24-w96. Non-overlapping images.
# 4 conditions: {SP, muP} x {baseline, dup-equiv}
#
# Uses GPUs 2-7 (parallel sweeps), then GPUs 2-5 (full training).
# Designed to run unattended (~4h total).
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600  # ~10 epochs
GPUS=(2 3 4 5 6 7)
LOG_DIR=/tmp/gmn_sizegen_sweep

mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

# Run a sweep for one condition (7 LRs, 6 at a time on GPUs 2-7)
run_sweep() {
    local CONF=$1
    local DUP_EQUIV=$2
    local PREFIX=$3

    echo "--- Sweep: $PREFIX ---"
    local DUP_FLAG=""
    if [ "$DUP_EQUIV" = "true" ]; then
        DUP_FLAG="--duplication-equiv True"
    fi

    # Run LRs in batches of 6
    for batch_start in 0 6; do
        PIDS=()
        for offset in 0 1 2 3 4 5; do
            local idx=$((batch_start + offset))
            if [ $idx -ge ${#LRS[@]} ]; then
                continue
            fi
            local gpu=${GPUS[$offset]}
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/train_sizegen.py \
                --conf "$CONF" \
                $DUP_FLAG \
                --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
                --run_name "gmn-sweep-${PREFIX}-${LRS[$idx]}" \
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
echo "=== GMN Size Generalization Experiment (GPUs 2-7)            ==="
echo "=== Started: $(date)                                         ==="
echo "================================================================"
echo ""

echo "=== Phase 1: LR Sweeps (4 conditions x 7 LRs = 28 runs) ==="

run_sweep "configs/mnist_cls/gmn_sizegen_sp_d3.yml" "false" "sp-baseline"
run_sweep "configs/mnist_cls/gmn_sizegen_sp_d3.yml" "true" "sp-dupequiv"
run_sweep "configs/mnist_cls/gmn_sizegen_mup_d3.yml" "false" "mup-baseline"
run_sweep "configs/mnist_cls/gmn_sizegen_mup_d3.yml" "true" "mup-dupequiv"

echo ""
echo "=== Phase 1 Complete ==="

# Extract best LRs
LR_SP_BASE=$(get_best_lr "sp-baseline")
LR_SP_DEQ=$(get_best_lr "sp-dupequiv")
LR_MUP_BASE=$(get_best_lr "mup-baseline")
LR_MUP_DEQ=$(get_best_lr "mup-dupequiv")

echo "Best LRs from sweep:"
echo "  SP baseline:   $LR_SP_BASE"
echo "  SP dup-equiv:  $LR_SP_DEQ"
echo "  muP baseline:  $LR_MUP_BASE"
echo "  muP dup-equiv: $LR_MUP_DEQ"
echo ""

echo "=== Phase 2: Full Training (200 epochs, patience=50) ==="

# Run all 4 conditions in parallel on GPUs 2-5
echo "--- Full training: all 4 conditions ---"
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_d3.yml \
    --wandb True --lr $LR_SP_BASE \
    --run_name "gmn-sizegen-sp-base" \
    2>&1 | tee ${LOG_DIR}/full_sp_base.log &
PID1=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_d3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SP_DEQ \
    --run_name "gmn-sizegen-sp-deq" \
    2>&1 | tee ${LOG_DIR}/full_sp_deq.log &
PID2=$!

CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup_d3.yml \
    --wandb True --lr $LR_MUP_BASE \
    --run_name "gmn-sizegen-mup-base" \
    2>&1 | tee ${LOG_DIR}/full_mup_base.log &
PID3=$!

CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup_d3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_MUP_DEQ \
    --run_name "gmn-sizegen-mup-deq" \
    2>&1 | tee ${LOG_DIR}/full_mup_deq.log &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "All 4 full training runs done."

echo ""
echo "================================================================"
echo "=== All done: $(date) ==="
echo "=== Logs in $LOG_DIR ==="
echo "=== Check wandb project 'mnist_cls' for results ==="
echo "================================================================"

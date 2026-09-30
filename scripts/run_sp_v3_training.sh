#!/bin/bash
# SP v3 training: LR sweep then full training for ScaleGMN/GMN × base/deq
# Splits already generated. INRs already fitted with per-width LR tuning.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/sp_sizegen_v3_training
mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

echo "================================================================"
echo "=== SP v3 Training: $(date)"
echo "================================================================"

# ============================================================
# Phase 1: LR Sweeps (4 conditions on 4 GPUs)
# ============================================================
echo ""
echo "=== Phase 1: LR Sweeps (ScaleGMN + GMN, SP base + deq) ==="

run_sweep() {
    local CONF=$1
    local DUP_EQUIV=$2
    local PREFIX=$3
    local GPU=$4

    local DUP_FLAG=""
    if [ "$DUP_EQUIV" = "true" ]; then
        DUP_FLAG="--duplication-equiv True"
    fi

    for idx in 0 1 2 3 4 5 6; do
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
            --conf "$CONF" \
            $DUP_FLAG \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-v3-${PREFIX}-${LRS[$idx]}" \
            2>&1 | tee "${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
    done
}

run_sweep "configs/mnist_cls/scalegmn_sizegen_relu_v3.yml" "false" "sgmn-sp-base" 0 &
PID1=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_relu_v3.yml" "true" "sgmn-sp-deq" 1 &
PID2=$!
run_sweep "configs/mnist_cls/gmn_sizegen_relu_v3.yml" "false" "gmn-sp-base" 2 &
PID3=$!
run_sweep "configs/mnist_cls/gmn_sizegen_relu_v3.yml" "true" "gmn-sp-deq" 3 &
PID4=$!

wait_all $PID1 $PID2 $PID3 $PID4
echo "LR sweeps complete: $(date)"

# Extract best LRs
get_best_lr() {
    local PREFIX=$1
    local BEST_ACC=0
    local BEST_LR="2.5e-4"

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

LR_SGMN_SP_BASE=$(get_best_lr "sgmn-sp-base")
LR_SGMN_SP_DEQ=$(get_best_lr "sgmn-sp-deq")
LR_GMN_SP_BASE=$(get_best_lr "gmn-sp-base")
LR_GMN_SP_DEQ=$(get_best_lr "gmn-sp-deq")

echo ""
echo "Best LRs found:"
echo "  ScaleGMN SP baseline:   $LR_SGMN_SP_BASE"
echo "  ScaleGMN SP dup-equiv:  $LR_SGMN_SP_DEQ"
echo "  GMN SP baseline:        $LR_GMN_SP_BASE"
echo "  GMN SP dup-equiv:       $LR_GMN_SP_DEQ"

# ============================================================
# Phase 2: Full Training (4 conditions on 4 GPUs)
# ============================================================
echo ""
echo "=== Phase 2: Full Training ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_relu_v3.yml \
    --wandb True --lr $LR_SGMN_SP_BASE \
    --run_name "sizegen-v3-sgmn-sp-base-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_base.log &
PID_F1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_relu_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SGMN_SP_DEQ \
    --run_name "sizegen-v3-sgmn-sp-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_deq.log &
PID_F2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_relu_v3.yml \
    --wandb True --lr $LR_GMN_SP_BASE \
    --run_name "sizegen-v3-gmn-sp-base-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_base.log &
PID_F3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_relu_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_GMN_SP_DEQ \
    --run_name "sizegen-v3-gmn-sp-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_deq.log &
PID_F4=$!

wait_all $PID_F1 $PID_F2 $PID_F3 $PID_F4
echo "Full training complete: $(date)"

echo ""
echo "================================================================"
echo "=== All done: $(date)"
echo "=== Logs in $LOG_DIR"
echo "=== Check wandb project 'mnist_cls' for sizegen-v3-* runs"
echo "================================================================"

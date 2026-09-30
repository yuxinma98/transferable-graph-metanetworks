#!/bin/bash
# Size Generalization v3 — training only (data + splits already done).
# Train w24, test w24-w256. 4 conditions each for ScaleGMN and GMN.
# Uses GPUs 0-3. Designed to run unattended.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/sizegen_v3_final
mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

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

echo "================================================================"
echo "=== Size Gen v3 Training: $(date)                            ==="
echo "================================================================"

# ============================================================
# Phase 1: ScaleGMN LR Sweeps (4 conditions, 4 GPUs in parallel)
# ============================================================
echo ""
echo "=== Phase 1: ScaleGMN LR Sweeps ==="

run_sweep "configs/mnist_cls/scalegmn_sizegen_sp_v3.yml" "false" "sgmn-sp-base" 0 &
PID1=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_sp_v3.yml" "true" "sgmn-sp-deq" 1 &
PID2=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml" "false" "sgmn-mup-base" 2 &
PID3=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml" "true" "sgmn-mup-deq" 3 &
PID4=$!

wait_all $PID1 $PID2 $PID3 $PID4
echo "ScaleGMN sweeps complete: $(date)"

LR_SGMN_SP_BASE=$(get_best_lr "sgmn-sp-base")
LR_SGMN_SP_DEQ=$(get_best_lr "sgmn-sp-deq")
LR_SGMN_MUP_BASE=$(get_best_lr "sgmn-mup-base")
LR_SGMN_MUP_DEQ=$(get_best_lr "sgmn-mup-deq")

echo "ScaleGMN best LRs: SP_base=$LR_SGMN_SP_BASE SP_deq=$LR_SGMN_SP_DEQ muP_base=$LR_SGMN_MUP_BASE muP_deq=$LR_SGMN_MUP_DEQ"

# ============================================================
# Phase 2: ScaleGMN Full Training (4 GPUs)
# ============================================================
echo ""
echo "=== Phase 2: ScaleGMN Full Training ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_sp_v3.yml \
    --wandb True --lr $LR_SGMN_SP_BASE \
    --run_name "sizegen-v3-sgmn-sp-base" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_base.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_sp_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SGMN_SP_DEQ \
    --run_name "sizegen-v3-sgmn-sp-deq" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_deq.log &
PID2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml \
    --wandb True --lr $LR_SGMN_MUP_BASE \
    --run_name "sizegen-v3-sgmn-mup-base" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_mup_base.log &
PID3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SGMN_MUP_DEQ \
    --run_name "sizegen-v3-sgmn-mup-deq" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_mup_deq.log &
PID4=$!

wait_all $PID1 $PID2 $PID3 $PID4
echo "ScaleGMN full training complete: $(date)"

# ============================================================
# Phase 3: GMN LR Sweeps (4 conditions, 4 GPUs)
# ============================================================
echo ""
echo "=== Phase 3: GMN LR Sweeps ==="

run_sweep "configs/mnist_cls/gmn_sizegen_sp_v3.yml" "false" "gmn-sp-base" 0 &
PID1=$!
run_sweep "configs/mnist_cls/gmn_sizegen_sp_v3.yml" "true" "gmn-sp-deq" 1 &
PID2=$!
run_sweep "configs/mnist_cls/gmn_sizegen_mup24_v3.yml" "false" "gmn-mup-base" 2 &
PID3=$!
run_sweep "configs/mnist_cls/gmn_sizegen_mup24_v3.yml" "true" "gmn-mup-deq" 3 &
PID4=$!

wait_all $PID1 $PID2 $PID3 $PID4
echo "GMN sweeps complete: $(date)"

LR_GMN_SP_BASE=$(get_best_lr "gmn-sp-base")
LR_GMN_SP_DEQ=$(get_best_lr "gmn-sp-deq")
LR_GMN_MUP_BASE=$(get_best_lr "gmn-mup-base")
LR_GMN_MUP_DEQ=$(get_best_lr "gmn-mup-deq")

echo "GMN best LRs: SP_base=$LR_GMN_SP_BASE SP_deq=$LR_GMN_SP_DEQ muP_base=$LR_GMN_MUP_BASE muP_deq=$LR_GMN_MUP_DEQ"

# ============================================================
# Phase 4: GMN Full Training (4 GPUs)
# ============================================================
echo ""
echo "=== Phase 4: GMN Full Training ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_v3.yml \
    --wandb True --lr $LR_GMN_SP_BASE \
    --run_name "sizegen-v3-gmn-sp-base" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_base.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_GMN_SP_DEQ \
    --run_name "sizegen-v3-gmn-sp-deq" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_deq.log &
PID2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup24_v3.yml \
    --wandb True --lr $LR_GMN_MUP_BASE \
    --run_name "sizegen-v3-gmn-mup-base" \
    2>&1 | tee ${LOG_DIR}/full_gmn_mup_base.log &
PID3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup24_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_GMN_MUP_DEQ \
    --run_name "sizegen-v3-gmn-mup-deq" \
    2>&1 | tee ${LOG_DIR}/full_gmn_mup_deq.log &
PID4=$!

wait_all $PID1 $PID2 $PID3 $PID4
echo "GMN full training complete: $(date)"

echo ""
echo "================================================================"
echo "=== All done: $(date)                                        ==="
echo "=== Logs in $LOG_DIR                                         ==="
echo "=== Check wandb project 'mnist_cls' groups sizegen-v3-*      ==="
echo "================================================================"

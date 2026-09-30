#!/bin/bash
# Size Generalization v3: train w24, test w24-w256.
# Full pipeline: data generation → splits → LR sweeps → full training.
# Uses up to 6 GPUs (leaves 2 free). Designed to run unattended.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600  # ~10 epochs
LOG_DIR=/tmp/sizegen_v3
mkdir -p $LOG_DIR

wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

echo "================================================================"
echo "=== Size Generalization v3 (train w24, test w24-w256)        ==="
echo "=== Started: $(date)                                         ==="
echo "================================================================"

# ============================================================
# Phase 1: Data Generation (6 GPUs in parallel)
# ============================================================
echo ""
echo "=== Phase 1: Data Generation ==="

# SP ReLU: generate w128, w192, w256
echo "--- Generating SP ReLU INRs (w128, w192, w256) ---"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 128 --init-type sp \
    2>&1 | tee ${LOG_DIR}/gen_sp_w128.log &
PID_SP128=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 192 --init-type sp \
    2>&1 | tee ${LOG_DIR}/gen_sp_w192.log &
PID_SP192=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 256 --init-type sp \
    2>&1 | tee ${LOG_DIR}/gen_sp_w256.log &
PID_SP256=$!

# muP24: generate all widths (base_width=24)
echo "--- Generating muP24 INRs (all widths, base_width=24) ---"
CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 24 32 48 --init-type sp \
    --mup --base-width 24 --output-suffix mup24 \
    2>&1 | tee ${LOG_DIR}/gen_mup24_small.log &
PID_MUP_SMALL=$!

CUDA_VISIBLE_DEVICES=4 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 64 80 96 --init-type sp \
    --mup --base-width 24 --output-suffix mup24 \
    2>&1 | tee ${LOG_DIR}/gen_mup24_med.log &
PID_MUP_MED=$!

CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/generate_inrs.py \
    --mode fixed --widths 128 192 256 --init-type sp \
    --mup --base-width 24 --output-suffix mup24 \
    2>&1 | tee ${LOG_DIR}/gen_mup24_large.log &
PID_MUP_LARGE=$!

wait_all $PID_SP128 $PID_SP192 $PID_SP256 $PID_MUP_SMALL $PID_MUP_MED $PID_MUP_LARGE
echo "Phase 1 complete: $(date)"

# ============================================================
# Phase 2: Generate Sizegen Splits
# ============================================================
echo ""
echo "=== Phase 2: Generate Sizegen Splits ==="

$PYTHON scripts/generate_sizegen_splits.py \
    --train-width 24 --test-widths 32 48 64 80 96 128 192 256 \
    --init-type sp --dataset mnist \
    --split-name mnist_sizegen_splits.json \
    2>&1 | tee ${LOG_DIR}/splits_sp.log

$PYTHON scripts/generate_sizegen_splits.py \
    --train-width 24 --test-widths 32 48 64 80 96 128 192 256 \
    --init-type mup24 --dataset mnist \
    --split-name mnist_sizegen_splits.json \
    2>&1 | tee ${LOG_DIR}/splits_mup24.log

echo "Phase 2 complete."

# ============================================================
# Phase 3: ScaleGMN LR Sweeps (4 conditions, 6 GPUs)
# ============================================================
echo ""
echo "=== Phase 3: ScaleGMN LR Sweeps ==="

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

# Run all 4 ScaleGMN sweeps in parallel (one per GPU)
run_sweep "configs/mnist_cls/scalegmn_sizegen_sp_v3.yml" "false" "sgmn-sp-base" 0 &
PID_SW1=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_sp_v3.yml" "true" "sgmn-sp-deq" 1 &
PID_SW2=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml" "false" "sgmn-mup-base" 2 &
PID_SW3=$!
run_sweep "configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml" "true" "sgmn-mup-deq" 3 &
PID_SW4=$!

wait_all $PID_SW1 $PID_SW2 $PID_SW3 $PID_SW4
echo "ScaleGMN sweeps complete: $(date)"

# Extract best LRs
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

LR_SGMN_SP_BASE=$(get_best_lr "sgmn-sp-base")
LR_SGMN_SP_DEQ=$(get_best_lr "sgmn-sp-deq")
LR_SGMN_MUP_BASE=$(get_best_lr "sgmn-mup-base")
LR_SGMN_MUP_DEQ=$(get_best_lr "sgmn-mup-deq")

echo "ScaleGMN best LRs:"
echo "  SP baseline:   $LR_SGMN_SP_BASE"
echo "  SP dup-equiv:  $LR_SGMN_SP_DEQ"
echo "  muP baseline:  $LR_SGMN_MUP_BASE"
echo "  muP dup-equiv: $LR_SGMN_MUP_DEQ"

# ============================================================
# Phase 4: ScaleGMN Full Training (4 GPUs)
# ============================================================
echo ""
echo "=== Phase 4: ScaleGMN Full Training ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_sp_v3.yml \
    --wandb True --lr $LR_SGMN_SP_BASE \
    --run_name "sizegen-v3-sgmn-sp-base-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_base.log &
PID_F1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_sp_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SGMN_SP_DEQ \
    --run_name "sizegen-v3-sgmn-sp-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_sp_deq.log &
PID_F2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml \
    --wandb True --lr $LR_SGMN_MUP_BASE \
    --run_name "sizegen-v3-sgmn-mup-base-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_mup_base.log &
PID_F3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/scalegmn_sizegen_mup24_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_SGMN_MUP_DEQ \
    --run_name "sizegen-v3-sgmn-mup-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_sgmn_mup_deq.log &
PID_F4=$!

wait_all $PID_F1 $PID_F2 $PID_F3 $PID_F4
echo "ScaleGMN full training complete: $(date)"

# ============================================================
# Phase 5: GMN LR Sweeps (4 conditions, 4 GPUs)
# ============================================================
echo ""
echo "=== Phase 5: GMN LR Sweeps ==="

run_sweep "configs/mnist_cls/gmn_sizegen_sp_v3.yml" "false" "gmn-sp-base" 0 &
PID_GW1=$!
run_sweep "configs/mnist_cls/gmn_sizegen_sp_v3.yml" "true" "gmn-sp-deq" 1 &
PID_GW2=$!
run_sweep "configs/mnist_cls/gmn_sizegen_mup24_v3.yml" "false" "gmn-mup-base" 2 &
PID_GW3=$!
run_sweep "configs/mnist_cls/gmn_sizegen_mup24_v3.yml" "true" "gmn-mup-deq" 3 &
PID_GW4=$!

wait_all $PID_GW1 $PID_GW2 $PID_GW3 $PID_GW4
echo "GMN sweeps complete: $(date)"

LR_GMN_SP_BASE=$(get_best_lr "gmn-sp-base")
LR_GMN_SP_DEQ=$(get_best_lr "gmn-sp-deq")
LR_GMN_MUP_BASE=$(get_best_lr "gmn-mup-base")
LR_GMN_MUP_DEQ=$(get_best_lr "gmn-mup-deq")

echo "GMN best LRs:"
echo "  SP baseline:   $LR_GMN_SP_BASE"
echo "  SP dup-equiv:  $LR_GMN_SP_DEQ"
echo "  muP baseline:  $LR_GMN_MUP_BASE"
echo "  muP dup-equiv: $LR_GMN_MUP_DEQ"

# ============================================================
# Phase 6: GMN Full Training (4 GPUs)
# ============================================================
echo ""
echo "=== Phase 6: GMN Full Training ==="

CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_v3.yml \
    --wandb True --lr $LR_GMN_SP_BASE \
    --run_name "sizegen-v3-gmn-sp-base-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_base.log &
PID_G1=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_sp_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_GMN_SP_DEQ \
    --run_name "sizegen-v3-gmn-sp-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_sp_deq.log &
PID_G2=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup24_v3.yml \
    --wandb True --lr $LR_GMN_MUP_BASE \
    --run_name "sizegen-v3-gmn-mup-base-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_mup_base.log &
PID_G3=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen.py \
    --conf configs/mnist_cls/gmn_sizegen_mup24_v3.yml \
    --duplication-equiv True \
    --wandb True --lr $LR_GMN_MUP_DEQ \
    --run_name "sizegen-v3-gmn-mup-deq-full" \
    2>&1 | tee ${LOG_DIR}/full_gmn_mup_deq.log &
PID_G4=$!

wait_all $PID_G1 $PID_G2 $PID_G3 $PID_G4
echo "GMN full training complete: $(date)"

echo ""
echo "================================================================"
echo "=== All done: $(date)                                        ==="
echo "=== Logs in $LOG_DIR                                         ==="
echo "=== Check wandb project 'mnist_cls' groups sizegen-v3-*      ==="
echo "================================================================"

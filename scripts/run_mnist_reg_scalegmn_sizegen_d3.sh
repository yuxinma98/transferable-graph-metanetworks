#!/bin/bash
# Deep (3 hidden layer) regression sizegen: LR sweeps + full training
# Train on w16, test on w24-w96. Non-overlapping images.
# 4 conditions: {SP, muP} x {baseline, dup-equiv}
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600  # ~10 epochs

# Function to wait for all background jobs
wait_all() {
    for pid in "$@"; do
        wait $pid
    done
}

echo "=== Phase 1: LR Sweeps (4 conditions x 7 LRs = 28 runs) ==="
echo "Using 6 GPUs (0-5), leaving 6,7 free."

# --- SP baseline sweep (7 runs on GPUs 0-5, then 1 more) ---
echo "--- SP baseline sweep ---"
PIDS=()
for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i $PYTHON scripts/train_sizegen_regression.py \
        --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
        --wandb True --max_steps $MAX_STEPS --lr ${LRS[$i]} \
        --run_name "lr-sweep-reg-d3-sp-baseline-${LRS[$i]}" \
        2>&1 | tee /tmp/sweep_reg_d3_sp_base_${i}.log &
    PIDS+=($!)
done
wait_all "${PIDS[@]}"
# Last LR
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
    --wandb True --max_steps $MAX_STEPS --lr ${LRS[6]} \
    --run_name "lr-sweep-reg-d3-sp-baseline-${LRS[6]}" \
    2>&1 | tee /tmp/sweep_reg_d3_sp_base_6.log
echo "SP baseline sweep done."

# --- SP dup-equiv sweep ---
echo "--- SP dup-equiv sweep ---"
PIDS=()
for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i $PYTHON scripts/train_sizegen_regression.py \
        --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
        --duplication-equiv True \
        --wandb True --max_steps $MAX_STEPS --lr ${LRS[$i]} \
        --run_name "lr-sweep-reg-d3-sp-dupequiv-${LRS[$i]}" \
        2>&1 | tee /tmp/sweep_reg_d3_sp_deq_${i}.log &
    PIDS+=($!)
done
wait_all "${PIDS[@]}"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
    --duplication-equiv True \
    --wandb True --max_steps $MAX_STEPS --lr ${LRS[6]} \
    --run_name "lr-sweep-reg-d3-sp-dupequiv-${LRS[6]}" \
    2>&1 | tee /tmp/sweep_reg_d3_sp_deq_6.log
echo "SP dup-equiv sweep done."

# --- muP baseline sweep ---
echo "--- muP baseline sweep ---"
PIDS=()
for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i $PYTHON scripts/train_sizegen_regression.py \
        --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
        --wandb True --max_steps $MAX_STEPS --lr ${LRS[$i]} \
        --run_name "lr-sweep-reg-d3-mup-baseline-${LRS[$i]}" \
        2>&1 | tee /tmp/sweep_reg_d3_mup_base_${i}.log &
    PIDS+=($!)
done
wait_all "${PIDS[@]}"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
    --wandb True --max_steps $MAX_STEPS --lr ${LRS[6]} \
    --run_name "lr-sweep-reg-d3-mup-baseline-${LRS[6]}" \
    2>&1 | tee /tmp/sweep_reg_d3_mup_base_6.log
echo "muP baseline sweep done."

# --- muP dup-equiv sweep ---
echo "--- muP dup-equiv sweep ---"
PIDS=()
for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$i $PYTHON scripts/train_sizegen_regression.py \
        --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
        --duplication-equiv True \
        --wandb True --max_steps $MAX_STEPS --lr ${LRS[$i]} \
        --run_name "lr-sweep-reg-d3-mup-dupequiv-${LRS[$i]}" \
        2>&1 | tee /tmp/sweep_reg_d3_mup_deq_${i}.log &
    PIDS+=($!)
done
wait_all "${PIDS[@]}"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
    --duplication-equiv True \
    --wandb True --max_steps $MAX_STEPS --lr ${LRS[6]} \
    --run_name "lr-sweep-reg-d3-mup-dupequiv-${LRS[6]}" \
    2>&1 | tee /tmp/sweep_reg_d3_mup_deq_6.log
echo "muP dup-equiv sweep done."

echo ""
echo "=== Phase 1 Complete: All LR sweeps done ==="
echo "Check wandb for best LRs, then Phase 2 (full training) will start."
echo ""

# --- Phase 2: Full training with best LR per condition ---
# We'll use the LRs that were best in the previous v2 experiment as starting point
# (can be updated after reviewing sweep results)
# SP baseline: 2e-3, muP baseline: 1e-3
# For dup-equiv, use same LR as baseline until sweep results are checked.

echo "=== Phase 2: Full Training (200 epochs, patience=50) ==="

# Run all 4 conditions in parallel on 4 GPUs
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
    --wandb True --lr 2e-3 \
    --run_name "sizegen-regression-d3-sp-baseline-full" \
    2>&1 | tee /tmp/full_reg_d3_sp_base.log &
PID_SP_BASE=$!

CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml \
    --duplication-equiv True \
    --wandb True --lr 2e-3 \
    --run_name "sizegen-regression-d3-sp-dupequiv-full" \
    2>&1 | tee /tmp/full_reg_d3_sp_deq.log &
PID_SP_DEQ=$!

CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
    --wandb True --lr 1e-3 \
    --run_name "sizegen-regression-d3-mup-baseline-full" \
    2>&1 | tee /tmp/full_reg_d3_mup_base.log &
PID_MUP_BASE=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON scripts/train_sizegen_regression.py \
    --conf configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml \
    --duplication-equiv True \
    --wandb True --lr 1e-3 \
    --run_name "sizegen-regression-d3-mup-dupequiv-full" \
    2>&1 | tee /tmp/full_reg_d3_mup_deq.log &
PID_MUP_DEQ=$!

wait $PID_SP_BASE $PID_SP_DEQ $PID_MUP_BASE $PID_MUP_DEQ

echo "=== Phase 2 Complete: All full training runs done ==="
echo "Check wandb and /tmp/full_*.log for final results."

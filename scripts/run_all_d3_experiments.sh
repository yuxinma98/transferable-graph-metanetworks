#!/bin/bash
# Master orchestration script for all d3 (3-hidden-layer) ScaleGMN experiments.
# Waits for INR generation to finish, generates splits, then runs all experiments.
# Dynamically selects free GPUs (max 4 at a time).
#
# Usage: screen -dmS d3_master bash scripts/run_all_d3_experiments.sh
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG=/tmp/d3_master.log
MAX_GPUS=4

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG
}

# Get list of free GPUs (memory usage < 100 MiB), up to MAX_GPUS
get_free_gpus() {
    local max=${1:-$MAX_GPUS}
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F', ' '$2 < 100 {print $1}' | head -n $max
}

# Wait until at least N GPUs are free
wait_for_gpus() {
    local needed=$1
    while true; do
        local free=$(get_free_gpus $needed | wc -l)
        if [ "$free" -ge "$needed" ]; then
            return
        fi
        log "  Waiting for $needed free GPUs (currently $free free)..."
        sleep 60
    done
}

# Run a sweep for one condition. Uses up to MAX_GPUS GPUs in parallel.
# Args: CONF DUP_EQUIV PREFIX TRAIN_SCRIPT LOG_DIR
run_sweep() {
    local CONF=$1
    local DUP_EQUIV=$2
    local PREFIX=$3
    local TRAIN_SCRIPT=$4
    local LOG_DIR=$5

    local LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
    local MAX_STEPS=17200  # ~10 epochs at batch_size=32

    log "  Sweep: $PREFIX (7 LRs, max $MAX_GPUS at a time)"
    local DUP_FLAG=""
    if [ "$DUP_EQUIV" = "true" ]; then
        DUP_FLAG="--duplication-equiv True"
    fi

    local idx=0
    while [ $idx -lt ${#LRS[@]} ]; do
        # Get available GPUs
        wait_for_gpus 1
        local AVAIL_GPUS=($(get_free_gpus $MAX_GPUS))
        local batch_size=${#AVAIL_GPUS[@]}

        local PIDS=()
        for gpu_idx in $(seq 0 $((batch_size - 1))); do
            if [ $idx -ge ${#LRS[@]} ]; then
                break
            fi
            local gpu=${AVAIL_GPUS[$gpu_idx]}
            CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/$TRAIN_SCRIPT \
                --conf "$CONF" \
                $DUP_FLAG \
                --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
                --run_name "${PREFIX}-lr${LRS[$idx]}" \
                > "${LOG_DIR}/sweep_${PREFIX}_${idx}.log" 2>&1 &
            PIDS+=($!)
            idx=$((idx + 1))
        done
        for pid in "${PIDS[@]}"; do wait $pid || true; done
    done
    log "  Sweep $PREFIX done."
}

# Extract best LR from sweep logs
get_best_lr() {
    local LOG_DIR=$1
    local PREFIX=$2
    local METRIC=${3:-"Best val acc:"}
    local BEST_VAL=""
    local BEST_LR="5e-4"
    local LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)

    for i in 0 1 2 3 4 5 6; do
        local LOGF="${LOG_DIR}/sweep_${PREFIX}_${i}.log"
        if [ ! -f "$LOGF" ]; then continue; fi
        local VAL=$(grep "$METRIC" "$LOGF" | tail -1 | awk '{print $NF}')
        if [ -z "$VAL" ]; then continue; fi
        if [ -z "$BEST_VAL" ]; then
            BEST_VAL=$VAL
            BEST_LR=${LRS[$i]}
        else
            local BETTER=$(python3 -c "print(1 if $VAL > $BEST_VAL else 0)" 2>/dev/null || echo 0)
            if [ "$BETTER" = "1" ]; then
                BEST_VAL=$VAL
                BEST_LR=${LRS[$i]}
            fi
        fi
    done
    echo "$BEST_LR"
}

# Run full training for 4 conditions in parallel (max MAX_GPUS at a time)
# Args: CONF_SP CONF_MUP TRAIN_SCRIPT LOG_DIR PREFIX
run_full_training() {
    local CONF_SP=$1
    local CONF_MUP=$2
    local TRAIN_SCRIPT=$3
    local LOG_DIR=$4
    local PREFIX=$5

    local LR_SP_BASE=$(get_best_lr "$LOG_DIR" "${PREFIX}-sp-base")
    local LR_SP_DEQ=$(get_best_lr "$LOG_DIR" "${PREFIX}-sp-deq")
    local LR_MUP_BASE=$(get_best_lr "$LOG_DIR" "${PREFIX}-mup-base")
    local LR_MUP_DEQ=$(get_best_lr "$LOG_DIR" "${PREFIX}-mup-deq")

    log "  Best LRs: SP-base=$LR_SP_BASE SP-deq=$LR_SP_DEQ muP-base=$LR_MUP_BASE muP-deq=$LR_MUP_DEQ"

    wait_for_gpus $MAX_GPUS
    local AVAIL_GPUS=($(get_free_gpus $MAX_GPUS))
    log "  Full training on GPUs: ${AVAIL_GPUS[*]}"

    CUDA_VISIBLE_DEVICES=${AVAIL_GPUS[0]} $PYTHON scripts/$TRAIN_SCRIPT \
        --conf "$CONF_SP" --wandb True --lr $LR_SP_BASE \
        --run_name "${PREFIX}-sp-base-full" \
        > "${LOG_DIR}/full_sp_base.log" 2>&1 &
    local PID1=$!

    CUDA_VISIBLE_DEVICES=${AVAIL_GPUS[1]} $PYTHON scripts/$TRAIN_SCRIPT \
        --conf "$CONF_SP" --duplication-equiv True --wandb True --lr $LR_SP_DEQ \
        --run_name "${PREFIX}-sp-deq-full" \
        > "${LOG_DIR}/full_sp_deq.log" 2>&1 &
    local PID2=$!

    CUDA_VISIBLE_DEVICES=${AVAIL_GPUS[2]} $PYTHON scripts/$TRAIN_SCRIPT \
        --conf "$CONF_MUP" --wandb True --lr $LR_MUP_BASE \
        --run_name "${PREFIX}-mup-base-full" \
        > "${LOG_DIR}/full_mup_base.log" 2>&1 &
    local PID3=$!

    CUDA_VISIBLE_DEVICES=${AVAIL_GPUS[3]} $PYTHON scripts/$TRAIN_SCRIPT \
        --conf "$CONF_MUP" --duplication-equiv True --wandb True --lr $LR_MUP_DEQ \
        --run_name "${PREFIX}-mup-deq-full" \
        > "${LOG_DIR}/full_mup_deq.log" 2>&1 &
    local PID4=$!

    wait $PID1 || true
    wait $PID2 || true
    wait $PID3 || true
    wait $PID4 || true
    log "  Full training done."
}

# Run a complete experiment (sweep + full training)
run_experiment() {
    local CONF_SP=$1
    local CONF_MUP=$2
    local TRAIN_SCRIPT=$3
    local LOG_DIR=$4
    local PREFIX=$5

    mkdir -p "$LOG_DIR"

    log "--- Experiment: $PREFIX ---"

    # LR sweeps (4 conditions)
    run_sweep "$CONF_SP" "false" "${PREFIX}-sp-base" "$TRAIN_SCRIPT" "$LOG_DIR"
    run_sweep "$CONF_SP" "true" "${PREFIX}-sp-deq" "$TRAIN_SCRIPT" "$LOG_DIR"
    run_sweep "$CONF_MUP" "false" "${PREFIX}-mup-base" "$TRAIN_SCRIPT" "$LOG_DIR"
    run_sweep "$CONF_MUP" "true" "${PREFIX}-mup-deq" "$TRAIN_SCRIPT" "$LOG_DIR"

    # Full training
    run_full_training "$CONF_SP" "$CONF_MUP" "$TRAIN_SCRIPT" "$LOG_DIR" "$PREFIX"

    log "--- Experiment $PREFIX complete ---"
}

# ============================================================
log "=== D3 Master Orchestration Started ==="

# ============================================================
# Phase 0: Wait for INR generation to complete
# ============================================================
log "Phase 0: Waiting for INR generation screen sessions to finish..."

wait_for_screen() {
    local name=$1
    while screen -ls 2>/dev/null | grep -q "$name"; do
        sleep 60
    done
    log "  $name finished."
}

wait_for_screen "gen_mnist_sp_d3"
wait_for_screen "gen_mnist_mup_d3"
wait_for_screen "gen_fmnist_sp_d3"
wait_for_screen "gen_fmnist_mup_d3"

log "Phase 0 complete: All INR generation done."

# Verify key directories exist
for w in 16 24 32 48 64 80 96 128 192; do
    for s in sp_d3 mup_d3; do
        if [ ! -d "$ANYDIM_DATA_ROOT/mnist_inrs/w${w}_${s}" ]; then
            log "ERROR: Missing $ANYDIM_DATA_ROOT/mnist_inrs/w${w}_${s}"
            exit 1
        fi
    done
done
for w in 16 24 32 40 48 64 80 96 128 192; do
    for s in sp_d3 mup_d3; do
        if [ ! -d "$ANYDIM_DATA_ROOT/fmnist_inrs/w${w}_${s}" ]; then
            log "ERROR: Missing $ANYDIM_DATA_ROOT/fmnist_inrs/w${w}_${s}"
            exit 1
        fi
    done
done
log "All INR directories verified."

# ============================================================
# Phase 1: Generate sizegen splits
# ============================================================
log "Phase 1: Generating sizegen splits..."

$PYTHON scripts/generate_sizegen_splits.py --dataset mnist --init-type sp_d3 \
    --train-width 16 --test-widths 24 32 48 64 80 96 128 192 \
    --split-name mnist_sizegen_splits.json 2>&1 | tee -a $LOG

$PYTHON scripts/generate_sizegen_splits.py --dataset mnist --init-type mup_d3 \
    --train-width 16 --test-widths 24 32 48 64 80 96 128 192 \
    --split-name mnist_sizegen_splits.json 2>&1 | tee -a $LOG

$PYTHON scripts/generate_sizegen_splits.py --dataset fmnist --init-type sp_d3 \
    --train-width 16 --test-widths 24 32 40 48 64 80 96 128 192 \
    --split-name fmnist_sizegen_splits.json 2>&1 | tee -a $LOG

$PYTHON scripts/generate_sizegen_splits.py --dataset fmnist --init-type mup_d3 \
    --train-width 16 --test-widths 24 32 40 48 64 80 96 128 192 \
    --split-name fmnist_sizegen_splits.json 2>&1 | tee -a $LOG

log "Phase 1 complete: All splits generated."

# ============================================================
# Phase 2: Generate duplicated INRs (for sanity check later)
# ============================================================
log "Phase 2: Generating duplicated INRs..."

for W in 32 48 64 80 96 128 192; do
    $PYTHON scripts/generate_duplicated_inrs.py --init-type sp_d3 --base-width 16 \
        --target-widths $W --dataset mnist 2>&1 | tee -a $LOG
    $PYTHON scripts/generate_duplicated_inrs.py --init-type mup_d3 --base-width 16 \
        --target-widths $W --dataset mnist 2>&1 | tee -a $LOG
done

log "Phase 2 complete: Duplicated INRs generated."

# ============================================================
# Phase 3: Training experiments (ScaleGMN only)
# ============================================================
log "Phase 3: Starting training experiments (ScaleGMN only, max $MAX_GPUS GPUs)..."

run_experiment \
    "configs/mnist_cls/scalegmn_sizegen_sp_d3.yml" \
    "configs/mnist_cls/scalegmn_sizegen_mup_d3.yml" \
    "train_sizegen.py" \
    "/tmp/d3_mnist_cls" \
    "d3-mnist-cls"

run_experiment \
    "configs/mnist_cls/scalegmn_sizegen_regression_sp_d3.yml" \
    "configs/mnist_cls/scalegmn_sizegen_regression_mup_d3.yml" \
    "train_sizegen_regression.py" \
    "/tmp/d3_mnist_reg" \
    "d3-mnist-reg"

run_experiment \
    "configs/fmnist_cls/scalegmn_sizegen_sp_d3.yml" \
    "configs/fmnist_cls/scalegmn_sizegen_mup_d3.yml" \
    "train_sizegen.py" \
    "/tmp/d3_fmnist_cls" \
    "d3-fmnist-cls"

log ""
log "================================================================"
log "=== ALL D3 EXPERIMENTS COMPLETE: $(date) ==="
log "=== Check wandb projects mnist_cls / fmnist_cls for results ==="
log "=== Master log: $LOG ==="
log "================================================================"

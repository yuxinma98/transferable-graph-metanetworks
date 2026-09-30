#!/bin/bash
# RETRAIN of every bidirectional ScaleGMN condition on Fashion-MNIST, now that the backward layers
# actually get the reciprocal edge feature 1/W[u,v] that `reciprocal: True` asks for
# (src/models/bidir_reciprocal.py, installed by train_sizegen.py / train_scalegmn.py).
#
# Without it a bidirectional `symmetry: scale` model is NOT scale-equivariant -- the
# property the model family is built around -- so every previously recorded bidirectional
# ScaleGMN number was produced by a model that silently lacked it. See
# results/VERIFICATION_gmn_properties.md, "Verification of scale equivariance", Test 4.
# Nothing else changes: same configs, same LR grid, same epochs/patience, same splits, so
# the new numbers are drop-in replacements in the existing tables.
#
# Five jobs (the affected rows of results/EXPERIMENTS_FMNIST_INR_classification.md):
#   sizegen   ScaleGMN x {SP, muP32} x {baseline, dup-equiv}, train w32        (Experiment 3)
#             = 7-LR sweep -> 200 epochs (v2, test w32-w256) -> eval-only re-score (v2_all, w32-w1024)
#   reproduce ScaleGMN-B on the w32 55k/5k/10k benchmark                       (Experiment 1)
#             = single training at the config LR, exactly as the original run. NOTE: the INRs
#               this reads are NOT the ones run aqf50z1c used -- those w32_sp fits were deleted
#               on 2026-08-19 and the size-gen v2 rebuild refitted w32 with a swept LR, and
#               fmnist_splits.json had to be regenerated (scripts/generate_splits.py, seed 42).
#               So this row changes for two reasons at once; the doc says so.
#
# Plain GMN / matrix-product GMN bidirectional rows are unaffected (symmetry: permutation,
# where the reciprocal flag is a no-op by construction), and mpsgmn is forward-only.
#
# The jobs are pushed onto a FIFO queue drained by one worker per GPU. Re-running is safe
# and is also how you ADD a worker: the queue file is only created if missing, finished
# sweep LRs / trainings / evals are detected in the logs and skipped, and pops are
# serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Launch:
#   screen -dmS fmnist_recip bash -c 'bash scripts/run_fmnist_cls_scalegmn_bidir_recip.sh "2"'
# Add a worker once a GPU of ours frees up:
#   screen -dmS fmnist_recip_w2 bash -c 'bash scripts/add_worker_when_free.sh 4 \
#       scripts/run_fmnist_cls_scalegmn_bidir_recip.sh'
# Results:  /tmp/sizegen_v3_bidir_recip/SUMMARY.md   (appended as each job finishes)
# Logs:     /tmp/sizegen_v3_bidir_recip/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/fmnist_sizegen_v2_bidir_recip
SUMMARY=$LOG_DIR/SUMMARY.md
GPUS=${1-}                # one worker per GPU; empty = seed the queue only.
                          # NOT ${1:-...}: that substitutes a default for an empty arg too,
                          # which would silently start a worker on some hardcoded GPU.
WANDB_PROJECT=fmnist_cls
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

if [ -f "$LOG_DIR/HOLD" ]; then
    echo "HOLD file present ($LOG_DIR/HOLD) -- exiting."
    exit 0
fi

# ---------------------------------------------------------------- helpers

run_sweep() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    for idx in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
        if grep -q "Best val acc:" "$LOG" 2>/dev/null; then
            echo "[$PREFIX] sweep lr=${LRS[$idx]} already done, skipping"
            continue
        fi
        echo "[$PREFIX] sweep lr=${LRS[$idx]} on GPU $GPU ($(date +%H:%M))"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
            --conf "$CONF" \
            --duplication-equiv "$DEQ" \
            --direction bidirectional \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-fmnist-v2-recip-${PREFIX}-${LRS[$idx]}" \
            > "$LOG" 2>&1
    done
}

get_best_lr() {
    local PREFIX=$1
    local BEST_ACC=0
    local BEST_LR="2.5e-4"
    for i in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${i}.log"
        [ -f "$LOG" ] || continue
        local ACC=$(grep "Best val acc:" "$LOG" | tail -1 | awk '{print $NF}')
        [ -z "$ACC" ] && continue
        local BETTER=$(python3 -c "print(1 if $ACC > $BEST_ACC else 0)")
        if [ "$BETTER" = "1" ]; then BEST_ACC=$ACC; BEST_LR=${LRS[$i]}; fi
    done
    echo "$BEST_LR"
}

sweep_accs() {
    local PREFIX=$1
    local OUT=""
    for i in 0 1 2 3 4 5 6; do
        local ACC=$(grep "Best val acc:" "${LOG_DIR}/sweep_${PREFIX}_${i}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [ -z "$ACC" ] && ACC="--"
        OUT="${OUT}${ACC} / "
    done
    echo "${OUT%" / "}"
}

run_full() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4 LR=$5
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    if grep -q "Test acc/loss @ best val:" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] full training already done, skipping"
        return
    fi
    echo "[$PREFIX] full training lr=$LR on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
        --conf "$CONF" \
        --duplication-equiv "$DEQ" \
        --direction bidirectional \
        --wandb True --lr "$LR" \
        --run_name "sizegen-fmnist-v2-recip-${PREFIX}" \
        > "$LOG" 2>&1
}

run_large_eval() {
    local ALL_CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    local LOG="${LOG_DIR}/eval_all_${PREFIX}.log"
    if grep -q "mean OOD acc" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] w32-w1024 eval already done, skipping"
        return
    fi
    local CKPT_DIR=$(grep -h "Checkpoints will be saved to:" "${LOG_DIR}/full_${PREFIX}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX] no checkpoint found for the w32-w1024 eval (looked for '${CKPT_DIR}/best.pt')"
        return
    fi
    echo "[$PREFIX] eval w32-w1024 on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
        --conf "$ALL_CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        --duplication-equiv "$DEQ" \
        --direction bidirectional \
        --wandb True --run_name "sizegen-fmnist-v2-eval-recip-${PREFIX}" \
        > "$LOG" 2>&1
}

append_summary() {
    local LABEL=$1 PREFIX=$2 LR=$3
    local FULL_LOG="${LOG_DIR}/full_${PREFIX}.log"
    local EVAL_LOG="${LOG_DIR}/eval_all_${PREFIX}.log"
    local CKPT_DIR=$(grep -h "Checkpoints will be saved to:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local BEST_VAL=$(grep "Best val acc:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local EPOCHS=$(grep -c "val_acc=" "$FULL_LOG" 2>/dev/null)
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$EVAL_LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- best LR: **$LR** (sweep val accs @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_accs "$PREFIX"))"
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "- checkpoint: \`$(basename "${CKPT_DIR:-n/a}")\`"
        echo "- w32-w1024 eval run: \`${EVAL_ID:-n/a}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- best val acc: ${BEST_VAL:-n/a}, epochs run: ${EPOCHS:-n/a}"
        echo "- logs: \`${FULL_LOG}\`, \`${EVAL_LOG}\`"
        echo ""
        echo "| Width | Test acc (1k disjoint images/width) |"
        echo "|---|---|"
        sed -n '/Per-width test accuracy:/,$p' "$EVAL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +acc=([0-9.]+).*/| w\1 (\2) | \3 |/'
        echo ""
        echo "OOD mean (w48-w1024): $(grep "mean OOD acc" "$EVAL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')"
    } >> "$SUMMARY.part.$PREFIX"
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.$PREFIX' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.$PREFIX"
}

# ------------------------------------------------- Experiment 1 (reproduce) job
# Goes through scripts/train_scalegmn.py, which runs upstream inr_classification.py with
# the reciprocal patch bootstrapped in. Upstream logs its accuracies to wandb only, so the
# summary reads val/best_acc and test/best_acc back through the wandb API.

run_reproduce() {
    local CONF=$1 PREFIX=$2 GPU=$3
    local LOG="${LOG_DIR}/reproduce_${PREFIX}.log"
    # upstream prints no completion marker, so record one ourselves for resumability
    if [ -f "${LOG_DIR}/reproduce_${PREFIX}.done" ]; then
        echo "[$PREFIX] reproduce training already done, skipping"
        return
    fi
    echo "[$PREFIX] reproduce training on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_scalegmn.py \
        --conf "$CONF" --wandb True \
        > "$LOG" 2>&1
    touch "${LOG_DIR}/reproduce_${PREFIX}.done"
}

append_reproduce_summary() {
    local LABEL=$1 PREFIX=$2 CONF=$3
    local LOG="${LOG_DIR}/reproduce_${PREFIX}.log"
    local RUN_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    local ACCS=$($PYTHON - "$WANDB_PROJECT" "$RUN_ID" <<'PY' 2>/dev/null
import sys
try:
    import wandb
    r = wandb.Api().run(f"yuxinma/{sys.argv[1]}/{sys.argv[2]}")
    print(f"{r.summary.get('val/best_acc', float('nan')):.4f} {r.summary.get('test/best_acc', float('nan')):.4f}")
except Exception:
    print("n/a n/a")
PY
)
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- config: \`$CONF\` (LR from the config, no sweep -- same protocol as the original run)"
        echo "- run: \`${RUN_ID:-n/a}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "- best val acc / best test acc: ${ACCS}"
        echo "- log: \`${LOG}\`"
    } >> "$SUMMARY.part.$PREFIX"
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.$PREFIX' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.$PREFIX"
}

# ---------------------------------------------------------------- job queue

pop_job() {
    flock "$QUEUE_LOCK" -c "
        JOB=\$(head -1 '$QUEUE' 2>/dev/null)
        if [ -n \"\$JOB\" ]; then tail -n +2 '$QUEUE' > '$QUEUE.tmp' && mv '$QUEUE.tmp' '$QUEUE'; fi
        printf '%s' \"\$JOB\""
}

worker() {
    local GPU=$1
    while true; do
        local SPEC=$(pop_job)
        [ -z "$SPEC" ] && break
        IFS='|' read -r TYPE CONF EVAL_CONF DEQ PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        if [ "$TYPE" = "sizegen" ]; then
            run_sweep "$CONF" "$DEQ" "$PREFIX" "$GPU"
            local LR=$(get_best_lr "$PREFIX")
            echo "[$PREFIX] best LR = $LR"
            run_full "$CONF" "$DEQ" "$PREFIX" "$GPU" "$LR"
            run_large_eval "$EVAL_CONF" "$DEQ" "$PREFIX" "$GPU"
            append_summary "$LABEL" "$PREFIX" "$LR"
        else
            run_reproduce "$CONF" "$PREFIX" "$GPU"
            append_reproduce_summary "$LABEL" "$PREFIX" "$CONF"
        fi
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    C=configs/fmnist_cls
    # <type>|<train conf>|<eval conf>|<dup-equiv>|<prefix>|<label>
    # muP32 first (the controlled comparison: flat INR quality out to w1024), dup-equiv
    # before baseline; the cheap Experiment 1 rerun goes last.
    cat > "$QUEUE" <<EOF
sizegen|$C/scalegmn_sizegen_mup32_v2.yml|$C/scalegmn_sizegen_mup32_v2_all.yml|True|sgmn-mup32-deq-bidir-recip|ScaleGMN, muP32, dup-equiv, bidirectional (reciprocal fix)
sizegen|$C/scalegmn_sizegen_mup32_v2.yml|$C/scalegmn_sizegen_mup32_v2_all.yml|False|sgmn-mup32-base-bidir-recip|ScaleGMN, muP32, baseline, bidirectional (reciprocal fix)
sizegen|$C/scalegmn_sizegen_sp_v2.yml|$C/scalegmn_sizegen_sp_v2_all.yml|True|sgmn-sp-deq-bidir-recip|ScaleGMN, SP, dup-equiv, bidirectional (reciprocal fix)
sizegen|$C/scalegmn_sizegen_sp_v2.yml|$C/scalegmn_sizegen_sp_v2_all.yml|False|sgmn-sp-base-bidir-recip|ScaleGMN, SP, baseline, bidirectional (reciprocal fix)
reproduce|$C/scalegmn_reproduce_sp_bidir.yml|-|-|exp1-sgmn-b-recip|Experiment 1: ScaleGMN-B (bidirectional, scale), w32 55k/5k/10k (reciprocal fix, refitted INRs)
EOF
    {
        echo "# Fashion-MNIST bidirectional ScaleGMN — RETRAIN with the reciprocal backward edge feature"
        echo ""
        echo "Started: $(date)."
        echo "Every bidirectional ScaleGMN run recorded in"
        echo "results/EXPERIMENTS_FMNIST_INR_classification.md was trained without the reciprocal"
        echo "backward edge feature \`reciprocal: True\` asks for, so it was not scale-equivariant"
        echo "(results/VERIFICATION_gmn_properties.md, Test 4). These runs install"
        echo "src/models/bidir_reciprocal.py and change nothing else: same configs, LR grid,"
        echo "epochs/patience and splits, so the Experiment 3 rows are drop-in replacements."
        echo ""
        echo "Four Experiment 3 conditions (ScaleGMN × {SP, muP32} × {baseline, dup-equiv}; sweep →"
        echo "200 epochs testing w32–w256 → eval-only re-score over w32–w1024) plus the Experiment 1"
        echo "ScaleGMN-B rerun at w32. The Experiment 1 row also changes data: its original w32_sp"
        echo "fits were deleted on 2026-08-19 and refitted with a swept LR by the size-gen v2"
        echo "rebuild, and fmnist_splits.json was regenerated (generate_splits.py, seed 42)."
        echo "Jobs appear as they finish; \"Finished:\" at the bottom means all five are done."
    } > $SUMMARY
    echo "=== Seeded queue with 5 jobs ==="
fi

echo "=== FMNIST bidirectional ScaleGMN retrain: starting workers on GPUs [$GPUS] $(date) ==="

PIDS=()
for GPU in $GPUS; do
    worker "$GPU" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

if [ ! -s "$QUEUE" ]; then
    echo "" >> $SUMMARY
    echo "Finished: $(date)" >> $SUMMARY
fi

echo "=== Workers done $(date). Summary: $SUMMARY ==="

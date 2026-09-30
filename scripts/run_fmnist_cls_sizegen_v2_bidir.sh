#!/bin/bash
# Fashion-MNIST Experiment 3 (size-gen v2), BIDIRECTIONAL variants: train w32,
# test w32-w256 during training and w32-w1024 as an eval-only re-score.
#
# Ten conditions, the exact bidirectional counterpart of the forward runs in
# results/EXPERIMENTS_FMNIST_INR_classification.md (Experiment 3), so they become new
# columns in the same per-width tables:
#   {ScaleGMN, plain GMN} x {SP, muP32} x {baseline, dup-equiv}   = 8
#   matrix-product GMN x {SP, muP32}   (dup-equiv only by construction)  = 2
#
# Protocol per condition, identical to the forward FMNIST runs and to MNIST bidir:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), 8600 steps each,
#      selected on best val accuracy at the training width w32
#   2. full training (200 epochs, patience 50) at the best sweep LR, testing w32-w256
#   3. eval-only re-score of that checkpoint on w32-w1024 (`*_v2_all.yml`)
#
# Both stages read the same fmnist_sizegen_splits_v2.json (56k train / 1k val at w32,
# 1k disjoint test images per width), so no split juggling is needed.
#
# The ten conditions are pushed onto a FIFO queue drained by one worker per GPU; each
# worker takes the next condition and runs sweep -> full training -> w1024 eval ->
# summary append, so results land progressively. muP32 is queued first (the controlled
# comparison: flat INR quality across widths), dup-equiv before baseline within each
# parameterization so the headline contrast completes early.
#
# Re-running is safe and is also how you ADD a worker: the queue file is only created
# if missing, finished sweep LRs / full trainings / evals are detected in the logs and
# skipped, and pops are serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Expect ~15-20 h per condition, so ten conditions on one GPU is multi-day; add GPUs as
# they free up rather than waiting.
#
# Launch:
#   screen -dmS fmnist_bidir bash -c 'bash scripts/run_fmnist_cls_sizegen_v2_bidir.sh "1"'
# Add a worker later, once another GPU frees up:
#   screen -dmS fmnist_bidir_w2 bash -c 'bash scripts/run_fmnist_cls_sizegen_v2_bidir.sh "4"'
#   # or have it wait for a GPU of yours to go idle first:
#   screen -dmS fmnist_bidir_w2 bash -c 'bash scripts/add_worker_when_free.sh 2 \
#       scripts/run_fmnist_cls_sizegen_v2_bidir.sh'
# Results:  /tmp/fmnist_sizegen_v2_bidir/SUMMARY.md   (appended as each condition finishes)
# Logs:     /tmp/fmnist_sizegen_v2_bidir/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/fmnist_sizegen_v2_bidir
SUMMARY=$LOG_DIR/SUMMARY.md
GPUS=${1:-"1"}            # one worker per GPU; keep >=2 GPUs vacant for other users
WANDB_PROJECT=fmnist_cls
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

# Hold switch: while $LOG_DIR/HOLD exists this runner refuses to start, so a caller that
# launches it unconditionally (scripts/orchestrate_cnn_first.sh step 3) becomes a no-op.
# Release with: rm /tmp/fmnist_sizegen_v2_bidir/HOLD
if [ -f "$LOG_DIR/HOLD" ]; then
    echo "HOLD file present ($LOG_DIR/HOLD) -- bidirectional runs are on hold, exiting."
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
            --run_name "lr-sweep-fmnist-v2-${PREFIX}-${LRS[$idx]}" \
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
    # compact inline list of the 7 sweep val accs
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
        --run_name "sizegen-fmnist-v2-${PREFIX}" \
        > "$LOG" 2>&1
}

run_large_eval() {
    # Re-score the checkpoint just trained on the full width range (w32-w1024).
    # Same split file as training, so train/val are untouched and no image the model
    # trained on appears in any test set.
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
        --wandb True --run_name "sizegen-fmnist-v2-eval-${PREFIX}" \
        > "$LOG" 2>&1
}

append_summary() {
    # $1 = human label, $2 = prefix, $3 = best lr
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
        # the eval-only path prints "Per-width test accuracy:"; widths and the IN/OUT
        # tag are space-padded ("w=   32 (IN ):"), hence the loose whitespace matching
        sed -n '/Per-width test accuracy:/,$p' "$EVAL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +acc=([0-9.]+).*/| w\1 (\2) | \3 |/'
        echo ""
        echo "OOD mean (w48-w1024): $(grep "mean OOD acc" "$EVAL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')"
    } >> "$SUMMARY.part.$PREFIX"
    # serialize the append so concurrent workers don't interleave
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.$PREFIX' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.$PREFIX"
}

# ---------------------------------------------------------------- job queue

pop_job() {
    # atomically remove and echo the first queue line ("" if empty)
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
        IFS='|' read -r CONF ALL_CONF DEQ PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        run_sweep "$CONF" "$DEQ" "$PREFIX" "$GPU"
        local LR=$(get_best_lr "$PREFIX")
        echo "[$PREFIX] best LR = $LR"
        run_full "$CONF" "$DEQ" "$PREFIX" "$GPU" "$LR"
        run_large_eval "$ALL_CONF" "$DEQ" "$PREFIX" "$GPU"
        append_summary "$LABEL" "$PREFIX" "$LR"
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    C=configs/fmnist_cls
    # <train conf>|<w32-w1024 conf>|<dup-equiv>|<prefix>|<label>
    cat > "$QUEUE" <<EOF
$C/scalegmn_sizegen_mup32_v2.yml|$C/scalegmn_sizegen_mup32_v2_all.yml|True|sgmn-mup32-deq-bidir|ScaleGMN, muP32, dup-equiv, bidirectional
$C/gmn_sizegen_mup32_v2.yml|$C/gmn_sizegen_mup32_v2_all.yml|True|gmn-mup32-deq-bidir|Plain GMN, muP32, dup-equiv, bidirectional
$C/scalegmn_sizegen_mup32_v2.yml|$C/scalegmn_sizegen_mup32_v2_all.yml|False|sgmn-mup32-base-bidir|ScaleGMN, muP32, baseline, bidirectional
$C/gmn_sizegen_mup32_v2.yml|$C/gmn_sizegen_mup32_v2_all.yml|False|gmn-mup32-base-bidir|Plain GMN, muP32, baseline, bidirectional
$C/mpgmn_sizegen_mup32_v2.yml|$C/mpgmn_sizegen_mup32_v2_all.yml|True|mpgmn-mup32-bidir|Matrix-product GMN, muP32, dup-equiv, bidirectional
$C/scalegmn_sizegen_sp_v2.yml|$C/scalegmn_sizegen_sp_v2_all.yml|True|sgmn-sp-deq-bidir|ScaleGMN, SP, dup-equiv, bidirectional
$C/gmn_sizegen_sp_v2.yml|$C/gmn_sizegen_sp_v2_all.yml|True|gmn-sp-deq-bidir|Plain GMN, SP, dup-equiv, bidirectional
$C/scalegmn_sizegen_sp_v2.yml|$C/scalegmn_sizegen_sp_v2_all.yml|False|sgmn-sp-base-bidir|ScaleGMN, SP, baseline, bidirectional
$C/gmn_sizegen_sp_v2.yml|$C/gmn_sizegen_sp_v2_all.yml|False|gmn-sp-base-bidir|Plain GMN, SP, baseline, bidirectional
$C/mpgmn_sizegen_sp_v2.yml|$C/mpgmn_sizegen_sp_v2_all.yml|True|mpgmn-sp-bidir|Matrix-product GMN, SP, dup-equiv, bidirectional
EOF
    {
        echo "# Fashion-MNIST size generalization v2 (train w32, test w32–w1024), BIDIRECTIONAL models"
        echo ""
        echo "Started: $(date)."
        echo "Ten conditions: {ScaleGMN, plain GMN} × {SP, muP32} × {baseline, dup-equiv},"
        echo "plus the matrix-product GMN × {SP, muP32} (dup-equiv only by construction),"
        echo "all \`--direction bidirectional\`. Same LR grid / epochs / patience / splits as the"
        echo "forward FMNIST Experiment 3 runs, so these are drop-in columns in the same tables."
        echo "Per-width accuracies come from the eval-only re-score over w32–w1024 (1k disjoint"
        echo "images per width); the training logs also carry the w32–w256 numbers per epoch."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all ten are done."
    } > $SUMMARY
    echo "=== Seeded queue with 10 conditions ==="
fi

echo "=== FMNIST size-gen v2 (bidirectional): starting workers on GPUs [$GPUS] $(date) ==="

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

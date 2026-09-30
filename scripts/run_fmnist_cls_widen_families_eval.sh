#!/bin/bash
# FMNIST Experiment 2 (widened-INR sanity check), scored over WIDENING FAMILIES x all 11
# conditions. The FMNIST counterpart of run_mnist_cls_widen_families_eval.sh, with w32 as the
# base width (the size-gen v2 training width) instead of MNIST's w24.
#
# No training: every job re-scores existing size-gen v2 checkpoints (train w32) on the w32
# test INRs widened to w64/w128/w192/w256/w320 (k = 2, 4, 6, 8, 10, as on MNIST). Two families:
#   general  (w{N}_{init}_gen/) -- the largest function-preserving widening (row condition
#            only). Only the FORWARD matrix-product models are invariant to it.
#   uniform  (w{N}_{init}_dup/) -- the Kronecker widening (row + col), the family the
#            edgewise-MLP GMNs' fan-in rescaling + mean aggregation is built for; the
#            per-condition reference column.
#
# 8 jobs = {general, uniform} x {mup32, sp} x {forward, bidirectional}. Each job runs
# `--model-type all`, so it scores 6 conditions forward / 5 bidirectional (mp-ScaleGMN is
# forward-only in every experiment: it is skipped and reported as out of scope, not missing).
# 11 conditions per (family, init) arm, 44 condition-evaluations in all.
#
# Data prerequisite (regenerate with; note the v2 split filename):
#   python scripts/generate_duplicated_inrs.py --dataset fmnist --init-type {mup32,sp} \
#       --base-width 32 --target-widths 64 128 192 256 320 --family {general,uniform} \
#       --split-name fmnist_sizegen_splits_v2.json
#
# Launch (fully detached, survives disconnect):
#   screen -dmS fmnist_widen_eval bash -c 'bash scripts/run_fmnist_cls_widen_families_eval.sh "0"'
# Results:  /tmp/fmnist_widen_families_eval/SUMMARY.md   (appended as each job finishes)
# Logs:     /tmp/fmnist_widen_families_eval/eval_<family>_<init>_<direction>.log
# Done:     /tmp/fmnist_widen_families_eval/DONE         (touched once the queue drains)
#
# Re-running is safe: finished jobs are detected in their logs and skipped, and re-running
# with a different GPU list ADDS workers rather than duplicating work. Delete
# /tmp/fmnist_widen_families_eval/queue.txt to hard-reset the queue.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/fmnist_widen_families_eval
SUMMARY=$LOG_DIR/SUMMARY.md
GPUS=${1-}                     # empty first arg seeds the queue and starts no worker
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock
SUMMARY_LOCK=$LOG_DIR/summary.lock

# ---------------------------------------------------------------- job list
# <family>|<init>|<direction>
# `general` first: it is the result being asked for; `uniform` is the reference column.
JOBS="general|mup32|forward
general|mup32|bidirectional
general|sp|forward
general|sp|bidirectional
uniform|mup32|forward
uniform|mup32|bidirectional
uniform|sp|forward
uniform|sp|bidirectional"

N_JOBS=$(echo "$JOBS" | grep -c .)

# ---------------------------------------------------------------- helpers

job_done() {
    # A job is finished iff its log carries the END_MD terminator of the markdown block,
    # which the scorer prints only after every condition has been scored.
    grep -q '^END_MD$' "$1" 2>/dev/null
}

append_summary() {
    local FAMILY=$1 INIT=$2 DIRECTION=$3 LOG=$4
    local RUN_ID
    RUN_ID=$(grep '^WANDB_RUN_ID:' "$LOG" | tail -1 | awk '{print $2}')
    (
        flock 9
        {
            echo ""
            echo "### \`$FAMILY\` widening — $INIT, $DIRECTION"
            echo ""
            echo "Eval run: [\`$RUN_ID\`](https://wandb.ai/yuxinma/fmnist_cls/runs/$RUN_ID)"
            echo ""
            sed -n '/^BEGIN_MD$/,/^END_MD$/p' "$LOG" | grep -vx 'BEGIN_MD\|END_MD'
            if grep -q 'forward-only in every experiment' "$LOG"; then
                echo ""
                echo "mp-ScaleGMN is forward-only in every experiment — no bidirectional"
                echo "checkpoint exists, so that cell is out of scope rather than missing."
            fi
        } >> "$SUMMARY"
    ) 9>"$SUMMARY_LOCK"
}

run_job() {
    local FAMILY=$1 INIT=$2 DIRECTION=$3 GPU=$4
    local TAG="${FAMILY}_${INIT}_${DIRECTION}"
    local LOG="${LOG_DIR}/eval_${TAG}.log"

    if job_done "$LOG"; then
        echo "[$TAG] already done, skipping"
        return
    fi
    echo "[$TAG] scoring on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_sizegen_duplicated.py \
        --dataset fmnist --init-type "$INIT" --model-type all --direction "$DIRECTION" \
        --family "$FAMILY" --base-width 32 --target-widths 64 128 192 256 320 \
        --wandb True > "$LOG" 2>&1

    if job_done "$LOG"; then
        append_summary "$FAMILY" "$INIT" "$DIRECTION" "$LOG"
        echo "[$TAG] done, appended to SUMMARY ($(date +%H:%M))"
    else
        echo "[$TAG] FAILED -- see $LOG"
    fi
}

# ---------------------------------------------------------------- queue

seed_queue() {
    (
        flock 9
        [ -f "$QUEUE" ] || echo "$JOBS" | grep . > "$QUEUE"
    ) 9>"$QUEUE_LOCK"
    echo "Queue: $(grep -c . "$QUEUE") of $N_JOBS jobs pending in $QUEUE"
}

pop_job() {
    (
        flock 9
        head -1 "$QUEUE"
        sed -i '1d' "$QUEUE"
    ) 9>"$QUEUE_LOCK"
}

worker() {
    local GPU=$1
    while true; do
        local SPEC
        SPEC=$(pop_job)
        [ -z "$SPEC" ] && break
        IFS='|' read -r FAMILY INIT DIRECTION <<< "$SPEC"
        run_job "$FAMILY" "$INIT" "$DIRECTION" "$GPU"
    done
    echo "[gpu $GPU] queue drained ($(date +%H:%M))"
}

# ---------------------------------------------------------------- main

if [ ! -f "$SUMMARY" ]; then
    {
        echo "# FMNIST Experiment 2 — widening families x all 11 conditions"
        echo ""
        echo "No training: size-gen v2 checkpoints (train w32) re-scored on the w32 test INRs"
        echo "widened to w64-w320. Every widening is function-preserving (verified to float32"
        echo "rounding by \`generate_duplicated_inrs.py --verify\`), so an accuracy drop is the"
        echo "metanetwork failing to be invariant to that family, never distribution shift."
        echo ""
        echo "- \`general\`: row condition only — the largest function-preserving family."
        echo "- \`uniform\`: Kronecker (row + col) — the per-condition reference."
        echo ""
        echo "Started $(date)."
    } > "$SUMMARY"
fi

seed_queue

if [ -z "$GPUS" ]; then
    echo "No GPU list given -- queue seeded, no worker started."
    exit 0
fi

echo "=== FMNIST widening-family eval: starting workers on GPUs [$GPUS] $(date) ==="
for GPU in $GPUS; do
    worker "$GPU" &
done
wait

# Drain marker, so the state is readable without reattaching to the screen.
{
    echo ""
    echo "---"
    echo ""
    echo "Queue drained $(date). Jobs whose log lacks END_MD failed; grep the logs in"
    echo "$LOG_DIR for tracebacks and re-run this script to retry only those."
} >> "$SUMMARY"
touch "$LOG_DIR/DONE"
echo "=== all workers finished $(date); see $SUMMARY ==="

#!/bin/bash
# Experiment 2 of the CNN accuracy-prediction study (widening sanity check), scored over
# WIDENING FAMILIES x all 9 conditions, for either CNN zoo.
#
# Every widened CNN in $ANYDIM_DATA_ROOT/{cnn_zoo,svhn_cnn_zoo}/w{32..128}_zoo_{dup,gen}/
# computes *exactly the same function* as its w16 original and carries the original's test
# accuracy as its label (scripts/generate_duplicated_cnns.py), so the question is
# confound-free: does the prediction move when the same function is presented at a wider
# architecture? Two families:
#   general  (w{N}_zoo_gen/) -- the row condition only, i.e. the LARGEST function-preserving
#            widening. Only the forward matrix-product models are invariant to it.
#   uniform  (w{N}_zoo_dup/) -- the channel-Kronecker widening (row + col), the family the
#            dup-equiv modifications (fan-in rescaling + mean aggregation + a
#            width-agnostic readout) are built for; the per-condition reference column.
#
# 9 conditions per dataset, all trained at w16 only:
#   forward:        ScaleGMN {baseline, dup-equiv}, plain GMN {baseline, dup-equiv},
#                   mp-GMN dup-equiv, mp-ScaleGMN dup-equiv                        (6)
#   bidirectional:  plain GMN {baseline, dup-equiv}, mp-GMN dup-equiv              (3)
# The matrix-product models have no baseline variant (the MSG constraint is defined on top of
# fan-in rescaling + mean aggregation); mp-ScaleGMN is forward-only in every experiment; and
# the 4 bidirectional `symmetry: scale` cells are OUT OF SCOPE on CNN graphs (the reciprocal
# backward edge feature overflows float32 on this zoo -- see
# src/models/bidir_reciprocal.py).
#
# Phase 1 trains the 5 conditions the original 4-condition Stage 2 never covered
# (mp-GMN fw, mp-ScaleGMN fw, plain GMN bd x2, mp-GMN bd); the 4 forward {ScaleGMN, plain
# GMN} x {baseline, dup-equiv} checkpoints already exist and are reused. No LR sweep: this is
# a property check, not a performance comparison, so every condition uses the config's LR
# (1e-3, as in Experiment 3). Model selection is on val Kendall tau at w16, the only width
# trained on.
# Phase 2 runs 4 eval jobs = {general, uniform} x {forward, bidirectional}, each with
# `--model-type all`, and resolves checkpoints by run-name prefix, so nothing is hardcoded.
#
# Data prerequisite (uniform is already on disk; the general tree is written by):
#   python scripts/generate_duplicated_cnns.py [--dataset svhn] --family general
#
# Launch (fully detached, survives disconnect):
#   screen -dmS widen_cnn bash -c 'bash scripts/run_predgen_widen_families.sh cifar10 "0 1 2"'
# Results:  /tmp/predgen_{cifar10,svhn}_widen_families/SUMMARY.md  (appended per job)
# Logs:     same dir: train_<tag>.log, eval_<family>_<direction>.log
# Done:     same dir: DONE  (touched once both phases drain)
#
# Re-running is safe and is how you ADD a worker: the queue files are created only if
# missing, and finished trainings and evals are detected in their logs and skipped. Delete
# $LOG_DIR/queue_{train,eval}.txt to hard-reset.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

DATASET=${1:?usage: run_predgen_widen_families.sh <cifar10|svhn> "<gpu list>"}
GPUS=${2-}                     # empty second arg seeds the queues and starts no worker

PYTHON=${PYTHON:-python}
WIDTHS="16 32 48 64 96 128"

case "$DATASET" in
    cifar10) CONF_DIR=configs/cifar10_predgen; RUN_PREFIX=predgen-dup
             WANDB_PROJECT=cifar10_predgen; LOG_DIR=/tmp/predgen_cifar10_widen_families ;;
    svhn)    CONF_DIR=configs/svhn_predgen;    RUN_PREFIX=svhn-predgen-dup
             WANDB_PROJECT=svhn_predgen;       LOG_DIR=/tmp/predgen_svhn_widen_families ;;
    *) echo "unknown dataset: $DATASET (expected cifar10 or svhn)"; exit 1 ;;
esac

SUMMARY=$LOG_DIR/SUMMARY.md
mkdir -p "$LOG_DIR"
TRAIN_QUEUE=$LOG_DIR/queue_train.txt
EVAL_QUEUE=$LOG_DIR/queue_eval.txt
QUEUE_LOCK=$LOG_DIR/queue.lock
SUMMARY_LOCK=$LOG_DIR/summary.lock
EVAL_LOCK=$LOG_DIR/eval.lock
touch "$QUEUE_LOCK" "$SUMMARY_LOCK" "$EVAL_LOCK"

# ---------------------------------------------------------------- job lists
# <model_type>|<direction>|<dup-equiv>|<tag>   (tag = the run-name suffix the scorer globs)
# The 4 forward {scalegmn,gmn} x {baseline,dup-equiv} checkpoints already exist.
TRAIN_JOBS="mpgmn|forward|True|mpgmn-deq
mpsgmn|forward|True|mpsgmn-deq
gmn|bidirectional|True|gmn-deq-bidir
mpgmn|bidirectional|True|mpgmn-deq-bidir
gmn|bidirectional|False|gmn-base-bidir"

# <family>|<direction>. `general` first: it is the result being asked for.
EVAL_JOBS="general|forward
general|bidirectional
uniform|forward
uniform|bidirectional"

# ---------------------------------------------------------------- helpers

train_done() {
    grep -q "Test metrics @ best val tau:" "$LOG_DIR/train_$1.log" 2>/dev/null
}

eval_done() {
    # The scorer prints END_MD only after every in-scope condition has been scored.
    grep -q '^END_MD$' "$LOG_DIR/eval_$1.log" 2>/dev/null
}

run_train() {
    local MODEL=$1 DIRECTION=$2 DEQ=$3 TAG=$4 GPU=$5
    local LOG="$LOG_DIR/train_${TAG}.log"
    if train_done "$TAG"; then
        echo "[train $TAG] already done, skipping"
        return
    fi
    echo "[train $TAG] $MODEL $DIRECTION deq=$DEQ on GPU $GPU ($(date +%H:%M))"
    # --duplication-equiv is always passed explicitly: the checkpoint records the mode, and
    # eval_sizegen_predgen_duplicated.py reads it back from there rather than re-specifying it
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF_DIR/${MODEL}_dup_w16.yml" \
        --duplication-equiv "$DEQ" \
        --direction "$DIRECTION" \
        --wandb True \
        --run_name "${RUN_PREFIX}-${TAG}" \
        > "$LOG" 2>&1
    if train_done "$TAG"; then
        echo "[train $TAG] done ($(date +%H:%M))"
    else
        echo "[train $TAG] FAILED -- see $LOG"
    fi
}

run_eval() {
    local FAMILY=$1 DIRECTION=$2 GPU=$3
    local TAG="${FAMILY}_${DIRECTION}"
    local LOG="$LOG_DIR/eval_${TAG}.log"
    if eval_done "$TAG"; then
        echo "[eval $TAG] already done, skipping"
        return
    fi
    echo "[eval $TAG] scoring w{$WIDTHS} on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_sizegen_predgen_duplicated.py \
        --dataset "$DATASET" --family "$FAMILY" --direction "$DIRECTION" \
        --model-type all --widths $WIDTHS --wandb True \
        --save-predictions "$LOG_DIR/preds_${TAG}.npz" \
        > "$LOG" 2>&1
    if eval_done "$TAG"; then
        append_summary "$FAMILY" "$DIRECTION" "$LOG"
        echo "[eval $TAG] done, appended to SUMMARY ($(date +%H:%M))"
    else
        echo "[eval $TAG] FAILED -- see $LOG"
    fi
}

append_summary() {
    local FAMILY=$1 DIRECTION=$2 LOG=$3
    local RUN_ID
    RUN_ID=$(grep '^WANDB_RUN_ID:' "$LOG" | tail -1 | awk '{print $2}')
    (
        flock 9
        {
            echo ""
            echo "## \`$FAMILY\` widening — $DIRECTION"
            echo ""
            echo "Eval run: [\`$RUN_ID\`](https://wandb.ai/yuxinma/$WANDB_PROJECT/runs/$RUN_ID)"
            echo "Predictions: \`$LOG_DIR/preds_${FAMILY}_${DIRECTION}.npz\`"
            sed -n '/^BEGIN_MD$/,/^END_MD$/p' "$LOG" | grep -vx 'BEGIN_MD\|END_MD'
            if grep -q '\[SKIP\]' "$LOG"; then
                echo ""
                echo "Out of scope in this cell:"
                grep '\[SKIP\]' "$LOG" | sed 's/^ *\[SKIP\] /- /'
            fi
        } >> "$SUMMARY"
    ) 9>"$SUMMARY_LOCK"
}

# ---------------------------------------------------------------- queues

pop_job() {
    (
        flock 9
        head -1 "$1" 2>/dev/null
        [ -f "$1" ] && sed -i '1d' "$1"
    ) 9>"$QUEUE_LOCK"
}

worker() {
    local GPU=$1 QUEUE=$2
    while true; do
        local SPEC
        SPEC=$(pop_job "$QUEUE")
        [ -z "$SPEC" ] && break
        if [ "$QUEUE" = "$TRAIN_QUEUE" ]; then
            IFS='|' read -r MODEL DIRECTION DEQ TAG <<< "$SPEC"
            run_train "$MODEL" "$DIRECTION" "$DEQ" "$TAG" "$GPU"
        else
            IFS='|' read -r FAMILY DIRECTION <<< "$SPEC"
            run_eval "$FAMILY" "$DIRECTION" "$GPU"
        fi
    done
    echo "[gpu $GPU] $(basename "$QUEUE") drained ($(date +%H:%M))"
}

drain() {
    local QUEUE=$1
    local PIDS=()
    for GPU in $GPUS; do
        worker "$GPU" "$QUEUE" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
}

# ---------------------------------------------------------------- main

(
    flock 9
    [ -f "$TRAIN_QUEUE" ] || echo "$TRAIN_JOBS" | grep . > "$TRAIN_QUEUE"
    [ -f "$EVAL_QUEUE" ] || echo "$EVAL_JOBS" | grep . > "$EVAL_QUEUE"
) 9>"$QUEUE_LOCK"

if [ ! -f "$SUMMARY" ]; then
    {
        echo "# $DATASET CNN accuracy prediction — Experiment 2, widening families x 9 conditions"
        echo ""
        echo "Metanetworks trained at w16 only, then scored on w16–w128 CNNs that are"
        echo "*function-identical* to their w16 originals (\`generate_duplicated_cnns.py\`), so a"
        echo "drop is the metanetwork failing to be invariant to that widening family, never"
        echo "distribution shift. A model invariant to a family reports the identical \$\\tau\$ at"
        echo "every width and \`max |Δy|\` at the float32 floor."
        echo ""
        echo "- \`general\`: row condition only — the largest function-preserving family."
        echo "- \`uniform\`: channel-Kronecker (row + col) — the per-condition reference."
        echo ""
        echo "No SP / muP distinction: no wide CNN is ever trained here."
        echo ""
        echo "Started $(date)."
    } > "$SUMMARY"
fi

echo "Queues: $(grep -c . "$TRAIN_QUEUE" || true) trainings, $(grep -c . "$EVAL_QUEUE" || true) evals pending"
if [ -z "$GPUS" ]; then
    echo "No GPU list given -- queues seeded, no worker started."
    exit 0
fi

echo "=== $DATASET widening families: workers on GPUs [$GPUS] $(date) ==="
echo "--- phase 1: training the 5 missing w16 conditions ---"
drain "$TRAIN_QUEUE"

# Phase 2 only makes sense once every training has produced a checkpoint: an eval job scores
# `--model-type all`, so a missing checkpoint would silently drop a condition. The lock makes
# a second, concurrent invocation help with phase 1 and then skip phase 2 rather than
# duplicate it.
MISSING=""
while IFS='|' read -r _M _D _Q TAG; do
    train_done "$TAG" || MISSING="$MISSING $TAG"
done <<< "$TRAIN_JOBS"

exec 9>"$EVAL_LOCK"
if [ -n "$MISSING" ]; then
    echo "=== phase 2 SKIPPED: trainings not finished:$MISSING (re-run to retry) ==="
    exit 1
elif flock -w 0 9; then
    echo "--- phase 2: scoring {general,uniform} x {forward,bidirectional} ---"
    drain "$EVAL_QUEUE"
    flock -u 9
else
    echo "=== another invocation holds the eval lock; skipping phase 2 ==="
    exit 0
fi

{
    echo ""
    echo "---"
    echo ""
    echo "Queue drained $(date). Jobs whose log lacks END_MD failed; grep $LOG_DIR for"
    echo "tracebacks and re-run this script to retry only those."
} >> "$SUMMARY"
touch "$LOG_DIR/DONE"
echo "=== all workers finished $(date); see $SUMMARY ==="

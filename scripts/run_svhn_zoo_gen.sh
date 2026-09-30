#!/bin/bash
# Stage 3a of the SVHN-GS CNN accuracy-prediction study: generate the multi-width CNN zoo and
# write its splits, both arms.
#
# Line-for-line the SVHN-GS twin of run_cifar10_zoo_gen.sh: identical widths, queue order,
# roles, seeds and finalize logic, with --dataset svhn threaded through so the zoo lands in
# $ANYDIM_DATA_ROOT/svhn_cnn_zoo/ instead of cnn_zoo/. Keeping everything else fixed is what
# makes the two studies comparable.
#
# This is only a driver around scripts/generate_cnn_zoo.py and
# scripts/generate_predgen_splits.py -- it holds no policy of its own. It exists because
# the chain is long (CIFAR-10 took ~100 GPU-hours; SVHN's train split is 73257 images vs
# 50000, so a step budget of `epochs * train_frac * n_train / batch_size` is ~1.47x longer
# per model, i.e. expect ~150 GPU-hours), must survive a disconnect, and must run its steps
# in the right order: the SP arm's w16 before the muP16 arm's w16 (which is hard-linked from
# it by the base-width identity), and both arms complete before their splits.
#
# One (arm, width) job per queue line, drained by one worker per GPU, longest-first
# (w128 -> w32) so the expensive widths do not end up as a serial tail. Two jobs never
# touch the same width directory, so workers cannot interleave writes to one meta_*.jsonl.
# generate_cnn_zoo.py is itself resumable per cohort (a cohort whose .pth files and meta
# records all exist is skipped), so a killed worker loses at most one cohort.
#
# Re-running is safe and is also how you ADD a worker: the queue file is only created if
# missing, and a job whose dry-run reports todo=0 for every role is skipped. To hard-reset,
# delete $LOG_DIR/queue.txt.
#
# The last worker out (queue empty and every job complete) does the finalize step: the
# w16 muP16 hard-link, then the splits for both arms, then touches $LOG_DIR/DONE. That
# sentinel is what unblocks Stage 3 (run_svhn_predgen_sizegen_v1.sh).
#
# The muP CNN implementation this drives is dataset-independent and already cleared
# verification gate V5 on the CIFAR-10 side (2026-08-21) -- muP scaling is a property of the
# architecture and optimizer, not of the images. Re-run `scripts/check_mup_cnn.py --only
# identity` against the SVHN w16 arms once they exist to confirm the base-width identity
# still holds on these weights.
#
# GPU budget: at most 4 workers at a time, and never on a GPU another user is on.
#
# Launch:
#   screen -dmS svhn_zoo_gen bash -c 'bash scripts/run_svhn_zoo_gen.sh "0 1"'
# Add a worker later:
#   bash scripts/add_worker_when_free.sh 2 scripts/run_svhn_zoo_gen.sh
# Logs:  /tmp/predgen_svhn_zoo/   (gen_{arm}_w{width}.log, splits_{arm}.log, DONE sentinel)

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/predgen_svhn_zoo
WIDTHS="16 32 48 64 96 128"
GPUS=${1:-"0 1"}          # one worker per GPU
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock
FINALIZE_LOCK=$LOG_DIR/finalize.lock

# ---------------------------------------------------------------- helpers

job_done() {
    # A job is done when the generator's own plan reports nothing left to do. Asking the
    # generator (rather than counting files here) keeps the completeness rule in one place.
    local ARM=$1 W=$2
    local PLAN=$($PYTHON scripts/generate_cnn_zoo.py --dataset svhn --arm "$ARM" --widths "$W" \
                     --dry-run 2>/dev/null | grep -oE "todo=[0-9]+")
    [ -n "$PLAN" ] && ! echo "$PLAN" | grep -qvE "todo=0$"
}

gen() {
    local ARM=$1 W=$2 GPU=$3
    local LOG="${LOG_DIR}/gen_${ARM}_w${W}.log"
    if job_done "$ARM" "$W"; then
        echo "[$ARM w$W] already complete, skipping"
        return 0
    fi
    echo "[$ARM w$W] generating on GPU $GPU ($(date +%H:%M))"
    # append: a resumed run must not overwrite the record of what was already trained
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/generate_cnn_zoo.py --dataset svhn \
        --arm "$ARM" --widths "$W" >> "$LOG" 2>&1 \
        || { echo "[$ARM w$W] FAILED -- see $LOG"; return 1; }
    echo "[$ARM w$W] done ($(date +%H:%M)): $(tail -2 "$LOG" | head -1)"
}

finalize() {
    # Runs once, under a lock, when a worker finds the queue empty. Everything here needs
    # the WHOLE zoo present, so it no-ops until the last job has landed.
    (
        flock 9
        [ -f "$LOG_DIR/DONE" ] && exit 0
        for ARM in sp mup16; do
            for W in $WIDTHS; do
                [ "$ARM" = mup16 ] && [ "$W" = 16 ] && continue   # hard-linked below
                job_done "$ARM" "$W" || { echo "[finalize] $ARM w$W incomplete, waiting"; exit 0; }
            done
        done
        echo "[finalize] hard-linking w16 muP16 from w16 SP ($(date +%H:%M))"
        $PYTHON scripts/generate_cnn_zoo.py --dataset svhn --arm mup16 --widths 16 \
            >> "${LOG_DIR}/gen_mup16_w16.log" 2>&1 \
            || { echo "[finalize] the w16 link FAILED"; exit 1; }
        for ARM in sp mup16; do
            echo "[finalize] writing $ARM splits ($(date +%H:%M))"
            $PYTHON scripts/generate_predgen_splits.py --dataset svhn --arm "$ARM" --widths $WIDTHS \
                > "${LOG_DIR}/splits_${ARM}.log" 2>&1 \
                || { echo "[finalize] $ARM splits FAILED -- see ${LOG_DIR}/splits_${ARM}.log"; exit 1; }
            tail -3 "${LOG_DIR}/splits_${ARM}.log"
        done
        date > "$LOG_DIR/DONE"
        echo "[finalize] zoo + splits complete ($(date)). Stage 3 is unblocked."
    ) 9> "$FINALIZE_LOCK"
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
        IFS='|' read -r ARM W <<< "$SPEC"
        gen "$ARM" "$W" "$GPU"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
    finalize
}

# Only seed the queue on a fresh start, so re-running this script just adds workers.
# Longest-processing-time-first: cost per model grows ~c^2, and w16 carries the 10.2k
# train/val models, which puts it between w64 and w48.
if [ ! -f "$QUEUE" ]; then
    : > "$QUEUE"
    for SPEC in "sp|128" "mup16|128" "sp|96" "mup16|96" "sp|64" "mup16|64" \
                "sp|16" "sp|48" "mup16|48" "sp|32" "mup16|32"; do
        echo "$SPEC" >> "$QUEUE"
    done
    echo "=== Seeded queue with $(wc -l < "$QUEUE") (arm, width) jobs ==="
fi

echo "=== zoo generation: GPUs [$GPUS], $(date) ==="
for GPU in $GPUS; do
    worker "$GPU" 2>&1 | sed "s/^/[gpu$GPU] /" &
done
wait
echo "=== workers done $(date) ==="

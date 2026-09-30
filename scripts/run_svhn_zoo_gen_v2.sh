#!/bin/bash
# CNN accuracy-prediction study, v2 zoo extension: generate w192/w256/w384/w512 in both
# arms and rewrite every width's split, so Stage 3's reach goes from 8x the train width
# to 32x -- matching the FMNIST INR size-gen v2 arm.
#
# Why these four widths: the existing grid is {3,4} * 2^k from w48 up (48, 64, 96, 128),
# i.e. alternating x1.5 / x1.33, which is the same ladder the INR experiments use. Adding
# only 256/512 would halve the log density above w128; 192/256/384/512 keeps it regular
# and costs ~40% more than the sparse set, because the top width dominates either way.
#
# NOTHING is retrained on the metanetwork side. The v1 checkpoints stay valid: the test
# role's HP draws come from a width-specific seed (3000 + width), so a new width adds
# fresh draws without touching any existing width's train/val/test, and every existing
# split JSON is reproduced byte-for-byte. scripts/run_svhn_predgen_sizegen_v2.sh does
# the re-score with --eval-only-ckpt.
#
# PREREQUISITE, already applied: src/data/train_zoo_cnns.py's export_state_dicts() now
# clones each per-model tensor. Without it every saved .pth is a *view* into the cohort's
# batch storage and torch.save writes the whole storage, inflating the zoo by the cohort
# size (145 MB per w128 file for 1.19 MB of tensors). At w512 that is 1.36 TB instead of
# 45 GB, i.e. it does not fit. The four new widths cost ~89 GB with the fix.
#
# Structure is copied from scripts/run_svhn_zoo_gen.sh: one (arm, width) job per queue
# line, one worker per GPU, longest-first, resumable per cohort. Two jobs never touch the
# same width directory. Re-running is safe and is how you ADD a worker; the queue file is
# only created if missing. To hard-reset, delete $LOG_DIR/queue.txt.
#
# The last worker out writes the splits for BOTH arms over ALL widths (16..512) -- not
# just the new ones, because generate_predgen_splits.py's cross-width disjointness
# assertion only covers the widths of one invocation -- then touches $LOG_DIR/DONE.
#
# Launch:
#   screen -dmS svhn_zoo_gen_v2 bash -c 'bash scripts/run_svhn_zoo_gen_v2.sh "0 4"'
# Add a worker once another GPU frees up:
#   bash scripts/add_worker_when_free.sh 2 scripts/run_svhn_zoo_gen_v2.sh
# Logs:  /tmp/predgen_svhn_zoo_v2/   (gen_{arm}_w{width}.log, splits_{arm}.log, DONE sentinel)

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/predgen_svhn_zoo_v2
NEW_WIDTHS="192 256 384 512"            # what this script generates
ALL_WIDTHS="16 32 48 64 96 128 192 256 384 512"   # what the splits are written over
GPUS=${1:-"4 5"}                        # one worker per GPU
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
    # Runs once, under a lock, when a worker finds the queue empty. The splits need the
    # whole zoo present, so it no-ops until the last job has landed.
    (
        flock 9
        [ -f "$LOG_DIR/DONE" ] && exit 0
        for ARM in sp mup16; do
            for W in $NEW_WIDTHS; do
                job_done "$ARM" "$W" || { echo "[finalize] $ARM w$W incomplete, waiting"; exit 0; }
            done
        done
        # Record the pre-existing splits so a regression is visible rather than silent:
        # every width already written must come back byte-identical.
        md5sum $ANYDIM_DATA_ROOT/svhn_cnn_zoo/w*_{sp,mup16}/svhn_predgen_splits.json \
            > "${LOG_DIR}/splits_md5_before.txt" 2>/dev/null
        for ARM in sp mup16; do
            echo "[finalize] writing $ARM splits over all widths ($(date +%H:%M))"
            $PYTHON scripts/generate_predgen_splits.py --dataset svhn --arm "$ARM" --widths $ALL_WIDTHS \
                > "${LOG_DIR}/splits_${ARM}.log" 2>&1 \
                || { echo "[finalize] $ARM splits FAILED -- see ${LOG_DIR}/splits_${ARM}.log"; exit 1; }
            grep -E "disjointness|wrote" "${LOG_DIR}/splits_${ARM}.log" | tail -12
        done
        md5sum -c "${LOG_DIR}/splits_md5_before.txt" 2>&1 | grep -v ': OK$' \
            > "${LOG_DIR}/splits_md5_changed.txt"
        if [ -s "${LOG_DIR}/splits_md5_changed.txt" ]; then
            echo "[finalize] WARNING: pre-existing split JSONs changed -- the v1"
            echo "           checkpoints' training data may no longer match. See"
            echo "           ${LOG_DIR}/splits_md5_changed.txt"
        else
            echo "[finalize] pre-existing split JSONs unchanged (v1 checkpoints stay valid)"
        fi
        date > "$LOG_DIR/DONE"
        echo "[finalize] v2 zoo + splits complete ($(date)). The re-score is unblocked:"
        echo "           bash scripts/run_svhn_predgen_sizegen_v2.sh \"<gpus>\""
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

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
# Longest-processing-time-first: cost per model grows towards ~c^2, so w512 must start
# first or it becomes a serial tail. No width here carries the train/val roles.
if [ ! -f "$QUEUE" ]; then
    : > "$QUEUE"
    for SPEC in "sp|512" "mup16|512" "sp|384" "mup16|384" \
                "sp|256" "mup16|256" "sp|192" "mup16|192"; do
        echo "$SPEC" >> "$QUEUE"
    done
    echo "=== Seeded queue with $(wc -l < "$QUEUE") (arm, width) jobs ==="
fi

echo "=== v2 zoo generation: GPUs [$GPUS], $(date) ==="
for GPU in $GPUS; do
    worker "$GPU" 2>&1 | sed "s/^/[gpu$GPU] /" &
done
wait
echo "=== workers done $(date) ==="

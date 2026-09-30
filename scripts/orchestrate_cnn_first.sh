#!/bin/bash
# Run the CNN accuracy-prediction study to completion on the whole GPU pool, then hand the
# pool back to the paused MNIST / FMNIST size-generalization queues.
#
# Ordering policy (2026-08-21): the CNN study has priority, the INR queues resume after it.
# Nothing here preempts a running job -- every GPU is taken only once it goes idle, via
# scripts/add_worker_when_free.sh, so a job that was already training keeps its GPU until
# it finishes on its own.
#
# What it does, in order:
#   1. grows the zoo-generation pool (scripts/run_cifar10_zoo_gen.sh) onto every GPU in
#      $GPUS as that GPU frees up -- GPU 1 is still finishing an FMNIST training and GPU 2
#      is running Stage 2 (scripts/run_cifar10_predgen_dup_stage2.sh);
#   2. waits for the zoo to be complete: the sentinel /tmp/predgen_zoo/DONE, written by the
#      last zoo worker after the w16 muP16 hard-link and both arms' splits;
#   3. starts Stage 3 (scripts/run_cifar10_predgen_sizegen_v1.sh, 10 conditions) on one GPU
#      to seed its queue, then adds a worker per remaining GPU as each frees up;
#   4. waits for Stage 3 to drain (queue empty and no trainer process left);
#   5. resumes the INR queues: MNIST size-gen v3 bidirectional (its queue was drained, so
#      it is reseeded from scratch -- finished sweeps/trainings are detected in the logs and
#      skipped) and FMNIST size-gen v2 bidirectional (its 9 pending conditions were saved to
#      queue.paused.txt when it was paused).
#
# GPU 3 is deliberately never used: it holds another user's allocation, and the etiquette
# rule is to leave a GPU free for others (6 and 7 are theirs as well).
#
# Launch:
#   screen -dmS cnn_first bash -c 'bash scripts/orchestrate_cnn_first.sh "0 1 2 4 5"'
# Log:  /tmp/orchestrate_cnn_first.log
# Stop it (without touching any running job): screen -S cnn_first -X quit

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPUS=${1:-"0 1 2 4 5"}
ZOO_DIR=/tmp/predgen_zoo
S3_DIR=/tmp/predgen_sizegen_v1
MNIST_DIR=/tmp/sizegen_v3_bidir
FMNIST_DIR=/tmp/fmnist_sizegen_v2_bidir
# GPU split for the resumed INR queues, mirroring how they were running before the pause
MNIST_GPUS=${2:-"0 2"}
FMNIST_GPUS=${3:-"1 4 5"}
# GPUs that still hold non-CNN work at launch time, to be claimed for zoo generation as
# they free (the other GPUs in $GPUS already run a zoo worker)
ZOO_POLL_GPUS=${4:-"1 2"}

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

poll_add() {
    # add a worker to $2's queue on GPU $1, once GPU $1 is idle; never preempts
    local GPU=$1 RUNNER=$2
    nohup bash scripts/add_worker_when_free.sh "$GPU" "$RUNNER" \
        >> "/tmp/orchestrate_poller_gpu${GPU}.log" 2>&1 &
    say "poller: GPU $GPU -> $(basename "$RUNNER")"
}

wait_drain() {
    # $1 = queue file, $2 = pgrep pattern of the runner's trainer process
    local QUEUE=$1 PATTERN=$2
    while true; do
        # NB: both counters must be a single integer. `cmd || echo 0` is wrong here --
        # grep -c / pgrep -fc already print "0" before exiting non-zero on no match, so
        # the fallback appends a second line and the [ -eq ] tests below abort with
        # "integer expression expected", looping forever. (This wedged the 2026-08-22
        # run after Stage 3 drained; the INR queues had to be resumed by hand.)
        local LEFT=0
        [ -f "$QUEUE" ] && LEFT=$(grep -c . "$QUEUE" 2>/dev/null | head -1)
        local RUNNING=$(pgrep -fc "$PATTERN" 2>/dev/null | head -1)
        LEFT=${LEFT:-0}
        RUNNING=${RUNNING:-0}
        if [ "$LEFT" -eq 0 ] && [ "$RUNNING" -eq 0 ]; then return 0; fi
        sleep 300
    done
}

say "=== orchestrator start; GPU pool [$GPUS] ==="

# ---------------------------------------------------------------- 1. zoo generation
# Zoo workers already run on GPUs 0, 4, 5. $ZOO_POLL_GPUS are the ones still busy with
# other work (GPU 1: an FMNIST training; GPU 2: Stage 2) -- claim them when they free.
for GPU in $ZOO_POLL_GPUS; do
    poll_add "$GPU" scripts/run_cifar10_zoo_gen.sh
done

say "waiting for the zoo + splits sentinel $ZOO_DIR/DONE"
while [ ! -f "$ZOO_DIR/DONE" ]; do sleep 300; done
say "zoo complete: $(cat "$ZOO_DIR/DONE")"

# any zoo pollers still waiting would now start a worker that finds an empty queue and
# exits immediately, which is harmless -- but stop them so they do not hold a GPU reading
pkill -f "add_worker_when_free.sh .* scripts/run_cifar10_zoo_gen.sh" 2>/dev/null || true

# ---------------------------------------------------------------- 2. Stage 3
# Seed the queue with a single worker first: run_cifar10_predgen_sizegen_v1.sh only seeds
# when queue.txt is missing, so two simultaneous first launches could double-seed it.
SEED_GPU=$(echo "$GPUS" | awk '{print $1}')
say "Stage 3: seeding queue with a worker on GPU $SEED_GPU"
nohup bash scripts/add_worker_when_free.sh "$SEED_GPU" \
    scripts/run_cifar10_predgen_sizegen_v1.sh >> /tmp/orchestrate_s3_seed.log 2>&1 &
while [ ! -f "$S3_DIR/queue.txt" ]; do sleep 60; done
say "Stage 3 queue seeded ($(grep -c . "$S3_DIR/queue.txt") conditions)"
for GPU in $GPUS; do
    [ "$GPU" = "$SEED_GPU" ] && continue
    poll_add "$GPU" scripts/run_cifar10_predgen_sizegen_v1.sh
done

say "waiting for Stage 3 to drain"
sleep 600      # let the first workers claim their jobs before testing for "drained"
wait_drain "$S3_DIR/queue.txt" "train_sizegen_predgen.py"
say "Stage 3 drained. Summary: $S3_DIR/SUMMARY.md"

# ---------------------------------------------------------------- 3. resume the INR queues
pkill -f "add_worker_when_free.sh .* scripts/run_cifar10_predgen_sizegen_v1.sh" 2>/dev/null || true

say "resuming MNIST size-gen v3 bidirectional on GPUs [$MNIST_GPUS]"
rm -f "$MNIST_DIR/queue.txt"       # drained at pause time; reseed and let the logs skip
screen -dmS sizegen_bidir bash -c \
    "bash scripts/run_mnist_cls_sizegen_v3_bidir.sh \"$MNIST_GPUS\" >> $MNIST_DIR/driver.log 2>&1"

if [ -f "$FMNIST_DIR/queue.paused.txt" ]; then
    say "resuming FMNIST size-gen v2 bidirectional on GPUs [$FMNIST_GPUS] ($(grep -c . "$FMNIST_DIR/queue.paused.txt") conditions restored)"
    cat "$FMNIST_DIR/queue.paused.txt" >> "$FMNIST_DIR/queue.txt"
    rm -f "$FMNIST_DIR/queue.paused.txt"
fi
screen -dmS fmnist_bidir bash -c \
    "bash scripts/run_fmnist_cls_sizegen_v2_bidir.sh \"$FMNIST_GPUS\" >> $FMNIST_DIR/worker_resumed.log 2>&1"

say "=== orchestrator done: CNN study finished, INR queues running ==="

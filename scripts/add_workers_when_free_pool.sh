#!/bin/bash
# Grow a job queue onto whichever GPU frees up FIRST, under a hard cap on how many GPUs
# this session may hold.
#
# The pool-level counterpart of add_worker_when_free.sh, which waits on ONE named GPU. On a
# shared node you usually cannot predict which of the other users' jobs ends first, so
# picking two fixed GPUs to wait on risks waiting on the two long ones and never growing at
# all. This polls every candidate GPU and claims them in the order they actually go idle,
# stopping at $CAP additional workers.
#
# It never preempts anything: a GPU is claimed only after $NEED consecutive readings below
# $FREE_MIB, and GPUs listed in $EXCLUDE (the ones this session already runs workers on)
# are never candidates.
#
# Usage:
#   bash scripts/add_workers_when_free_pool.sh <cap> <exclude> <runner...> \
#        [-- free_mib consec_checks]
#
#   cap            how many ADDITIONAL workers to start before exiting
#   exclude        space-separated GPUs to ignore (quote it), e.g. "2"
#   runner...      one or more queue scripts, run in sequence on each claimed GPU as
#                  `bash <runner> "<gpu>"`; the next one starts when the previous queue
#                  drains, so a GPU flows from stage to stage on its own
#   free_mib       treat a GPU as free below this many MiB used (default 1500)
#   consec_checks  consecutive free readings, 60 s apart, required to claim (default 20 --
#                  long enough that another user's inter-job gap cannot trigger a claim)
#
# Example - hold at most 3 GPUs total, one already busy on GPU 2, chaining all three
# SVHN training stages onto each GPU claimed:
#   screen -dmS svhn_grow bash -c 'bash scripts/add_workers_when_free_pool.sh 2 "2" \
#       scripts/run_svhn_predgen_sizegen_v1.sh \
#       scripts/run_svhn_predgen_sizegen_v1_bidir.sh \
#       scripts/run_svhn_predgen_mpsgmn_sizegen.sh'
#
# Log: whatever the caller redirects; each claim is announced with a timestamp.

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CAP=${1:?usage: add_workers_when_free_pool.sh <cap> <exclude> <runner...> [-- free_mib consec]}
EXCLUDE=${2:?usage: add_workers_when_free_pool.sh <cap> <exclude> <runner...> [-- free_mib consec]}
shift 2

RUNNERS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    [ -f "$1" ] || { echo "no such runner script: $1"; exit 1; }
    RUNNERS+=("$1")
    shift
done
[ ${#RUNNERS[@]} -gt 0 ] || { echo "no runner scripts given"; exit 1; }
[ "${1-}" = "--" ] && shift
FREE_MIB=${1:-1500}
NEED=${2:-20}
INTERVAL=60

ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
CANDIDATES=()
for g in $ALL_GPUS; do
    SKIP=0
    for e in $EXCLUDE; do [ "$g" = "$e" ] && SKIP=1; done
    [ "$SKIP" = 0 ] && CANDIDATES+=("$g")
done

echo "=== pool watcher: claiming up to $CAP GPU(s) from [${CANDIDATES[*]}] $(date) ==="
echo "    idle = <${FREE_MIB} MiB for ${NEED} consecutive checks ${INTERVAL}s apart"
echo "    excluded (already ours): [$EXCLUDE]"
echo "    chain per claimed GPU: ${RUNNERS[*]}"

declare -A STREAK
for g in "${CANDIDATES[@]}"; do STREAK[$g]=0; done
CLAIMED=0

while [ "$CLAIMED" -lt "$CAP" ]; do
    for g in "${CANDIDATES[@]}"; do
        [ "${STREAK[$g]}" = "-1" ] && continue      # already claimed
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        if [ -z "$USED" ]; then
            STREAK[$g]=0
        elif [ "$USED" -lt "$FREE_MIB" ]; then
            STREAK[$g]=$(( ${STREAK[$g]} + 1 ))
            echo "[$(date +%H:%M)] GPU $g free (${USED} MiB), streak ${STREAK[$g]}/${NEED}"
            if [ "${STREAK[$g]}" -ge "$NEED" ]; then
                CLAIMED=$((CLAIMED + 1))
                echo "=== claiming GPU $g as worker $CLAIMED/$CAP ($(date)) ==="
                # Each claimed GPU runs the whole chain in its own background subshell, so
                # the watcher stays free to claim the next GPU that frees up.
                (
                    for r in "${RUNNERS[@]}"; do
                        echo "=== [GPU $g] starting $r ($(date)) ==="
                        bash "$r" "$g"
                    done
                    echo "=== [GPU $g] chain complete ($(date)) ==="
                ) &
                STREAK[$g]=-1
                [ "$CLAIMED" -ge "$CAP" ] && break
            fi
        else
            if [ "${STREAK[$g]}" -gt 0 ]; then
                echo "[$(date +%H:%M)] GPU $g busy again (${USED} MiB), resetting streak"
            fi
            STREAK[$g]=0
        fi
    done
    [ "$CLAIMED" -ge "$CAP" ] && break
    sleep $INTERVAL
done

echo "=== cap of $CAP reached; waiting for the claimed GPUs' chains to finish ($(date)) ==="
wait
echo "=== pool watcher done $(date) ==="

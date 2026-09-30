#!/bin/bash
# Wait until a GPU goes idle, then add a queue worker on it.
#
# Used to grow a running job queue overnight without babysitting it: point it at a GPU
# that is currently busy with one of YOUR OWN jobs, and it starts a worker there the
# moment that job exits. It never preempts anything - it only waits for the GPU to be
# genuinely free.
#
# Usage:
#   bash scripts/add_worker_when_free.sh <gpu> <runner-script> [free_mib] [consec_checks]
#
#   gpu            physical GPU index (as reported by nvidia-smi -i)
#   runner-script  queue script to launch; it is called as `bash <runner> "<gpu>"`,
#                  e.g. scripts/run_fmnist_cls_sizegen_v2.sh (re-running such a script
#                  adds a worker rather than duplicating work)
#   free_mib       treat the GPU as free below this many MiB used (default 1500)
#   consec_checks  require this many consecutive free readings, 60 s apart, so a brief
#                  gap between two of someone else's jobs does not count (default 3)
#
# Example - start FMNIST workers on GPUs 1 and 4 as soon as the running eval jobs end:
#   screen -dmS fmnist_w2 bash -c 'bash scripts/add_worker_when_free.sh 1 \
#       scripts/run_fmnist_cls_sizegen_v2.sh'
#   screen -dmS fmnist_w3 bash -c 'bash scripts/add_worker_when_free.sh 4 \
#       scripts/run_fmnist_cls_sizegen_v2.sh'

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU=${1:?usage: add_worker_when_free.sh <gpu> <runner-script> [free_mib] [consec_checks]}
RUNNER=${2:?usage: add_worker_when_free.sh <gpu> <runner-script> [free_mib] [consec_checks]}
FREE_MIB=${3:-1500}
NEED=${4:-3}
INTERVAL=60

[ -f "$RUNNER" ] || { echo "no such runner script: $RUNNER"; exit 1; }

echo "=== waiting for GPU $GPU to drop below ${FREE_MIB} MiB for ${NEED} consecutive checks"
echo "    (polling every ${INTERVAL}s; will then run: bash $RUNNER \"$GPU\") $(date) ==="

STREAK=0
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
    if [ -z "$USED" ]; then
        echo "[$(date +%H:%M)] could not read GPU $GPU, retrying"
        STREAK=0
    elif [ "$USED" -lt "$FREE_MIB" ]; then
        STREAK=$((STREAK + 1))
        echo "[$(date +%H:%M)] GPU $GPU free (${USED} MiB), streak ${STREAK}/${NEED}"
        [ "$STREAK" -ge "$NEED" ] && break
    else
        [ "$STREAK" -gt 0 ] && echo "[$(date +%H:%M)] GPU $GPU busy again (${USED} MiB), resetting streak"
        STREAK=0
    fi
    sleep $INTERVAL
done

echo "=== GPU $GPU is free, starting worker: bash $RUNNER \"$GPU\" ($(date)) ==="
exec bash "$RUNNER" "$GPU"

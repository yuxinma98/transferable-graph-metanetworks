#!/bin/bash
# Size Generalization v4: score the existing v3 checkpoints on the extended test
# widths (w24-w1024) — no retraining.
#
# The v4 split keeps v3's train (51k) and val (1k) images byte-identical and only
# re-partitions the test pool, so every v3 checkpoint is still valid on it: none of
# its training images appear in any v4 test set. Only the per-width test sets change
# (2k -> 1k images each, disjoint across all 13 widths), which is what makes room
# for w384-w1024.
#
# Usage:
#   bash scripts/run_mnist_cls_sizegen_v4_eval.sh                 # GPUs 0, one worker
#   bash scripts/run_mnist_cls_sizegen_v4_eval.sh "0 2"           # two workers
#   INCLUDE_MPGMN_BIDIR=1 bash scripts/run_mnist_cls_sizegen_v4_eval.sh "0 2"
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

GPUS=${1:-0}
PYTHON=${PYTHON:-python}
CKPT_ROOT=${ANYDIM_DATA_ROOT}/checkpoints
LOG_DIR=/tmp/sizegen_v4_eval
mkdir -p $LOG_DIR

# job := <model> <param> <checkpoint dir> <dup-equiv> <direction> <tag>
JOBS=(
    "scalegmn sp    sizegen-v3-sgmn-sp-base-full_f8f8r0zp  False forward       sgmn-sp-base"
    "scalegmn sp    sizegen-v3-sgmn-sp-deq-full_bngbeln2   True  forward       sgmn-sp-deq"
    "scalegmn mup24 sizegen-v3-sgmn-mup-base_2ao0wbnm      False forward       sgmn-mup24-base"
    "scalegmn mup24 sizegen-v3-sgmn-mup-deq_vgr92oh9       True  forward       sgmn-mup24-deq"
    "gmn      sp    sizegen-v3-gmn-sp-base-full_lqrhopx1   False forward       gmn-sp-base"
    "gmn      sp    sizegen-v3-gmn-sp-deq-full_c9y36coc    True  forward       gmn-sp-deq"
    "gmn      mup24 sizegen-v3-gmn-mup-base_to0udreq       False forward       gmn-mup24-base"
    "gmn      mup24 sizegen-v3-gmn-mup-deq_ikmnmpj4        True  forward       gmn-mup24-deq"
    "mpgmn    sp    sizegen-v3-mpgmn-sp-fw_u1ox1bd0        True  forward       mpgmn-sp-fw"
    "mpgmn    mup24 sizegen-v3-mpgmn-mup24-fw_ktv38f0r     True  forward       mpgmn-mup24-fw"
)

# The bidirectional matrix-product runs were still training when this was written;
# opt in once their best.pt has stopped moving.
if [ "${INCLUDE_MPGMN_BIDIR:-0}" = "1" ]; then
    JOBS+=(
        "mpgmn sp    sizegen-v3-mpgmn-sp-bidir_it3vpqm0    True  bidirectional mpgmn-sp-bidir"
        "mpgmn mup24 sizegen-v3-mpgmn-mup24-bidir_aid3oqcf True  bidirectional mpgmn-mup24-bidir"
    )
fi

run_job() {
    local gpu=$1 job=$2
    read -r model param ckpt deq direction tag <<< "$job"
    local ckpt_path=${CKPT_ROOT}/${ckpt}/best.pt
    local log=${LOG_DIR}/${tag}.log

    if [ ! -f "$ckpt_path" ]; then
        echo "[SKIP] $tag: no checkpoint at $ckpt_path"; return
    fi
    if grep -q "mean OOD acc" "$log" 2>/dev/null; then
        echo "[DONE] $tag: already evaluated"; return
    fi

    echo "[GPU $gpu] $tag ($model/$param, deq=$deq, $direction)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/train_sizegen.py \
        --conf configs/mnist_cls/${model}_sizegen_${param}_v4.yml \
        --eval-only-ckpt "$ckpt_path" \
        --duplication-equiv $deq --direction $direction \
        --run_name "sizegen-v4-eval-${tag}" --wandb True \
        > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "[FAIL] $tag — see $log"
    else
        echo "[OK]   $tag"
    fi
}

# One worker per GPU, draining a shared index into JOBS.
QUEUE=$LOG_DIR/queue
printf '%s\n' "${JOBS[@]}" > $QUEUE
i=0
declare -A WORKER_PIDS
for gpu in $GPUS; do
    (
        n=0
        while IFS= read -r job; do
            # each worker takes every Nth line
            if [ $((n % $(echo $GPUS | wc -w))) -eq $i ]; then
                run_job "$gpu" "$job"
            fi
            n=$((n + 1))
        done < $QUEUE
    ) &
    WORKER_PIDS[$gpu]=$!
    i=$((i + 1))
done
for pid in "${WORKER_PIDS[@]}"; do wait $pid; done

# ---------------------------------------------------------------
# Collect per-width accuracy into one table
# ---------------------------------------------------------------
echo ""
echo "=== Size generalization v4 — per-width test accuracy (train w24) ==="
$PYTHON - "$LOG_DIR" <<'PYEOF'
import re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
rows = {}
widths = []
for log in sorted(log_dir.glob("*.log")):
    accs = {int(w): float(a) for w, a in
            re.findall(r"w=\s*(\d+) \((?:IN |OUT)\): acc=([0-9.]+)", log.read_text())}
    if accs:
        rows[log.stem] = accs
        widths = sorted(set(widths) | set(accs))

if not rows:
    print("no results yet")
else:
    print(f"{'condition':<20}" + "".join(f"{'w'+str(w):>8}" for w in widths))
    for name, accs in rows.items():
        print(f"{name:<20}" + "".join(
            f"{accs[w]*100:>7.1f}%" if w in accs else f"{'-':>8}" for w in widths))
PYEOF
echo "Logs: $LOG_DIR"

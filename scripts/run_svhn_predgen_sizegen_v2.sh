#!/bin/bash
# CNN accuracy-prediction study, Stage 3 v2: re-score the ten v1 conditions on the
# extended width ladder (w16-w512, 32x the train width) WITHOUT retraining anything.
#
# This is the CNN analogue of MNIST v4, which re-scored the v3 checkpoints instead of
# retraining: the v2 configs differ from the v1 ones only in their test widths, the train
# width's split JSON is byte-identical, and every new width's test set is a fresh
# hyperparameter draw (seed 3000 + width) that touches no existing width. So the v1
# checkpoint IS the v2 model, and each condition is one eval-only forward pass.
#
# Per condition: one `train_sizegen_predgen.py --eval-only-ckpt` run over all ten widths,
# dumping per-width (pred, actual) arrays for the Stage 3c diagnostics. Its w16-w128 rows
# must reproduce the v1 numbers exactly -- printed as a check by check_v1_agreement(),
# which is the whole reason this is safe to do from a checkpoint.
#
# The checkpoint path is read out of the v1 log (`Checkpoints will be saved to:` in
# /tmp/predgen_svhn_sizegen_v1/full_<prefix>.log), so the two runs cannot disagree about which
# model was scored. Requires /tmp/predgen_svhn_zoo_v2/DONE (the w192-w512 zoo and the rewritten
# splits).
#
# Ten conditions on a FIFO queue drained by one worker per GPU. Re-running is safe and is
# how you ADD a worker: a finished eval is detected in its log and skipped, and pops are
# serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Memory: eval batch size is bs*(16/w)^2, which floors at 1 from w256 on. A w512 CNN graph
# has ~527k edges, each carrying a 9-vector, through 4 GNN layers at d_hid=128 -- the
# largest single graph in the study. It runs under torch.no_grad(), so ~0.3 GB of
# activations; if a width OOMs anyway, that width alone needs a smaller d_hid eval or CPU.
#
# Launch:
#   screen -dmS svhn_predgen_v2 bash -c 'bash scripts/run_svhn_predgen_sizegen_v2.sh "0 2"'
# Add a worker later, once another GPU frees up:
#   bash scripts/add_worker_when_free.sh 1 scripts/run_svhn_predgen_sizegen_v2.sh
# Results:  /tmp/predgen_svhn_sizegen_v2/SUMMARY.md   (appended as each condition finishes)
# Logs:     /tmp/predgen_svhn_sizegen_v2/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/predgen_svhn_sizegen_v2
V1_DIR=/tmp/predgen_svhn_sizegen_v1     # where the checkpoints and the v1 numbers live
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=svhn_predgen
# One worker per GPU; keep >=2 GPUs vacant for other users. Note ${1-} and NOT ${1:-}: an
# EMPTY first argument must seed the queue and start no worker (see the guard below), so
# that the queue can be laid down while the pool is full and drained later by
# add_worker_when_free.sh. With ${1:-"0 2"} an empty argument silently falls through to the
# default pair instead -- and those two GPUs are not always ours.
GPUS=${1-"0 2"}
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

if [ ! -f /tmp/predgen_svhn_zoo_v2/DONE ]; then
    echo "ABORT: /tmp/predgen_svhn_zoo_v2/DONE is missing -- the w192-w512 zoo and the"
    echo "       rewritten splits are not ready. Run scripts/run_svhn_zoo_gen_v2.sh first."
    exit 1
fi

# ---------------------------------------------------------------- helpers

# always pass the flag explicitly, so the eval runs in the mode the checkpoint was trained in
deq_flag() { echo "--duplication-equiv $1"; }

ckpt_of() {
    # The v1 checkpoint directory, read off the v1 full-training log.
    grep -h "Checkpoints will be saved to:" "${V1_DIR}/full_${1}.log" 2>/dev/null \
        | tail -1 | awk '{print $NF}'
}

metrics_of() {
    # "w<width> <tau>" per line, from a log section starting at $2.
    sed -n "/$2/,\$p" "$1" 2>/dev/null \
        | grep -E "^ +w= *[0-9]+" \
        | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+).*/w\1 \3/'
}

check_v1_agreement() {
    # The v2 eval must reproduce v1's w16-w128 taus exactly: same checkpoint, same data,
    # same split file. A mismatch means one of those three assumptions broke -- most
    # likely a rewritten split JSON -- and the v2 columns cannot be trusted.
    local PREFIX=$1
    local V2=$(metrics_of "${LOG_DIR}/eval_${PREFIX}.log" "Per-width test metrics:")
    local V1=$(metrics_of "${V1_DIR}/full_${PREFIX}.log" "Test metrics @ best val tau:")
    [ -z "$V1" ] && { echo "no v1 numbers to compare against"; return; }
    local BAD=0
    while read -r W TAU; do
        [ -z "$W" ] && continue
        local T2=$(echo "$V2" | awk -v w="$W" '$1==w {print $2}')
        [ -z "$T2" ] && continue
        [ "$T2" != "$TAU" ] && { echo "MISMATCH at $W: v1 tau=$TAU, v2 tau=$T2"; BAD=1; }
    done <<< "$V1"
    [ "$BAD" = 0 ] && echo "v1 agreement: OK (w16-w128 taus reproduce exactly)"
}

run_eval() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    local LOG="${LOG_DIR}/eval_${PREFIX}.log"
    if grep -q "mean OOD tau" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] eval already done, skipping"
        return
    fi
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX] no v1 checkpoint (looked for '${CKPT_DIR}/best.pt')"
        return 1
    fi
    echo "[$PREFIX] re-scoring $(basename "$CKPT_DIR") on w16-w512, GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        $(deq_flag "$DEQ") \
        --direction forward \
        --wandb True --run_name "svhn-predgen-sizegen-v2-eval-${PREFIX}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}.npz" \
        > "$LOG" 2>&1 \
        || { echo "[$PREFIX] FAILED -- see $LOG"; return 1; }
}

append_summary() {
    local LABEL=$1 PREFIX=$2
    local LOG="${LOG_DIR}/eval_${PREFIX}.log"
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    local OOD=$(grep "mean OOD tau" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- v1 checkpoint re-scored: \`$(basename "${CKPT_DIR:-n/a}")\` (training run \`${RUN_ID}\`)"
        echo "- v2 eval run: \`${EVAL_ID:-n/a}\` (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- mean OOD tau (w32-w512): **${OOD:-n/a}**"
        echo "- $(check_v1_agreement "$PREFIX")"
        echo "- predictions: \`${LOG_DIR}/preds_${PREFIX}.npz\`, log: \`${LOG}\`"
        echo ""
        echo "| Width | tau | R2 | L1 | R2_recal |"
        echo "|---|---|---|---|---|"
        sed -n '/Per-width test metrics:/,$p' "$LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/'
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
        IFS='|' read -r CONF DEQ PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        if run_eval "$CONF" "$DEQ" "$PREFIX" "$GPU"; then
            append_summary "$LABEL" "$PREFIX"
        fi
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    C=configs/svhn_predgen
    # <conf>|<dup-equiv>|<prefix>|<label>   -- prefixes match the v1 run's, which is how
    # the checkpoint and the v1 comparison numbers are found. Ordered most-informative
    # first: muP16 is the controlled arm and the conv matrix-product GMN was v1's winner.
    cat > "$QUEUE" <<EOF
$C/mpgmn_sizegen_mup16_v2.yml|True|mpgmn-mup16-deq|Conv matrix-product GMN, muP16, dup-equiv
$C/scalegmn_sizegen_mup16_v2.yml|True|sgmn-mup16-deq|ScaleGMN, muP16, dup-equiv
$C/gmn_sizegen_mup16_v2.yml|True|gmn-mup16-deq|Plain GMN, muP16, dup-equiv
$C/scalegmn_sizegen_mup16_v2.yml|False|sgmn-mup16-base|ScaleGMN, muP16, baseline
$C/gmn_sizegen_mup16_v2.yml|False|gmn-mup16-base|Plain GMN, muP16, baseline
$C/mpgmn_sizegen_sp_v2.yml|True|mpgmn-sp-deq|Conv matrix-product GMN, SP, dup-equiv
$C/scalegmn_sizegen_sp_v2.yml|True|sgmn-sp-deq|ScaleGMN, SP, dup-equiv
$C/gmn_sizegen_sp_v2.yml|True|gmn-sp-deq|Plain GMN, SP, dup-equiv
$C/scalegmn_sizegen_sp_v2.yml|False|sgmn-sp-base|ScaleGMN, SP, baseline
$C/gmn_sizegen_sp_v2.yml|False|gmn-sp-base|Plain GMN, SP, baseline
EOF
    {
        echo "# CNN accuracy prediction — size generalization v2 (train w16, test w16–w512)"
        echo ""
        echo "Started: $(date)."
        echo "The **same ten v1 conditions and the same ten v1 checkpoints**, re-scored on the"
        echo "extended width ladder (w16, 32, 48, 64, 96, 128, 192, 256, 384, 512 — 32× the"
        echo "train width). Nothing was retrained: the v2 configs differ from v1 only in their"
        echo "test widths, so each condition is one \`--eval-only-ckpt\` forward pass. Every"
        echo "condition's w16–w128 taus are checked against its v1 numbers; a \"v1 agreement: OK\""
        echo "line means the checkpoint, data and splits are the same ones v1 measured."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all ten are done."
    } > $SUMMARY
    echo "=== Seeded queue with 10 conditions ==="
fi

if [ -z "${GPUS// /}" ]; then
    echo "=== No GPUs given: queue seeded at $QUEUE, no workers started ==="
    exit 0
fi

echo "=== predgen size-gen v2 re-score: starting workers on GPUs [$GPUS] $(date) ==="

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

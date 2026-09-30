#!/bin/bash
# CNN accuracy-prediction study, Stage 3 v2 BIDIRECTIONAL: re-score the bidirectional
# checkpoints on the extended width ladder (w16-w512, 32x the train width) WITHOUT
# retraining anything -- the bidirectional counterpart of run_svhn_predgen_sizegen_v2.sh,
# for the SIX valid `symmetry: permutation` conditions only.
#
# The four `symmetry: scale` bidirectional conditions (ScaleGMN baseline/dup-equiv x
# {SP,muP16}) are OUT OF SCOPE -- there are no SVHN-GS checkpoints to re-score, because
# they are never trained: on CIFAR-10 a bidirectional ScaleGMN on CNN graphs NaNs by
# training step ~13 at every LR tried, the elementwise-division instability the ScaleGMN
# paper's own Appendix A.2 reports for positive-scale symmetry and never resolves. The
# cause is architectural (float32 reciprocal of near-zero conv weights), not dataset-
# specific. See src/models/bidir_reciprocal.py and the same note in
# run_svhn_predgen_sizegen_v1_bidir.sh. Do not add sgmn-{base,deq}-bidir to the queue below.
#
# Each of the three (model, variant) jobs was trained ONCE under the muP16 config
# (base width = train width, so w16 is byte-identical across arms -- v1 verified this with
# `max |dw| = 0.00e+00`) and is re-scored under BOTH arms' v2 configs from that same
# checkpoint, exactly like run_svhn_predgen_sizegen_v1_bidir.sh's run_eval/w16-identity
# pattern. Checkpoint paths are read out of the v1 bidir training log (`Checkpoints will be
# saved to:` in /tmp/predgen_svhn_sizegen_v1_bidir/full_<prefix>.log), so this script cannot
# disagree with v1 about which model is being scored.
#
# Each arm's v2 eval must reproduce that arm's v1 w16-w128 taus exactly (checked below,
# analogous to check_v1_agreement in the forward v2 script) -- same checkpoint, same data,
# same split file for those six widths. And the two arms' w16 rows must still agree with
# each other to the last printed digit (w16 identity), same audit v1 ran.
#
# Requires /tmp/predgen_svhn_zoo_v2/DONE (the w192-w512 zoo and the rewritten splits).
#
# Three jobs on a FIFO queue drained by one worker per GPU. Re-running is safe and is how
# you ADD a worker: a finished eval is detected in its log and skipped, and pops are
# serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Launch:
#   screen -dmS svhn_predgen_v2_bidir bash -c 'bash scripts/run_svhn_predgen_sizegen_v2_bidir.sh "4"'
# Results:  /tmp/predgen_svhn_sizegen_v2_bidir/SUMMARY.md   (appended as each condition finishes)
# Logs:     /tmp/predgen_svhn_sizegen_v2_bidir/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/predgen_svhn_sizegen_v2_bidir
V1_DIR=/tmp/predgen_svhn_sizegen_v1_bidir     # where the checkpoints and the v1 numbers live
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=svhn_predgen
GPUS=${1-"4"}     # one worker per GPU; keep >=2 GPUs vacant for other users
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

if [ ! -f /tmp/predgen_svhn_zoo_v2/DONE ]; then
    echo "ABORT: /tmp/predgen_svhn_zoo_v2/DONE is missing -- the w192-w512 zoo and the"
    echo "       rewritten splits are not ready. Run scripts/run_svhn_zoo_gen_v2.sh first."
    exit 1
fi

# ---------------------------------------------------------------- helpers

deq_flag() { echo "--duplication-equiv $1"; }

ckpt_of() {
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
    # $1 = prefix, $2 = arm (mup16 | sp). v1's own per-arm eval log carries the w16-w128
    # taus this arm's v2 eval must reproduce exactly.
    local PREFIX=$1 ARM=$2
    local V2=$(metrics_of "${LOG_DIR}/eval_${PREFIX}_${ARM}.log" "Per-width test metrics:")
    local V1=$(metrics_of "${V1_DIR}/eval_${PREFIX}_${ARM}.log" "Per-width test metrics:")
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
    # $1 = conf, $2 = dup-equiv, $3 = prefix, $4 = gpu, $5 = arm tag (mup16 | sp)
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4 ARM=$5
    local LOG="${LOG_DIR}/eval_${PREFIX}_${ARM}.log"
    if grep -q "mean OOD tau" "$LOG" 2>/dev/null; then
        echo "[$PREFIX/$ARM] eval already done, skipping"
        return 0
    fi
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX/$ARM] no v1 checkpoint (looked for '${CKPT_DIR}/best.pt')"
        return 1
    fi
    echo "[$PREFIX/$ARM] re-scoring $(basename "$CKPT_DIR") on w16-w512, GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        $(deq_flag "$DEQ") \
        --direction bidirectional \
        --wandb True --run_name "svhn-predgen-sizegen-v2-bidir-eval-${PREFIX}-${ARM}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}_${ARM}.npz" \
        > "$LOG" 2>&1 \
        || { echo "[$PREFIX/$ARM] FAILED -- see $LOG"; return 1; }
}

w16_row() {
    grep -E "^ +w= *16 \(IN" "${LOG_DIR}/eval_${1}_${2}.log" 2>/dev/null | tail -1 \
        | sed -E 's/^ +//'
}

check_w16_identity() {
    # the two arms must still agree to the last digit at w16 after the extension.
    local PREFIX=$1
    local A=$(w16_row "$PREFIX" mup16)
    local B=$(w16_row "$PREFIX" sp)
    if [ -z "$A" ] || [ -z "$B" ]; then
        echo "n/a"; return
    fi
    if [ "$A" = "$B" ]; then echo "OK"; else echo "MISMATCH"; fi
}

append_summary() {
    # $1 = human label, $2 = prefix, $3 = arm tag, $4 = w16-identity verdict
    local LABEL=$1 PREFIX=$2 ARM=$3 IDENT=$4
    local LOG="${LOG_DIR}/eval_${PREFIX}_${ARM}.log"
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    local OOD=$(grep "mean OOD tau" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- v1 bidir checkpoint re-scored: \`$(basename "${CKPT_DIR:-n/a}")\` (training run \`${RUN_ID}\`)"
        echo "- v2 eval run: \`${EVAL_ID:-n/a}\` (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- mean OOD tau (w32-w512): **${OOD:-n/a}**"
        echo "- $(check_v1_agreement "$PREFIX" "$ARM")"
        echo "- w16 identity vs the other arm: **${IDENT}**"
        echo "- predictions: \`${LOG_DIR}/preds_${PREFIX}_${ARM}.npz\`, log: \`${LOG}\`"
        echo ""
        echo "| Width | tau | R2 | L1 | R2_recal |"
        echo "|---|---|---|---|---|"
        sed -n '/Per-width test metrics:/,$p' "$LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/'
    } >> "$SUMMARY.part.${PREFIX}.${ARM}"
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.${PREFIX}.${ARM}' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.${PREFIX}.${ARM}"
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
        IFS='|' read -r MUP_CONF SP_CONF DEQ PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        if run_eval "$MUP_CONF" "$DEQ" "$PREFIX" "$GPU" mup16; then
            if run_eval "$SP_CONF" "$DEQ" "$PREFIX" "$GPU" sp; then
                local IDENT=$(check_w16_identity "$PREFIX")
                append_summary "$LABEL, muP16" "$PREFIX" mup16 "$IDENT"
                append_summary "$LABEL, SP"    "$PREFIX" sp    "$IDENT"
            fi
        fi
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    C=configs/svhn_predgen
    # <mup16 v2 conf>|<sp v2 conf>|<dup-equiv>|<prefix>|<label>  -- prefixes match the v1
    # bidir run's, which is how the checkpoint and the v1 comparison numbers are found.
    # The two sgmn-*-bidir conditions are deliberately absent -- out of scope, see header.
    cat > "$QUEUE" <<EOF
$C/mpgmn_sizegen_mup16_v2.yml|$C/mpgmn_sizegen_sp_v2.yml|True|mpgmn-deq-bidir|Conv matrix-product GMN, dup-equiv, bidirectional
$C/gmn_sizegen_mup16_v2.yml|$C/gmn_sizegen_sp_v2.yml|True|gmn-deq-bidir|Plain GMN, dup-equiv, bidirectional
$C/gmn_sizegen_mup16_v2.yml|$C/gmn_sizegen_sp_v2.yml|False|gmn-base-bidir|Plain GMN, baseline, bidirectional
EOF
    {
        echo "# CNN accuracy prediction — size generalization v1 BIDIRECTIONAL, extended to w512"
        echo ""
        echo "Started: $(date)."
        echo "The three valid \`symmetry: permutation\` bidirectional conditions from"
        echo "\`/tmp/predgen_svhn_sizegen_v1_bidir/SUMMARY.md\` (six {SP,muP16} rows), re-scored on the"
        echo "extended width ladder (w16, 32, 48, 64, 96, 128, 192, 256, 384, 512 — 32× the train"
        echo "width) from the SAME checkpoints. Nothing was retrained: each checkpoint is scored"
        echo "twice (once per arm), same as v1 did to w128. The four \`symmetry: scale\`"
        echo "(ScaleGMN) bidirectional conditions are OUT OF SCOPE and are not queued here — no"
        echo "checkpoint exists to re-score, because a bidirectional ScaleGMN on CNN graphs NaNs"
        echo "in training (ScaleGMN paper Appendix A.2; architectural, not dataset-specific)."
        echo ""
        echo "Every row's w16-w128 taus are checked against that arm's v1 eval log"
        echo "(\"v1 agreement: OK\" = same checkpoint, data and splits as v1 measured), and each"
        echo "condition's two arms are checked against each other at w16 (\"w16 identity\")."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all three are done."
    } > $SUMMARY
    echo "=== Seeded queue with 3 jobs (= 6 conditions) ==="
fi

if [ -z "${GPUS// /}" ]; then
    echo "=== No GPUs given: queue seeded at $QUEUE, no workers started ==="
    exit 0
fi

echo "=== predgen size-gen v2 BIDIR re-score: starting workers on GPUs [$GPUS] $(date) ==="

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

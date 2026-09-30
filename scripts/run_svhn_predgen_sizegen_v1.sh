#!/bin/bash
# Stage 3 of the CNN accuracy-prediction study: size generalization on OUR multi-width
# CNN zoo — train on w16 only, test on w16-w128.
#
# Ten conditions = {ScaleGMN, plain GMN} x {SP, muP16} x {baseline, dup-equiv}, plus the
# conv matrix-product GMN x {SP, muP16} (dup-equiv only -- it has no baseline variant).
# All `--direction forward`, matching the FMNIST size-gen v2 arm; bidirectional was
# deliberately not run for the CNN study.  (The `inf`s in the dataset's unused
# `bw_edge_attr` field are not the reason -- see check_duplication_equiv_cnn.py.)
#
# Protocol per condition, identical to MNIST Experiment 3:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), MAX_STEPS each,
#      selected on **val Kendall tau at w16** (in-distribution only)
#   2. full training (200 epochs, patience from the config) at the best sweep LR
#   3. one eval-only re-score of the kept checkpoint, purely to dump the per-width
#      (pred, actual) arrays the Stage 3c diagnostics need
# Unlike the MNIST v3/v4 runs, step 3 adds no test widths: only six widths x 1k models
# exist, so train_sizegen_predgen.py already scores all of them every epoch (with the
# eval batch size scaled by (16/w)^2 to keep activation memory flat). Its numbers must
# therefore reproduce step 2's exactly, which makes it a checkpoint round-trip check.
#
# The ten conditions are pushed onto a FIFO queue drained by one worker per GPU.
# Re-running is safe and is also how you ADD a worker: the queue file is only created if
# missing, finished sweep LRs and finished full trainings are detected in the logs and
# skipped, and pops are serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Launch:
#   screen -dmS svhn_predgen_v1 bash -c 'bash scripts/run_svhn_predgen_sizegen_v1.sh "0 2"'
# Add a worker later, once another GPU frees up:
#   bash scripts/add_worker_when_free.sh 1 scripts/run_svhn_predgen_sizegen_v1.sh
# Results:  /tmp/predgen_svhn_sizegen_v1/SUMMARY.md   (appended as each condition finishes)
# Logs:     /tmp/predgen_svhn_sizegen_v1/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/predgen_svhn_sizegen_v1
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=svhn_predgen
GPUS=${1:-"0 2"}          # one worker per GPU; keep >=2 GPUs vacant for other users
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

# ---------------------------------------------------------------- helpers

# always pass the flag explicitly, so the checkpoint records which mode it was trained in
deq_flag() { echo "--duplication-equiv $1"; }

run_sweep() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    for idx in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
        if grep -q "Best val tau:" "$LOG" 2>/dev/null; then
            echo "[$PREFIX] sweep lr=${LRS[$idx]} already done, skipping"
            continue
        fi
        echo "[$PREFIX] sweep lr=${LRS[$idx]} on GPU $GPU ($(date +%H:%M))"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
            --conf "$CONF" \
            $(deq_flag "$DEQ") \
            --direction forward \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-svhn-predgen-v1-${PREFIX}-${LRS[$idx]}" \
            > "$LOG" 2>&1
    done
}

get_best_lr() {
    # selection metric: val Kendall tau at the TRAIN width (w16). No OOD width may
    # influence which LR — or later which checkpoint — is kept.
    local PREFIX=$1
    local BEST_TAU=-1
    local BEST_LR="2.5e-4"
    for i in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${i}.log"
        [ -f "$LOG" ] || continue
        local TAU=$(grep "Best val tau:" "$LOG" | tail -1 | awk '{print $NF}')
        [ -z "$TAU" ] && continue
        local BETTER=$(python3 -c "print(1 if $TAU > $BEST_TAU else 0)")
        if [ "$BETTER" = "1" ]; then BEST_TAU=$TAU; BEST_LR=${LRS[$i]}; fi
    done
    echo "$BEST_LR"
}

sweep_taus() {
    # compact inline list of the 7 sweep val taus (never a table — see repo convention)
    local PREFIX=$1
    local OUT=""
    for i in 0 1 2 3 4 5 6; do
        local TAU=$(grep "Best val tau:" "${LOG_DIR}/sweep_${PREFIX}_${i}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [ -z "$TAU" ] && TAU="--"
        OUT="${OUT}${TAU} / "
    done
    echo "${OUT%" / "}"
}

run_full() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4 LR=$5
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    if grep -q "Test metrics @ best val tau:" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] full training already done, skipping"
        return
    fi
    echo "[$PREFIX] full training lr=$LR on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        $(deq_flag "$DEQ") \
        --direction forward \
        --wandb True --lr "$LR" \
        --run_name "svhn-predgen-sizegen-v1-${PREFIX}" \
        > "$LOG" 2>&1
}

run_eval() {
    # Re-score the kept checkpoint once, only to dump per-width (pred, actual) arrays:
    # the Stage 3c diagnostics (tau within fixed-lr HP strata, calibration, tau_model -
    # tau_HP) all need the raw predictions, and --save-predictions is wired into the
    # eval-only path. Numbers here must reproduce run_full's "Test metrics @ best val
    # tau" exactly, which also makes it a checkpoint round-trip check.
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    local LOG="${LOG_DIR}/eval_${PREFIX}.log"
    if grep -q "mean OOD tau" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] eval already done, skipping"
        return
    fi
    local CKPT_DIR=$(grep -h "Checkpoints will be saved to:" "${LOG_DIR}/full_${PREFIX}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX] no checkpoint for the prediction dump (looked for '${CKPT_DIR}/best.pt')"
        return
    fi
    echo "[$PREFIX] prediction dump on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        $(deq_flag "$DEQ") \
        --direction forward \
        --wandb True --run_name "svhn-predgen-sizegen-v1-eval-${PREFIX}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}.npz" \
        > "$LOG" 2>&1
}

append_summary() {
    # $1 = human label, $2 = prefix, $3 = best lr
    local LABEL=$1 PREFIX=$2 LR=$3
    local FULL_LOG="${LOG_DIR}/full_${PREFIX}.log"
    local CKPT_DIR=$(grep -h "Checkpoints will be saved to:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local BEST_VAL=$(grep "Best val tau:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local EPOCHS=$(grep -c "val_tau=" "$FULL_LOG" 2>/dev/null)
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "${LOG_DIR}/eval_${PREFIX}.log" 2>/dev/null | tail -1 | sed 's|.*/||')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- best LR: **$LR** (sweep val taus @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_taus "$PREFIX"))"
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "- checkpoint: \`$(basename "${CKPT_DIR:-n/a}")\`"
        echo "- best val tau (w16): ${BEST_VAL:-n/a}, epochs run: ${EPOCHS:-n/a}"
        echo "- eval/dump run: \`${EVAL_ID:-n/a}\` (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- predictions: \`${LOG_DIR}/preds_${PREFIX}.npz\`, logs: \`${FULL_LOG}\`, \`${LOG_DIR}/eval_${PREFIX}.log\`"
        echo ""
        echo "| Width | tau | R2 | L1 | R2_recal |"
        echo "|---|---|---|---|---|"
        # the trainer prints "  w=   16 (IN ): tau=0.9315 R2=+0.991 L1=0.0077 R2rc=+0.991"
        sed -n '/Test metrics @ best val tau:/,$p' "$FULL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/'
    } >> "$SUMMARY.part.$PREFIX"
    # serialize the append so concurrent workers don't interleave
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.$PREFIX' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.$PREFIX"
}

# ---------------------------------------------------------------- job queue

pop_job() {
    # atomically remove and echo the first queue line ("" if empty)
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
        run_sweep "$CONF" "$DEQ" "$PREFIX" "$GPU"
        local LR=$(get_best_lr "$PREFIX")
        echo "[$PREFIX] best LR = $LR"
        run_full "$CONF" "$DEQ" "$PREFIX" "$GPU" "$LR"
        run_eval "$CONF" "$DEQ" "$PREFIX" "$GPU"
        append_summary "$LABEL" "$PREFIX" "$LR"
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    C=configs/svhn_predgen
    # <conf>|<dup-equiv>|<prefix>|<label>
    # Ordered most-informative first: muP16 is the controlled arm (flat normalized
    # spectral norms across widths, so the OOD covariate shift is what it isolates) and
    # dup-equiv is the headline variant.
    cat > "$QUEUE" <<EOF
$C/scalegmn_sizegen_mup16_v1.yml|True|sgmn-mup16-deq|ScaleGMN, muP16, dup-equiv
$C/gmn_sizegen_mup16_v1.yml|True|gmn-mup16-deq|Plain GMN, muP16, dup-equiv
$C/scalegmn_sizegen_mup16_v1.yml|False|sgmn-mup16-base|ScaleGMN, muP16, baseline
$C/gmn_sizegen_mup16_v1.yml|False|gmn-mup16-base|Plain GMN, muP16, baseline
$C/scalegmn_sizegen_sp_v1.yml|True|sgmn-sp-deq|ScaleGMN, SP, dup-equiv
$C/gmn_sizegen_sp_v1.yml|True|gmn-sp-deq|Plain GMN, SP, dup-equiv
$C/scalegmn_sizegen_sp_v1.yml|False|sgmn-sp-base|ScaleGMN, SP, baseline
$C/gmn_sizegen_sp_v1.yml|False|gmn-sp-base|Plain GMN, SP, baseline
$C/mpgmn_sizegen_mup16_v1.yml|True|mpgmn-mup16-deq|Conv matrix-product GMN, muP16, dup-equiv
$C/mpgmn_sizegen_sp_v1.yml|True|mpgmn-sp-deq|Conv matrix-product GMN, SP, dup-equiv
EOF
    {
        echo "# CNN accuracy prediction — size generalization v1 (train w16, test w16–w128)"
        echo ""
        echo "Started: $(date)."
        echo "Ten conditions: {ScaleGMN, plain GMN} × {SP, muP16} × {baseline, dup-equiv},"
        echo "plus the conv matrix-product GMN × {SP, muP16} (dup-equiv only), all"
        echo "\`--direction forward\`. LR selected on **val Kendall tau at w16** from a"
        echo "7-value ×2 grid (6.25e-5 … 4e-3), then 200 epochs at that LR; the kept checkpoint"
        echo "is also the best-val-tau epoch, so no OOD width influences selection."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all ten are done."
    } > $SUMMARY
    echo "=== Seeded queue with 10 conditions ==="
fi

echo "=== predgen size-gen v1: starting workers on GPUs [$GPUS] $(date) ==="

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

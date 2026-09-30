#!/bin/bash
# Stage 2 of the CNN accuracy-prediction study: the duplicated-CNN sanity check.
#
# No CNNs are trained here. Every widened network in
# $ANYDIM_DATA_ROOT/svhn_cnn_zoo/w{32..128}_zoo_dup/ computes *exactly the same function* as its
# w16 original (uniform channel Kronecker, scripts/generate_duplicated_cnns.py) and carries
# the original's test accuracy as its label. So the question is confound-free: does the
# prediction move when the same function is presented at a wider architecture?
#
# Four conditions = {ScaleGMN, plain GMN} x {baseline, dup-equiv}, all trained at w16 only
# (`--direction forward`; see check_duplication_equiv_cnn.py for why forward is a design
# choice and not a numerical necessity). Then one eval pass per model type scores both of
# its checkpoints at w16-w128 with scripts/eval_sizegen_predgen_duplicated.py.
#
# Expected, from results/VERIFICATION_gmn_properties.md ("Duplication equivalence of the
# GMN on CNN graphs"): the dup-equiv conditions report the IDENTICAL tau at every width and
# max |dy| ~ 1e-7 against the w16 predictions, the baselines drift.
#
# Unlike the size-gen runner there is no LR sweep: this is a property check, not a
# performance comparison, so all four conditions use the config's LR. Model selection is on
# val Kendall tau at w16, which is the only width trained on.
#
# The four conditions are pushed onto a FIFO queue drained by one worker per GPU; the two
# eval passes run afterwards, on the first worker to find the queue empty. Re-running is
# safe and is how you ADD a worker: the queue file is only created if missing, finished
# trainings and finished evals are detected in the logs and skipped. To hard-reset, delete
# $LOG_DIR/queue.txt.
#
# Launch:
#   screen -dmS svhn_predgen_dup bash -c 'bash scripts/run_svhn_predgen_dup_stage2.sh "3"'
# Or start it the moment one of your own GPUs frees up:
#   bash scripts/add_worker_when_free.sh 3 scripts/run_svhn_predgen_dup_stage2.sh
# Results:  /tmp/predgen_svhn_dup_stage2/SUMMARY.md
# Logs:     /tmp/predgen_svhn_dup_stage2/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LOG_DIR=/tmp/predgen_svhn_dup_stage2
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=svhn_predgen
WIDTHS="16 32 48 64 96 128"
GPUS=${1:-"3"}            # one worker per GPU; keep >=2 GPUs vacant for other users
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock
EVAL_LOCK=$LOG_DIR/eval.lock

# ---------------------------------------------------------------- helpers

run_full() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    if grep -q "Test metrics @ best val tau:" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] training already done, skipping"
        return
    fi
    echo "[$PREFIX] training on GPU $GPU ($(date +%H:%M))"
    # always pass the flag explicitly, so the checkpoint records which mode it was
    # trained in -- eval_sizegen_predgen_duplicated.py reads it back from there
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --duplication-equiv "$DEQ" \
        --direction forward \
        --wandb True \
        --run_name "svhn-predgen-dup-${PREFIX}" \
        > "$LOG" 2>&1
}

ckpt_dir() {
    grep -h "Checkpoints will be saved to:" "${LOG_DIR}/full_$1.log" 2>/dev/null \
        | tail -1 | awk '{print $NF}'
}

run_eval() {
    # $1 = model type (scalegmn|gmn), $2 = GPU. Scores BOTH of that model's checkpoints
    # in one process, so the two conditions share the dataset reads.
    local MODEL=$1 GPU=$2
    local LOG="${LOG_DIR}/eval_${MODEL}.log"
    if grep -q "max |dy| against the base-width prediction" "$LOG" 2>/dev/null; then
        echo "[eval $MODEL] already done, skipping"
        return
    fi
    local TAG
    [ "$MODEL" = "scalegmn" ] && TAG=sgmn || TAG=gmn
    local CK_BASE=$(ckpt_dir "${TAG}-base") CK_DEQ=$(ckpt_dir "${TAG}-deq")
    local ARGS=()
    [ -n "$CK_BASE" ] && [ -f "${CK_BASE}/best.pt" ] && ARGS+=(--ckpt "baseline=${CK_BASE}/best.pt")
    [ -n "$CK_DEQ" ] && [ -f "${CK_DEQ}/best.pt" ] && ARGS+=(--ckpt "dup-equiv=${CK_DEQ}/best.pt")
    if [ ${#ARGS[@]} -eq 0 ]; then
        echo "[eval $MODEL] no checkpoints yet, skipping"
        return
    fi
    echo "[eval $MODEL] scoring w{$WIDTHS} on GPU $GPU ($(date +%H:%M))"
    # every path/project default in the scorer points at the CIFAR-10 study, so all three
    # have to be overridden explicitly -- otherwise this would silently re-score that zoo
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/eval_sizegen_predgen_duplicated.py \
        --model-type "$MODEL" \
        "${ARGS[@]}" \
        --widths $WIDTHS \
        --dup-root "$ANYDIM_DATA_ROOT/svhn_cnn_zoo" \
        --conf "configs/svhn_predgen/${MODEL}_dup_w16.yml" \
        --wandb True \
        --wandb-project "$WANDB_PROJECT" \
        --save-predictions "${LOG_DIR}/preds_dup_${MODEL}.npz" \
        > "$LOG" 2>&1
}

append_summary() {
    # $1 = human label, $2 = prefix
    local LABEL=$1 PREFIX=$2
    local FULL_LOG="${LOG_DIR}/full_${PREFIX}.log"
    local CK=$(ckpt_dir "$PREFIX")
    local RUN_ID=$(basename "${CK:-_unknown}" | sed 's/.*_//')
    local BEST_VAL=$(grep "Best val tau:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "- checkpoint: \`$(basename "${CK:-n/a}")\`, best val tau (w16): ${BEST_VAL:-n/a}"
        echo "- log: \`${FULL_LOG}\`"
        echo ""
        sed -n '/Test metrics @ best val tau:/,$p' "$FULL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/' \
            | { read -r first; [ -n "$first" ] && {
                    echo "| Width | tau | R2 | L1 | R2_recal |"; echo "|---|---|---|---|---|"
                    echo "$first"; cat; }; }
    } >> "$SUMMARY.part.$PREFIX"
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.$PREFIX' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.$PREFIX"
}

append_eval_summary() {
    local MODEL=$1
    local LOG="${LOG_DIR}/eval_${MODEL}.log"
    [ -f "$LOG" ] || return
    {
        echo ""
        echo "### Duplicated-width evaluation — ${MODEL}"
        echo ""
        echo "Predictions: \`${LOG_DIR}/preds_dup_${MODEL}.npz\`, log: \`${LOG}\`"
        echo ""
        echo '```'
        sed -n '/^Summary — tau against the labels/,$p' "$LOG" | grep -v "^predictions ->"
        echo '```'
    } >> "$SUMMARY.eval.$MODEL"
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.eval.$MODEL' >> '$SUMMARY'"
    rm -f "$SUMMARY.eval.$MODEL"
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
        run_full "$CONF" "$DEQ" "$PREFIX" "$GPU"
        append_summary "$LABEL" "$PREFIX"
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK" "$EVAL_LOCK"

if [ ! -f "$QUEUE" ]; then
    C=configs/svhn_predgen
    # <conf>|<dup-equiv>|<prefix>|<label>
    # dup-equiv first: it is the condition the property predicts something sharp about.
    cat > "$QUEUE" <<EOF
$C/scalegmn_dup_w16.yml|True|sgmn-deq|ScaleGMN, dup-equiv
$C/gmn_dup_w16.yml|True|gmn-deq|Plain GMN, dup-equiv
$C/scalegmn_dup_w16.yml|False|sgmn-base|ScaleGMN, baseline
$C/gmn_dup_w16.yml|False|gmn-base|Plain GMN, baseline
EOF
    {
        echo "# CNN accuracy prediction — Stage 2, duplicated-CNN sanity check"
        echo ""
        echo "Started: $(date)."
        echo "Four conditions: {ScaleGMN, plain GMN} × {baseline, dup-equiv}, trained at w16"
        echo "only, then scored at w16–w128 on CNNs that are *function-identical* to their w16"
        echo "originals. A duplication-equivariant model must report the identical tau at every"
        echo "width; the baselines are expected to drift."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all four plus"
        echo "both eval passes are done."
    } > $SUMMARY
    echo "=== Seeded queue with 4 conditions ==="
fi

echo "=== predgen Stage 2 (dup): starting workers on GPUs [$GPUS] $(date) ==="

PIDS=()
for GPU in $GPUS; do
    worker "$GPU" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

# Training drained -> score the checkpoints at the widened widths. If a second invocation
# (an added worker) reaches this at the same time, the lock makes it skip rather than run a
# duplicate pass; the per-eval log grep also makes a re-run a no-op.
FIRST_GPU=$(echo $GPUS | awk '{print $1}')
exec 9>"$EVAL_LOCK"
if [ ! -s "$QUEUE" ] && flock -w 0 9; then
    for M in scalegmn gmn; do
        run_eval "$M" "$FIRST_GPU"
        append_eval_summary "$M"
    done
    flock -u 9
elif [ ! -s "$QUEUE" ]; then
    echo "=== another invocation holds the eval lock; skipping the eval passes ==="
fi

if [ ! -s "$QUEUE" ]; then
    echo "" >> $SUMMARY
    echo "Finished: $(date)" >> $SUMMARY
fi

echo "=== Workers done $(date). Summary: $SUMMARY ==="

#!/bin/bash
# Stage 3 of the CNN accuracy-prediction study, FORWARD MATRIX-PRODUCT ScaleGMN
# (`mpsgmn`, conv variant): size generalization on our multi-width CNN zoo -- train on
# w16 only, test on w16-w128 (v1 configs) and w16-w512 (v2 configs).
#
# ONE training, FOUR recorded conditions
# --------------------------------------
# The MSG function has no baseline variant (it presupposes mean aggregation and fan-in
# rescaled edge features), so this family contributes a single (model, variant) cell:
#   conv matrix-product ScaleGMN x {SP, muP16}, dup-equiv, forward.
# And base width = train width makes the two arms' w16 inputs the same data -- Stage 3
# measured `max |dw| = 0.00e+00` on all five of its SP/muP16 pairs *after* paying to
# train both. So this script trains ONCE under the muP16 config and scores that
# checkpoint under each arm's config with `--eval-only-ckpt`, at both width ranges:
#   v1 (w16-w128) x {muP16, SP}  +  v2 (w16-w512) x {muP16, SP}  = 4 summary sections.
#
# The premise is re-verified on disk before anything is queued (verify_arm_identity):
#   - w16_sp/ and w16_mup16/ weight files are hard links (same inode)
#   - their meta_{train,val,test,paired}.jsonl are byte-identical
#   - their cifar10_predgen_splits.json are byte-identical
#   - the mpsgmn SP and muP16 configs differ only in data paths, comments and wandb names
# and each recorded pair prints a `w16 identity` line -- the SP re-score must reproduce the
# muP16 run's w16 row exactly, since that row is literally the same data through the same
# weights. A MISMATCH means the arms are not sharing what this script assumes and the SP
# columns must not be trusted.
#
# Protocol:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), MAX_STEPS each, selected on
#      **val Kendall tau at w16** (in-distribution only -- no OOD width influences it)
#   2. full training (200 epochs, patience from the config) at the best sweep LR
#   3. eval-only re-score + prediction dump under the muP16 v1 config (also a checkpoint
#      round-trip check: it must reproduce step 2's per-width numbers)
#   4. same under the SP v1 config, from the SAME checkpoint; w16 identity check
#   5. the v2 (w16-w512) pair of re-scores -- but only once the Stage 3 v2 zoo has
#      finished generating, i.e. once $ZOO_V2_DONE exists. It is a SEPARATE queue job, so
#      a worker never sits idle waiting for the zoo: if the marker is missing the job is
#      simply not queued, and re-running this script queues it as soon as it appears.
#
# Re-running is safe and is also how you ADD a worker: the queue file is only created if
# missing, finished sweep LRs / trainings / evals are detected in the logs and skipped,
# and pops are serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Seed the queue without starting any worker (GPU budget is full):
#   bash scripts/run_cifar10_predgen_mpsgmn_sizegen.sh ""
# Launch a worker on a specific GPU:
#   screen -dmS mpsgmn_cnn bash -c 'bash scripts/run_cifar10_predgen_mpsgmn_sizegen.sh "0"'
# Or have it wait for one of your own GPUs to free up first (20 idle minutes required, so
# an inter-job data-loading gap in another queue cannot trigger it):
#   screen -dmS mpsgmn_cnn bash -c 'bash scripts/add_worker_when_free.sh 0 \
#       scripts/run_cifar10_predgen_mpsgmn_sizegen.sh 1500 20'
# Results:  /tmp/mpsgmn_predgen/SUMMARY.md   (appended as each pair finishes)
# Logs:     /tmp/mpsgmn_predgen/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/mpsgmn_predgen
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=cifar10_predgen
GPUS=${1-}                # one worker per GPU; empty = seed the queue only
PREFIX=mpsgmn-deq         # single condition cell, so the prefix is fixed
LABEL="Conv matrix-product ScaleGMN, dup-equiv, forward"
# Stage 3 v2 marker: written by run_cifar10_zoo_gen_v2.sh once w192-w512 are generated.
# NOT /tmp/predgen_zoo/DONE — that is the v1 zoo's marker and has existed since Aug 22.
ZOO_V2_DONE=/tmp/predgen_zoo_v2/DONE
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

C=configs/cifar10_predgen

# ---------------------------------------------------------------- helpers

# Re-verify the train-once-score-twice premise on disk. Anything unexpected here and the
# queue would silently mislabel half the recorded conditions, so this is fatal.
verify_arm_identity() {
    local Z="$ANYDIM_DATA_ROOT/cnn_zoo"
    local FAIL=0

    for r in train val test paired; do
        local A=$(md5sum "$Z/w16_sp/meta_${r}.jsonl" 2>/dev/null | cut -d' ' -f1)
        local B=$(md5sum "$Z/w16_mup16/meta_${r}.jsonl" 2>/dev/null | cut -d' ' -f1)
        if [ -z "$A" ] || [ "$A" != "$B" ]; then
            echo "  FAIL meta_${r}.jsonl differs: sp=$A mup16=$B"; FAIL=1
        fi
    done

    local SA=$(md5sum "$Z/w16_sp/cifar10_predgen_splits.json" 2>/dev/null | cut -d' ' -f1)
    local SB=$(md5sum "$Z/w16_mup16/cifar10_predgen_splits.json" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$SA" ] || [ "$SA" != "$SB" ]; then
        echo "  FAIL split json differs: sp=$SA mup16=$SB"; FAIL=1
    fi

    # the w16 weights themselves must be the same files, not merely equal ones
    local INODES=$($PYTHON - <<'PY'
import json, os, itertools
Z = os.path.expandvars("$ANYDIM_DATA_ROOT/cnn_zoo")
same = diff = 0
for r in ("train", "val", "test", "paired"):
    for line in itertools.islice(open(f"{Z}/w16_sp/meta_{r}.jsonl"), 200):
        p = json.loads(line)["path"]
        same += os.stat(f"{Z}/w16_sp/{p}").st_ino == os.stat(f"{Z}/w16_mup16/{p}").st_ino
        diff += os.stat(f"{Z}/w16_sp/{p}").st_ino != os.stat(f"{Z}/w16_mup16/{p}").st_ino
print(f"{same} {diff}")
PY
    )
    read -r SAME DIFF <<< "$INODES"
    if [ "${DIFF:-1}" != "0" ]; then
        echo "  FAIL w16 weights not hard-linked: $SAME same inode, $DIFF different"; FAIL=1
    fi

    # The configs must differ only in data paths, comments and wandb identifiers -- every
    # model/optimizer arg has to match, or the two arms would not be the same metanetwork.
    for v in v1 v2; do
        local BAD=$(diff "$C/mpsgmn_sizegen_sp_${v}.yml" "$C/mpsgmn_sizegen_mup16_${v}.yml" \
                    | grep -E '^[<>]' \
                    | grep -vE 'cnn_zoo/w[0-9]+_(sp|mup16)' \
                    | grep -vE '^[<>] *(#|name:|group:|tags:|project:|entity:|$)')
        if [ -n "$BAD" ]; then
            echo "  FAIL ${v}: sp/mup16 configs differ beyond data paths:"; echo "$BAD"; FAIL=1
        fi
    done

    if [ "$FAIL" != "0" ]; then
        echo "=== ABORT: the SP/muP16 arms do not share w16, so one training cannot serve"
        echo "    both conditions. Train each arm separately instead. ==="
        return 1
    fi
    echo "  arm identity OK: w16 weights hard-linked ($SAME/$SAME sampled), metas and"
    echo "  split byte-identical (md5 ${SA:0:8}...), configs differ only in data paths"
    return 0
}

run_sweep() {
    local CONF=$1 GPU=$2
    for idx in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
        if grep -q "Best val tau:" "$LOG" 2>/dev/null; then
            echo "[$PREFIX] sweep lr=${LRS[$idx]} already done, skipping"
            continue
        fi
        echo "[$PREFIX] sweep lr=${LRS[$idx]} on GPU $GPU ($(date +%H:%M))"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
            --conf "$CONF" \
            --duplication-equiv True \
            --direction forward \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-predgen-mpsgmn-${LRS[$idx]}" \
            > "$LOG" 2>&1
    done
}

get_best_lr() {
    # selection metric: val Kendall tau at the TRAIN width (w16). No OOD width may
    # influence which LR — or later which checkpoint — is kept.
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
    local OUT=""
    for i in 0 1 2 3 4 5 6; do
        local TAU=$(grep "Best val tau:" "${LOG_DIR}/sweep_${PREFIX}_${i}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [ -z "$TAU" ] && TAU="--"
        OUT="${OUT}${TAU} / "
    done
    echo "${OUT%" / "}"
}

run_full() {
    # trained under the muP16 config; every recorded condition re-scores this checkpoint
    local CONF=$1 GPU=$2 LR=$3
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    if grep -q "Test metrics @ best val tau:" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] full training already done, skipping"
        return
    fi
    echo "[$PREFIX] full training lr=$LR on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --duplication-equiv True \
        --direction forward \
        --wandb True --lr "$LR" \
        --run_name "predgen-sizegen-mpsgmn" \
        > "$LOG" 2>&1
}

ckpt_of() {
    grep -h "Checkpoints will be saved to:" "${LOG_DIR}/full_${PREFIX}.log" 2>/dev/null \
        | tail -1 | awk '{print $NF}'
}

run_eval() {
    # Score the kept checkpoint under one arm's config at one width range.
    # $3 = tag, one of v1_mup16 / v1_sp / v2_mup16 / v2_sp.
    local CONF=$1 GPU=$2 TAG=$3
    local LOG="${LOG_DIR}/eval_${PREFIX}_${TAG}.log"
    if grep -q "mean OOD tau" "$LOG" 2>/dev/null; then
        echo "[$PREFIX/$TAG] eval already done, skipping"
        return
    fi
    local CKPT_DIR=$(ckpt_of)
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX/$TAG] no checkpoint to score (looked for '${CKPT_DIR}/best.pt')"
        return
    fi
    echo "[$PREFIX/$TAG] eval + prediction dump on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        --duplication-equiv True \
        --direction forward \
        --wandb True --run_name "predgen-sizegen-mpsgmn-eval-${TAG}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}_${TAG}.npz" \
        > "$LOG" 2>&1
}

w16_row() {
    # the "  w=   16 (IN ): tau=... R2=... L1=... R2rc=..." line of an eval log
    grep -E "^ +w= *16 \(IN" "${LOG_DIR}/eval_${PREFIX}_${1}.log" 2>/dev/null | tail -1 \
        | sed -E 's/^ +//'
}

check_w16_identity() {
    # w16_sp/ and w16_mup16/ are the same files, and both arms score the same weights, so
    # the w16 row must be identical to the last printed digit. This is the audit trail for
    # having trained once instead of twice.  $1 = version tag (v1|v2)
    local V=$1
    local A=$(w16_row "${V}_mup16")
    local B=$(w16_row "${V}_sp")
    if [ -z "$A" ] || [ -z "$B" ]; then
        echo "[$PREFIX/$V] w16 identity: SKIPPED (an eval log has no w16 row)" >&2
        echo "n/a"; return
    fi
    if [ "$A" = "$B" ]; then
        echo "[$PREFIX/$V] w16 identity: OK ($A)" >&2
        echo "OK"
    else
        echo "[$PREFIX/$V] w16 identity: MISMATCH -- the SP columns cannot be trusted" >&2
        echo "    muP16: $A" >&2
        echo "    SP   : $B" >&2
        echo "MISMATCH"
    fi
}

append_summary() {
    # $1 = human label, $2 = best lr, $3 = tag (v1_sp etc), $4 = w16-identity verdict,
    # $5 = OOD width range for the headline line
    local SECTION=$1 LR=$2 TAG=$3 IDENT=$4 RANGE=$5
    local FULL_LOG="${LOG_DIR}/full_${PREFIX}.log"
    local EVAL_LOG="${LOG_DIR}/eval_${PREFIX}_${TAG}.log"
    local CKPT_DIR=$(ckpt_of)
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local BEST_VAL=$(grep "Best val tau:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local EPOCHS=$(grep -c "val_tau=" "$FULL_LOG" 2>/dev/null)
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$EVAL_LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    {
        echo ""
        echo "### $SECTION"
        echo ""
        echo "- best LR: **$LR** (sweep val taus @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_taus))"
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "  — trained once under the muP16 v1 config and scored under both arms; see \`w16 identity\` below"
        echo "- checkpoint: \`$(basename "${CKPT_DIR:-n/a}")\`"
        echo "- best val tau (w16): ${BEST_VAL:-n/a}, epochs run: ${EPOCHS:-n/a}"
        echo "- eval/dump run: \`${EVAL_ID:-n/a}\` (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- w16 identity vs the other arm: **${IDENT}**"
        echo "- predictions: \`${LOG_DIR}/preds_${PREFIX}_${TAG}.npz\`, logs: \`${FULL_LOG}\`, \`${EVAL_LOG}\`"
        echo ""
        echo "| Width | tau | R2 | L1 | R2_recal |"
        echo "|---|---|---|---|---|"
        # the eval dump prints "  w=   16 (IN ): tau=0.9315 R2=+0.991 L1=0.0077 R2rc=+0.991"
        sed -n '/Per-width test metrics:/,$p' "$EVAL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/'
        echo ""
        # headline metric, and the line scripts/watch_experiments.sh puts in Slack
        echo "OOD mean tau (${RANGE}): $(grep -m1 'mean OOD tau' "$EVAL_LOG" 2>/dev/null | awk '{print $NF}')"
    } >> "$SUMMARY.part.${TAG}"
    # serialize the append so concurrent workers don't interleave
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.${TAG}' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.${TAG}"
}

# ---------------------------------------------------------------- job queue

pop_job() {
    # atomically remove and echo the first queue line ("" if empty)
    flock "$QUEUE_LOCK" -c "
        JOB=\$(head -1 '$QUEUE' 2>/dev/null)
        if [ -n \"\$JOB\" ]; then tail -n +2 '$QUEUE' > '$QUEUE.tmp' && mv '$QUEUE.tmp' '$QUEUE'; fi
        printf '%s' \"\$JOB\""
}

push_job() {
    flock "$QUEUE_LOCK" -c "printf '%s\n' '$1' >> '$QUEUE'"
}

maybe_queue_v2() {
    # Queue the w16-w512 re-score, but only once its zoo exists. Idempotent: skips if the
    # job is already queued or already scored, so it is safe to call on every invocation
    # and at the end of the training job.
    if grep -q "mean OOD tau" "${LOG_DIR}/eval_${PREFIX}_v2_sp.log" 2>/dev/null; then
        return
    fi
    if grep -q "^evalv2|" "$QUEUE" 2>/dev/null; then
        return
    fi
    if [ ! -f "$ZOO_V2_DONE" ]; then
        echo "[v2] Stage 3 v2 zoo not finished ($ZOO_V2_DONE missing) — the w16-w512"
        echo "     re-score is NOT queued. Re-run this script once the marker appears;"
        echo "     it will queue and run the re-score without retraining anything."
        return
    fi
    push_job "evalv2|$C/mpsgmn_sizegen_mup16_v2.yml|$C/mpsgmn_sizegen_sp_v2.yml"
    echo "[v2] Stage 3 v2 zoo is done — queued the w16-w512 re-score."
}

worker() {
    local GPU=$1
    while true; do
        local SPEC=$(pop_job)
        [ -z "$SPEC" ] && break
        IFS='|' read -r KIND MUP_CONF SP_CONF <<< "$SPEC"
        echo "[GPU $GPU] picked up $KIND ($(date +%H:%M))"
        if [ "$KIND" = "train" ]; then
            run_sweep "$MUP_CONF" "$GPU"
            local LR=$(get_best_lr)
            echo "[$PREFIX] best LR = $LR"
            run_full "$MUP_CONF" "$GPU" "$LR"
            run_eval "$MUP_CONF" "$GPU" v1_mup16
            run_eval "$SP_CONF"  "$GPU" v1_sp
            local IDENT=$(check_w16_identity v1)
            append_summary "$LABEL, muP16 (v1, w16–w128)" "$LR" v1_mup16 "$IDENT" "w32-w128"
            append_summary "$LABEL, SP (v1, w16–w128)"    "$LR" v1_sp    "$IDENT" "w32-w128"
            echo "[$PREFIX] v1 done, 2 conditions recorded ($(date))"
            maybe_queue_v2
        elif [ "$KIND" = "evalv2" ]; then
            local LR=$(get_best_lr)
            run_eval "$MUP_CONF" "$GPU" v2_mup16
            run_eval "$SP_CONF"  "$GPU" v2_sp
            local IDENT=$(check_w16_identity v2)
            append_summary "$LABEL, muP16 (v2, w16–w512)" "$LR" v2_mup16 "$IDENT" "w32-w512"
            append_summary "$LABEL, SP (v2, w16–w512)"    "$LR" v2_sp    "$IDENT" "w32-w512"
            echo "[$PREFIX] v2 done, 2 conditions recorded ($(date))"
        else
            echo "[GPU $GPU] unknown job kind '$KIND', skipping"
        fi
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    echo "=== verifying the SP/muP16 w16 identity that lets one training serve both arms ==="
    verify_arm_identity || exit 1
    # <kind>|<mup16 conf>|<sp conf>
    cat > "$QUEUE" <<EOF
train|$C/mpsgmn_sizegen_mup16_v1.yml|$C/mpsgmn_sizegen_sp_v1.yml
EOF
    {
        echo "# Conv matrix-product ScaleGMN — CNN accuracy prediction, size generalization (train w16)"
        echo ""
        echo "Started: $(date)."
        echo "One (model, variant) cell — conv matrix-product ScaleGMN, dup-equiv, \`--direction"
        echo "forward\` — recorded as four conditions: × {SP, muP16} input networks × {v1 (w16–w128),"
        echo "v2 (w16–w512)} width ranges. There is no baseline variant of this MSG function."
        echo ""
        echo "**One training.** Base width = train width makes the two arms' w16 inputs the same data"
        echo "(w16 weights hard-linked, metas and split byte-identical, configs differing only in data"
        echo "paths), which Stage 3 confirmed by training both arms and measuring \`max |dw| = 0.00e+00\`."
        echo "So the model is trained once under the muP16 v1 config and that checkpoint is scored under"
        echo "each arm's config at each width range. Every pair carries a \`w16 identity\` line: the two"
        echo "arms' w16 rows must match to the last digit, since that row is the same data through the"
        echo "same weights."
        echo ""
        echo "LR selected on **val Kendall tau at w16** from a 7-value ×2 grid (6.25e-5 … 4e-3),"
        echo "then 200 epochs at that LR; the kept checkpoint is the best-val-tau epoch, so no OOD"
        echo "width influences selection. Same grid / epochs / splits / readout as the ten forward"
        echo "Stage 3 conditions, so these are drop-in columns in the same per-width tables."
        echo ""
        echo "The v2 (w16–w512) pair waits for the Stage 3 v2 zoo marker \`$ZOO_V2_DONE\`; it is a"
        echo "separate queue job, so no GPU sits idle waiting for the zoo."
    } > $SUMMARY
    echo "=== Seeded queue with the training job (= 2 conditions; +2 once the v2 zoo lands) ==="
fi

# Pick up the v2 re-score as soon as its zoo finishes, on any invocation.
maybe_queue_v2

if [ -z "${GPUS// /}" ]; then
    echo "=== No GPUs given: queue seeded at $QUEUE, no workers started ==="
    exit 0
fi

echo "=== mpsgmn predgen size-gen: starting workers on GPUs [$GPUS] $(date) ==="

PIDS=()
for GPU in $GPUS; do
    worker "$GPU" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

if [ ! -s "$QUEUE" ] && grep -q "mean OOD tau" "${LOG_DIR}/eval_${PREFIX}_v2_sp.log" 2>/dev/null; then
    echo "" >> $SUMMARY
    echo "Finished: $(date)" >> $SUMMARY
fi

echo "=== Workers done $(date). Summary: $SUMMARY ==="

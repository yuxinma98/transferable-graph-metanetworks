#!/bin/bash
# Stage 3 of the CNN accuracy-prediction study, BIDIRECTIONAL variants: size
# generalization on our multi-width CNN zoo -- train on w16 only, test on w16-w128.
#
# The bidirectional counterpart of run_svhn_predgen_sizegen_v1.sh (same LR grid, epochs,
# patience, splits, readout and selection metric), so its numbers are drop-in columns in
# the same per-width tables. SIX conditions, all `symmetry: permutation`:
#   plain GMN x {SP, muP16} x {baseline, dup-equiv}                          = 4
#   conv matrix-product GMN x {SP, muP16}  (dup-equiv only by construction)  = 2
#
# The four `symmetry: scale` (ScaleGMN) bidirectional conditions the CIFAR-10 study also
# lists are DELIBERATELY OUT OF SCOPE here, not pending: on CIFAR-10 they NaN by training
# step ~13 at every LR tried. That is the elementwise-division backward instability the
# ScaleGMN paper's own Appendix A.2 reports for positive-scale symmetry and never resolves
# (its one bidirectional result uses the sign-symmetric sine activation, where 1/q = q
# sidesteps the division). The cause is architectural -- float32 reciprocal of near-zero
# conv weights -- not dataset-specific, so SVHN-GS is expected to fail identically and
# nothing is gained by spending the GPU time to confirm it.
# See src/models/bidir_reciprocal.py.
#
# Bidirectional CNN graphs are otherwise fine for the permutation models: cnn_to_tg_data
# fills `bw_edge_attr` with reciprocal(edge_attr), which is largely `inf` on CNN graphs,
# but GNN_layer's bidirectional forward reads only `batch.bw_edge_index`, so the field is
# dead. This was smoke-tested on the CIFAR-10 side (all pairs build and forward at w16 and
# w128 with finite outputs while bw_edge_attr carried 20480 infs at w128); the graph
# construction is dataset-independent, so it carries over.
#
# THREE trainings, SIX conditions -- the cost fix the CIFAR-10 run identified too late
# --------------------------------------------------------------------------------------
# The CIFAR-10 bidirectional run trained both arms of every condition and then measured
# that each SP/muP16 pair had produced bit-identical weights (`max |dw| = 0.00e+00` on all
# five pairs), because base width = train width makes the two arms' w16 inputs the same
# data. So half of that 3.2-day drain was redundant, and its results doc recommends one
# training plus one eval-only re-score per pair. This runner is built that way from the
# start, so each queue job is a (model, variant) PAIR:
#     sweep + train under the muP16 config  ->  score that checkpoint TWICE,
#     once under the muP16 config and once under the SP config (--eval-only-ckpt).
# The two scores differ only at the OOD widths, which is exactly the SP-vs-muP16
# input-network contrast the experiment is asking about.
#
# The premise is re-verified on disk before anything is queued (verify_arm_identity):
#   - w16_sp/ and w16_mup16/ weight files are hard links (same inode)
#   - their meta_{train,val,test,paired}.jsonl are byte-identical
#   - their svhn_predgen_splits.json are byte-identical (md5 printed at launch)
#   - the SP and muP16 configs differ only in data paths, comments and wandb names
# and each job then prints a `w16 identity: OK` line -- the SP re-score must reproduce the
# muP16 run's w16 row exactly, since that row is literally the same data through the same
# weights. A MISMATCH means the arms are not sharing what this script assumes and the SP
# columns must not be trusted.
#
# Protocol per job, identical to the forward v1 runs:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), MAX_STEPS each, selected on
#      **val Kendall tau at w16** (in-distribution only -- no OOD width influences it)
#   2. full training (200 epochs, patience from the config) at the best sweep LR
#   3. eval-only re-score + prediction dump under the muP16 config
#   4. eval-only re-score + prediction dump under the SP config, from the SAME checkpoint
#   5. w16 identity check between 3 and 4, then two summary sections appended
# Step 3 adds no test widths (all six already scored every epoch), so it doubles as a
# checkpoint round-trip check against step 2's numbers.
#
# The three jobs are pushed onto a FIFO queue drained by one worker per GPU. Re-running is
# safe and is also how you ADD a worker: the queue file is only created if missing,
# finished sweep LRs / trainings / evals are detected in the logs and skipped, and pops
# are serialized with flock. To hard-reset, delete $LOG_DIR/queue.txt.
#
# Launch:
#   screen -dmS svhn_predgen_bidir bash -c 'bash scripts/run_svhn_predgen_sizegen_v1_bidir.sh "5"'
# Add a worker later, once one of your own GPUs frees up (20 idle minutes required, so an
# inter-job data-loading gap in another queue cannot trigger it):
#   screen -dmS svhn_predgen_bidir_w2 bash -c 'bash scripts/add_worker_when_free.sh 4 \
#       scripts/run_svhn_predgen_sizegen_v1_bidir.sh 1500 20'
# Results:  /tmp/predgen_svhn_sizegen_v1_bidir/SUMMARY.md   (appended as each job finishes)
# Logs:     /tmp/predgen_svhn_sizegen_v1_bidir/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/predgen_svhn_sizegen_v1_bidir
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=svhn_predgen
GPUS=${1:-"5"}            # one worker per GPU; keep >=2 GPUs vacant for other users
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

# ---------------------------------------------------------------- helpers

# always pass the flag explicitly, so the checkpoint records which mode it was trained in
deq_flag() { echo "--duplication-equiv $1"; }

# Re-verify the train-once-score-twice premise on disk. Anything unexpected here and the
# three-job queue would silently mislabel three of the six conditions, so this is fatal.
verify_arm_identity() {
    local Z="$ANYDIM_DATA_ROOT/svhn_cnn_zoo"
    local FAIL=0

    for r in train val test paired; do
        local A=$(md5sum "$Z/w16_sp/meta_${r}.jsonl" 2>/dev/null | cut -d' ' -f1)
        local B=$(md5sum "$Z/w16_mup16/meta_${r}.jsonl" 2>/dev/null | cut -d' ' -f1)
        if [ -z "$A" ] || [ "$A" != "$B" ]; then
            echo "  FAIL meta_${r}.jsonl differs: sp=$A mup16=$B"; FAIL=1
        fi
    done

    local SA=$(md5sum "$Z/w16_sp/svhn_predgen_splits.json" 2>/dev/null | cut -d' ' -f1)
    local SB=$(md5sum "$Z/w16_mup16/svhn_predgen_splits.json" 2>/dev/null | cut -d' ' -f1)
    if [ -z "$SA" ] || [ "$SA" != "$SB" ]; then
        echo "  FAIL split json differs: sp=$SA mup16=$SB"; FAIL=1
    fi

    # the w16 weights themselves must be the same files, not merely equal ones
    local INODES=$($PYTHON - <<'PY'
import json, os, itertools
Z = os.path.expandvars("$ANYDIM_DATA_ROOT/svhn_cnn_zoo")
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
    for m in scalegmn gmn mpgmn; do
        local BAD=$(diff "configs/svhn_predgen/${m}_sizegen_sp_v1.yml" \
                         "configs/svhn_predgen/${m}_sizegen_mup16_v1.yml" \
                    | grep -E '^[<>]' \
                    | grep -vE 'cnn_zoo/w[0-9]+_(sp|mup16)' \
                    | grep -vE '^[<>] *(#|name:|group:|tags:|project:|entity:|$)')
        if [ -n "$BAD" ]; then
            echo "  FAIL ${m}: sp/mup16 configs differ beyond data paths:"; echo "$BAD"; FAIL=1
        fi
    done

    if [ "$FAIL" != "0" ]; then
        echo "=== ABORT: the SP/muP16 arms do not share w16, so one training cannot serve"
        echo "    both conditions. Use run_svhn_predgen_sizegen_v1.sh’s ten-job layout"
        echo "    (with --direction bidirectional) instead. ==="
        return 1
    fi
    echo "  arm identity OK: w16 weights hard-linked ($SAME/$SAME sampled), metas and"
    echo "  split byte-identical (md5 ${SA:0:8}...), configs differ only in data paths"
    return 0
}

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
            --direction bidirectional \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-svhn-predgen-v1-bidir-${PREFIX}-${LRS[$idx]}" \
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
    # trained under the muP16 config; the SP condition re-scores this same checkpoint
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
        --direction bidirectional \
        --wandb True --lr "$LR" \
        --run_name "svhn-predgen-sizegen-v1-bidir-${PREFIX}" \
        > "$LOG" 2>&1
}

ckpt_of() {
    grep -h "Checkpoints will be saved to:" "${LOG_DIR}/full_${1}.log" 2>/dev/null \
        | tail -1 | awk '{print $NF}'
}

run_eval() {
    # Score the kept checkpoint under one arm's config. $5 = arm tag (mup16 | sp).
    # For mup16 this must reproduce run_full's "Test metrics @ best val tau" exactly (a
    # checkpoint round-trip check); for sp only the OOD widths change. Both dump the
    # per-width (pred, actual) arrays the Stage 3c diagnostics need.
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4 ARM=$5
    local LOG="${LOG_DIR}/eval_${PREFIX}_${ARM}.log"
    if grep -q "mean OOD tau" "$LOG" 2>/dev/null; then
        echo "[$PREFIX/$ARM] eval already done, skipping"
        return
    fi
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    if [ -z "$CKPT_DIR" ] || [ ! -f "${CKPT_DIR}/best.pt" ]; then
        echo "[$PREFIX/$ARM] no checkpoint to score (looked for '${CKPT_DIR}/best.pt')"
        return
    fi
    echo "[$PREFIX/$ARM] eval + prediction dump on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        --eval-only-ckpt "${CKPT_DIR}/best.pt" \
        $(deq_flag "$DEQ") \
        --direction bidirectional \
        --wandb True --run_name "svhn-predgen-sizegen-v1-bidir-eval-${PREFIX}-${ARM}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}_${ARM}.npz" \
        > "$LOG" 2>&1
}

w16_row() {
    # the "  w=   16 (IN ): tau=... R2=... L1=... R2rc=..." line of an eval log
    grep -E "^ +w= *16 \(IN" "${LOG_DIR}/eval_${1}_${2}.log" 2>/dev/null | tail -1 \
        | sed -E 's/^ +//'
}

check_w16_identity() {
    # w16_sp/ and w16_mup16/ are the same files, and both arms score the same weights, so
    # the w16 row must be identical to the last printed digit. This is the audit trail for
    # having trained once instead of twice.
    local PREFIX=$1
    local A=$(w16_row "$PREFIX" mup16)
    local B=$(w16_row "$PREFIX" sp)
    if [ -z "$A" ] || [ -z "$B" ]; then
        echo "[$PREFIX] w16 identity: SKIPPED (an eval log has no w16 row)"
        echo "n/a"; return
    fi
    if [ "$A" = "$B" ]; then
        echo "[$PREFIX] w16 identity: OK ($A)" >&2
        echo "OK"
    else
        echo "[$PREFIX] w16 identity: MISMATCH -- the SP columns cannot be trusted" >&2
        echo "    muP16: $A" >&2
        echo "    SP   : $B" >&2
        echo "MISMATCH"
    fi
}

append_summary() {
    # $1 = human label, $2 = prefix, $3 = best lr, $4 = arm tag, $5 = w16-identity verdict
    local LABEL=$1 PREFIX=$2 LR=$3 ARM=$4 IDENT=$5
    local FULL_LOG="${LOG_DIR}/full_${PREFIX}.log"
    local EVAL_LOG="${LOG_DIR}/eval_${PREFIX}_${ARM}.log"
    local CKPT_DIR=$(ckpt_of "$PREFIX")
    local RUN_ID=$(basename "${CKPT_DIR:-_unknown}" | sed 's/.*_//')
    local BEST_VAL=$(grep "Best val tau:" "$FULL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local EPOCHS=$(grep -c "val_tau=" "$FULL_LOG" 2>/dev/null)
    local EVAL_ID=$(grep -o "${WANDB_PROJECT}/runs/[a-z0-9]*" "$EVAL_LOG" 2>/dev/null | tail -1 | sed 's|.*/||')
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- best LR: **$LR** (sweep val taus @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_taus "$PREFIX"))"
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "  — trained once under the muP16 config and scored under both arms; see \`w16 identity\` below"
        echo "- checkpoint: \`$(basename "${CKPT_DIR:-n/a}")\`"
        echo "- best val tau (w16): ${BEST_VAL:-n/a}, epochs run: ${EPOCHS:-n/a}"
        echo "- eval/dump run: \`${EVAL_ID:-n/a}\` (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${EVAL_ID})"
        echo "- w16 identity vs the other arm: **${IDENT}**"
        echo "- predictions: \`${LOG_DIR}/preds_${PREFIX}_${ARM}.npz\`, logs: \`${FULL_LOG}\`, \`${EVAL_LOG}\`"
        echo ""
        echo "| Width | tau | R2 | L1 | R2_recal |"
        echo "|---|---|---|---|---|"
        # the eval dump prints "  w=   16 (IN ): tau=0.9315 R2=+0.991 L1=0.0077 R2rc=+0.991"
        sed -n '/Per-width test metrics:/,$p' "$EVAL_LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT) *\): +tau=([-0-9.]+) R2=([-+0-9.]+) L1=([0-9.]+) R2rc=([-+0-9.]+).*/| w\1 (\2) | \3 | \4 | \5 | \6 |/'
        echo ""
        # headline metric, and the line scripts/watch_experiments.sh puts in Slack
        echo "OOD mean tau (w32-w128): $(grep -m1 'mean OOD tau' "$EVAL_LOG" 2>/dev/null | awk '{print $NF}')"
    } >> "$SUMMARY.part.${PREFIX}.${ARM}"
    # serialize the append so concurrent workers don't interleave
    flock "$QUEUE_LOCK" -c "cat '$SUMMARY.part.${PREFIX}.${ARM}' >> '$SUMMARY'"
    rm -f "$SUMMARY.part.${PREFIX}.${ARM}"
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
        IFS='|' read -r MUP_CONF SP_CONF DEQ PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        run_sweep "$MUP_CONF" "$DEQ" "$PREFIX" "$GPU"
        local LR=$(get_best_lr "$PREFIX")
        echo "[$PREFIX] best LR = $LR"
        run_full "$MUP_CONF" "$DEQ" "$PREFIX" "$GPU" "$LR"
        run_eval "$MUP_CONF" "$DEQ" "$PREFIX" "$GPU" mup16
        run_eval "$SP_CONF"  "$DEQ" "$PREFIX" "$GPU" sp
        local IDENT=$(check_w16_identity "$PREFIX")
        append_summary "$LABEL, muP16" "$PREFIX" "$LR" mup16 "$IDENT"
        append_summary "$LABEL, SP"    "$PREFIX" "$LR" sp    "$IDENT"
        echo "[$PREFIX] done, 2 conditions recorded ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

touch "$QUEUE_LOCK"

# Only seed the queue on a fresh start, so re-running this script just adds workers.
if [ ! -f "$QUEUE" ]; then
    echo "=== verifying the SP/muP16 w16 identity that lets one training serve both arms ==="
    verify_arm_identity || exit 1
    C=configs/svhn_predgen
    # <mup16 conf>|<sp conf>|<dup-equiv>|<prefix>|<label>
    # Ordered most-informative first: dup-equiv is the headline variant and the conv
    # matrix-product GMN was v1's best OOD condition, so the interesting bidirectional
    # comparisons complete before the baselines.
    cat > "$QUEUE" <<EOF
$C/mpgmn_sizegen_mup16_v1.yml|$C/mpgmn_sizegen_sp_v1.yml|True|mpgmn-deq-bidir|Conv matrix-product GMN, dup-equiv, bidirectional
$C/gmn_sizegen_mup16_v1.yml|$C/gmn_sizegen_sp_v1.yml|True|gmn-deq-bidir|Plain GMN, dup-equiv, bidirectional
$C/gmn_sizegen_mup16_v1.yml|$C/gmn_sizegen_sp_v1.yml|False|gmn-base-bidir|Plain GMN, baseline, bidirectional
EOF
    {
        echo "# CNN accuracy prediction — size generalization v1, BIDIRECTIONAL (train w16, test w16–w128)"
        echo ""
        echo "Started: $(date)."
        echo "Six conditions, all \`symmetry: permutation\`: plain GMN × {SP, muP16} ×"
        echo "{baseline, dup-equiv}, plus the conv matrix-product GMN × {SP, muP16}"
        echo "(dup-equiv only), all \`--direction bidirectional\` — the counterpart of the"
        echo "forward v1 runs in \`/tmp/predgen_svhn_sizegen_v1/SUMMARY.md\`, so these are"
        echo "drop-in columns."
        echo ""
        echo "The four \`symmetry: scale\` (ScaleGMN) bidirectional conditions are **deliberately"
        echo "out of scope**, not missing: on CIFAR-10 they NaN by training step ~13 at every LR"
        echo "tried, which is the elementwise-division instability the ScaleGMN paper's own"
        echo "Appendix A.2 reports for positive-scale symmetry and never resolves. The cause is"
        echo "architectural (reciprocal of near-zero conv weights in float32), not dataset-"
        echo "specific, so SVHN-GS is expected to hit it identically. See"
        echo "§ Bidirectional scale equivariance."
        echo ""
        echo "**Three trainings, six conditions.** Base width = train width makes the two arms'"
        echo "w16 inputs the same data (w16 weights hard-linked, metas and split byte-identical,"
        echo "configs differing only in data paths), which v1 confirmed by training both arms and"
        echo "measuring \`max |dw| = 0.00e+00\`. So each (model, variant) is trained once under the"
        echo "muP16 config and that checkpoint is scored under both arms. Every condition carries a"
        echo "\`w16 identity\` line: the two arms' w16 rows must match to the last digit, since that"
        echo "row is the same data through the same weights."
        echo ""
        echo "LR selected on **val Kendall tau at w16** from a 7-value ×2 grid (6.25e-5 … 4e-3),"
        echo "then 200 epochs at that LR; the kept checkpoint is the best-val-tau epoch, so no OOD"
        echo "width influences selection."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all six are done."
    } > $SUMMARY
    echo "=== Seeded queue with 3 jobs (= 6 conditions) ==="
fi

echo "=== predgen size-gen v1 BIDIR: starting workers on GPUs [$GPUS] $(date) ==="

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

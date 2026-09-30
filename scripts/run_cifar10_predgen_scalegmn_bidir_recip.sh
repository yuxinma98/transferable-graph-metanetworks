#!/bin/bash
# DO NOT RUN -- kept only as a record of the attempt.
#
# Every LR NaNs by training step ~13: the smallest nonzero |W| across cnn_zoo/w16_mup16 is
# ~1.2e-15, whose reciprocal (~8.4e14) overflows float32 inside EquivariantNet's unnormalized
# io-node branch after a few rounds of bw_update_edge_attr_fn threading. This is not a bug --
# the ScaleGMN paper (arXiv:2406.10685, Appendix A.2) defines the identical construction and
# reports the same class of instability for positive-scale symmetry, works around it only for
# symmetry: sign (1/q=q needs no division), and reports no bidirectional positive-scale result
# anywhere in the paper. See src/models/bidir_reciprocal.py and
# results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md for the full trace.
# Dropped as out of scope, not queued.
#
# Stage 3 bidir of the CNN accuracy-prediction study, ScaleGMN conditions only --
# RETRAIN with the reciprocal backward edge feature.
#
# Why this exists
# ---------------
# A bidirectional `symmetry: scale` model is *defined* with the reciprocal backward edge
# feature 1/W[u, v]: without it a backward message into u carries lam_v^2/lam_u where
# equivariance requires lam_u, so the model is not scale-equivariant at all. Upstream asks
# for exactly that feature with `reciprocal: True`, assigns it at models.py:305, and never
# applies it (models.py:418 hardcodes `bw_edge_attr = batch.edge_attr`). So the four
# ScaleGMN conditions of run_cifar10_predgen_sizegen_v1_bidir.sh -- rows 2, 3, 5 and 7 of
# the Experiment 3 bidir table -- are not measurements of a bidirectional ScaleGMN. They
# are replaced by this script.
#
# src/models/bidir_reciprocal.py installs the feature and works unchanged on conv graphs:
# 20.7% of the R^9 edge components at c=16 are structurally zero (the dense head's 1x1
# kernels are zero-padded), so plain torch.reciprocal is `inf` there, but under the
# 1/0 := 0 pseudo-inverse the feature is finite and law-preserving (zero is a fixed point
# of the rescaling). scripts/check_scale_equiv_cnn.py Test 4: the shipped code path fails
# at 2.0e-03 (node) / 2.7e-02 (graph lw-mean), with the fix installed it is exactly
# equivariant at 1.5e-08 / 1.9e-07 -- the float32 floor of the forward arm.
# train_sizegen_predgen.py installs it by default for bidirectional scale models, so this
# runner is the v1 bidir runner with nothing changed but the ScaleGMN job list; the LR
# grid, epochs, patience, splits, readout and selection metric are identical, so its four
# conditions are drop-in replacements in the same per-width tables.
#
# The `symmetry: permutation` conditions (plain GMN, conv matrix-product GMN) need no
# retrain: install_bidir_reciprocal is a no-op there by construction, and those models have
# no homogeneity structure for it to fix. The conv matrix-product ScaleGMN is forward-only
# in every experiment, so it has no bidirectional condition to redo.
#
# TWO trainings, FOUR conditions
# ------------------------------
# Same trick as the v1 bidir runner, and it is why this retrain costs ~30 GPU-h instead of
# ~60: base width = train width, so cnn_zoo/w16_mup16/ is hard-linked from cnn_zoo/w16_sp/
# and both carry the same split JSON. The SP and muP16 arms of a given variant therefore
# consume byte-identical inputs and produce bit-identical weights (v1 measured
# `max |dw| = 0.00e+00` on all five pairs after paying to train both). Each job trains once
# under the muP16 config and scores that checkpoint under both arms with --eval-only-ckpt,
# which loads only model_state_dict into a net built from the CLI --conf.
#   verify_arm_identity() re-verifies the premise on disk before anything is queued
#   (inodes, meta_*.jsonl md5s, split md5, and a config diff that must reduce to data
#   paths), and every condition prints a `w16 identity` line: the two arms' w16 rows must
#   match to the last digit, since that row is the same data through the same weights.
#
# Protocol per job, identical to the v1 bidir runs:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), MAX_STEPS each, selected on
#      **val Kendall tau at w16** (in-distribution only)
#   2. full training (200 epochs, patience from the config) at the best sweep LR
#   3. eval-only re-score + prediction dump under the muP16 config
#   4. eval-only re-score + prediction dump under the SP config, from the SAME checkpoint
#   5. w16 identity check between 3 and 4, then two summary sections appended
# Every sweep, training and eval log is additionally checked for the
# "Installed reciprocal backward edge features" line -- if the fix ever fails to install,
# this runner would silently reproduce the invalid runs it exists to replace.
#
# Queue drained by one worker per GPU, resumable, pops serialized with flock. NOTE
# `GPUS=${1-}`: passing an EMPTY first argument seeds the queue and starts no worker, so
# the queue can be laid down while the GPU budget is full and drained by watchers later.
#
# Launch:
#   bash scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh ""          # seed only
#   screen -dmS predgen_recip bash -c 'bash scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh "5"'
# Add a worker once one of your own GPUs frees up (20 idle minutes required):
#   screen -dmS predgen_recip_w bash -c 'bash scripts/add_worker_when_free.sh 4 \
#       scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh 1500 20'
# Results:  /tmp/predgen_sizegen_v1_bidir_recip/SUMMARY.md   (appended as each job finishes)
# Logs:     /tmp/predgen_sizegen_v1_bidir_recip/

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/predgen_sizegen_v1_bidir_recip
SUMMARY=$LOG_DIR/SUMMARY.md
WANDB_PROJECT=cifar10_predgen
GPUS=${1-}                # empty => seed the queue only; keep >=2 GPUs vacant for others
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

RECIP_LINE="Installed reciprocal backward edge features"

# ---------------------------------------------------------------- helpers

# always pass the flag explicitly, so the checkpoint records which mode it was trained in
deq_flag() { echo "--duplication-equiv $1"; }

# The whole point of this retrain. A log that ran without the fix is the invalid model
# again, so say so loudly rather than recording the number.
check_recip_installed() {
    local LOG=$1 WHAT=$2
    [ -f "$LOG" ] || return 0
    if ! grep -q "$RECIP_LINE" "$LOG"; then
        echo "  !! WARNING $WHAT ran WITHOUT the reciprocal backward edge feature"
        echo "     ($LOG has no '$RECIP_LINE' line) -- that log reproduces the invalid"
        echo "     model this retrain replaces. Delete it and re-run."
    fi
}

# Re-verify the train-once-score-twice premise on disk. Anything unexpected here and the
# queue would silently mislabel two of the four conditions, so this is fatal.
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

    # The two arms' configs must differ only in data paths, comments and wandb identifiers.
    local BAD=$(diff "configs/cifar10_predgen/scalegmn_sizegen_sp_v1.yml" \
                     "configs/cifar10_predgen/scalegmn_sizegen_mup16_v1.yml" \
                | grep -E '^[<>]' \
                | grep -vE 'cnn_zoo/w[0-9]+_(sp|mup16)' \
                | grep -vE '^[<>] *(#|name:|group:|tags:|project:|entity:|$)')
    if [ -n "$BAD" ]; then
        echo "  FAIL scalegmn: sp/mup16 configs differ beyond data paths:"; echo "$BAD"; FAIL=1
    fi

    # ... and both must actually ask for the reciprocal, or install_bidir_reciprocal is a
    # no-op and this retrain would reproduce the runs it replaces.
    for arm in sp mup16; do
        if ! grep -qE '^\s*reciprocal:\s*True' "configs/cifar10_predgen/scalegmn_sizegen_${arm}_v1.yml"; then
            echo "  FAIL scalegmn_sizegen_${arm}_v1.yml does not set reciprocal: True"; FAIL=1
        fi
    done

    if [ "$FAIL" != "0" ]; then
        echo "=== ABORT: the SP/muP16 arms do not share w16 (or the configs do not ask for"
        echo "    the reciprocal), so one training cannot serve both conditions. ==="
        return 1
    fi
    echo "  arm identity OK: w16 weights hard-linked ($SAME/$SAME sampled), metas and"
    echo "  split byte-identical (md5 ${SA:0:8}...), configs differ only in data paths,"
    echo "  both arms set reciprocal: True"
    return 0
}

run_sweep() {
    local CONF=$1 DEQ=$2 PREFIX=$3 GPU=$4
    for idx in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
        if grep -q "Best val tau:" "$LOG" 2>/dev/null; then
            echo "[$PREFIX] sweep lr=${LRS[$idx]} already done, skipping"
            check_recip_installed "$LOG" "sweep lr=${LRS[$idx]}"
            continue
        fi
        echo "[$PREFIX] sweep lr=${LRS[$idx]} on GPU $GPU ($(date +%H:%M))"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
            --conf "$CONF" \
            $(deq_flag "$DEQ") \
            --direction bidirectional \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-predgen-v1-bidir-recip-${PREFIX}-${LRS[$idx]}" \
            > "$LOG" 2>&1
        check_recip_installed "$LOG" "sweep lr=${LRS[$idx]}"
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
        check_recip_installed "$LOG" "full training"
        return
    fi
    echo "[$PREFIX] full training lr=$LR on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen_predgen.py \
        --conf "$CONF" \
        $(deq_flag "$DEQ") \
        --direction bidirectional \
        --wandb True --lr "$LR" \
        --run_name "predgen-sizegen-v1-bidir-recip-${PREFIX}" \
        > "$LOG" 2>&1
    check_recip_installed "$LOG" "full training"
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
        check_recip_installed "$LOG" "eval ($ARM)"
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
        --wandb True --run_name "predgen-sizegen-v1-bidir-recip-eval-${PREFIX}-${ARM}" \
        --save-predictions "${LOG_DIR}/preds_${PREFIX}_${ARM}.npz" \
        > "$LOG" 2>&1
    check_recip_installed "$LOG" "eval ($ARM)"
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
    local RECIP="yes"
    grep -q "$RECIP_LINE" "$FULL_LOG" 2>/dev/null || RECIP="**NO — this row is invalid**"
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- best LR: **$LR** (sweep val taus @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_taus "$PREFIX"))"
        echo "- training run: \`${RUN_ID}\`  (https://wandb.ai/yuxinma/${WANDB_PROJECT}/runs/${RUN_ID})"
        echo "  — trained once under the muP16 config and scored under both arms; see \`w16 identity\` below"
        echo "- reciprocal backward edge feature installed: ${RECIP}"
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
    C=configs/cifar10_predgen
    # <mup16 conf>|<sp conf>|<dup-equiv>|<prefix>|<label>
    # dup-equiv first: it is the headline variant and the stronger of the two ScaleGMN
    # conditions in every arm of the study so far.
    cat > "$QUEUE" <<EOF
$C/scalegmn_sizegen_mup16_v1.yml|$C/scalegmn_sizegen_sp_v1.yml|True|sgmn-deq-bidir-recip|ScaleGMN, dup-equiv, bidirectional (reciprocal fix)
$C/scalegmn_sizegen_mup16_v1.yml|$C/scalegmn_sizegen_sp_v1.yml|False|sgmn-base-bidir-recip|ScaleGMN, baseline, bidirectional (reciprocal fix)
EOF
    {
        echo "# CNN accuracy prediction — bidirectional ScaleGMN, RETRAIN with the reciprocal backward edge feature"
        echo ""
        echo "Started: $(date)."
        echo "The four bidirectional \`symmetry: scale\` conditions of Experiment 3 bidir"
        echo "(rows 2, 3, 5, 7 of \`/tmp/predgen_sizegen_v1_bidir/SUMMARY.md\`) were trained"
        echo "without the reciprocal backward edge feature \`reciprocal: True\` asks for, so"
        echo "they were not scale-equivariant at all (results/VERIFICATION_gmn_properties.md,"
        echo "§ Scale equivariance on CNN graphs, Test 4). These runs install"
        echo "src/models/bidir_reciprocal.py and change nothing else: same configs, LR grid,"
        echo "epochs/patience, splits, readout and w16-only selection rule, so they are"
        echo "drop-in replacements in the same per-width tables."
        echo ""
        echo "**Two trainings, four conditions.** Base width = train width makes the two arms'"
        echo "w16 inputs the same data (w16 weights hard-linked, metas and split byte-identical,"
        echo "configs differing only in data paths), so each variant is trained once under the"
        echo "muP16 config and that checkpoint is scored under both arms. Every condition carries"
        echo "a \`w16 identity\` line (the two arms' w16 rows must match to the last digit) and a"
        echo "\`reciprocal backward edge feature installed\` line — if the latter ever says NO, the"
        echo "row reproduces the invalid model this retrain exists to replace."
        echo ""
        echo "The plain-GMN and conv matrix-product GMN conditions need no retrain"
        echo "(\`symmetry: permutation\`, where the fix is a no-op), and the conv matrix-product"
        echo "ScaleGMN is forward-only, so it has no bidirectional condition."
        echo "Conditions appear as they finish; \"Finished:\" at the bottom means all four are done."
    } > $SUMMARY
    echo "=== Seeded queue with 2 jobs (= 4 conditions) ==="
fi

if [ -z "${GPUS// /}" ]; then
    echo "=== No GPU list given: queue seeded, no worker started. Arm one with:"
    echo "    bash scripts/add_worker_when_free.sh <gpu> $0 1500 20"
    exit 0
fi

echo "=== predgen bidir ScaleGMN RECIP retrain: starting workers on GPUs [$GPUS] $(date) ==="

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

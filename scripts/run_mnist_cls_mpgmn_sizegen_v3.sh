#!/bin/bash
# Experiment 3 (size-gen v3) for the matrix-product GMN: train w24, test w24-w256.
# Four conditions = {SP, muP24} x {forward, bidirectional}, all dup-equiv
# (the matrix-product MSG is only defined on top of fan-in rescaling + mean aggregation
# + layer-wise mean readout, so there is no baseline variant).
#
# Protocol per condition, mirroring run_mnist_cls_sizegen_v3_final.sh:
#   1. LR sweep over 7 values (6.25e-5 ... 4e-3, x2 grid), 8600 steps each
#   2. full training (200 epochs, patience 50) at the best sweep LR
#
# The four conditions are pushed onto a FIFO queue drained by one worker per GPU
# (see GPUS below); each worker takes the next condition and runs sweep -> full
# training -> summary append, so results land progressively.
# Re-running the script is safe: finished sweep LRs and finished full-training runs
# are detected in the logs and skipped.
#
# Launch:
#   screen -dmS mpgmn_v3 bash -c 'bash scripts/run_mnist_cls_mpgmn_sizegen_v3.sh'
# Results:  /tmp/mpgmn_sizegen_v3/SUMMARY.md   (written incrementally + at the end)
# Logs:     /tmp/mpgmn_sizegen_v3/

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f .env ]; then source .env; fi

PYTHON=${PYTHON:-python}
LRS=(6.25e-5 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3)
MAX_STEPS=8600
LOG_DIR=/tmp/mpgmn_sizegen_v3
SUMMARY=$LOG_DIR/SUMMARY.md
GPUS=(1 2 4)          # one worker per GPU; GPU 5 deliberately left free for others
mkdir -p $LOG_DIR
QUEUE=$LOG_DIR/queue.txt
QUEUE_LOCK=$LOG_DIR/queue.lock

# ---------------------------------------------------------------- helpers

run_sweep() {
    local CONF=$1 DIRECTION=$2 PREFIX=$3 GPU=$4
    for idx in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${idx}.log"
        if grep -q "Best val acc:" "$LOG" 2>/dev/null; then
            echo "[$PREFIX] sweep lr=${LRS[$idx]} already done, skipping"
            continue
        fi
        echo "[$PREFIX] sweep lr=${LRS[$idx]} on GPU $GPU ($(date +%H:%M))"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
            --conf "$CONF" \
            --duplication-equiv True \
            --direction "$DIRECTION" \
            --wandb True --max_steps $MAX_STEPS --lr ${LRS[$idx]} \
            --run_name "lr-sweep-v3-${PREFIX}-${LRS[$idx]}" \
            > "$LOG" 2>&1
    done
}

get_best_lr() {
    local PREFIX=$1
    local BEST_ACC=0
    local BEST_LR="2.5e-4"
    for i in 0 1 2 3 4 5 6; do
        local LOG="${LOG_DIR}/sweep_${PREFIX}_${i}.log"
        [ -f "$LOG" ] || continue
        local ACC=$(grep "Best val acc:" "$LOG" | tail -1 | awk '{print $NF}')
        [ -z "$ACC" ] && continue
        local BETTER=$(python3 -c "print(1 if $ACC > $BEST_ACC else 0)")
        if [ "$BETTER" = "1" ]; then BEST_ACC=$ACC; BEST_LR=${LRS[$i]}; fi
    done
    echo "$BEST_LR"
}

sweep_accs() {
    # compact inline list of the 7 sweep val accs
    local PREFIX=$1
    local OUT=""
    for i in 0 1 2 3 4 5 6; do
        local ACC=$(grep "Best val acc:" "${LOG_DIR}/sweep_${PREFIX}_${i}.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        [ -z "$ACC" ] && ACC="--"
        OUT="${OUT}${ACC} / "
    done
    echo "${OUT%" / "}"
}

run_full() {
    local CONF=$1 DIRECTION=$2 PREFIX=$3 GPU=$4 LR=$5
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    if grep -q "Test acc/loss @ best val:" "$LOG" 2>/dev/null; then
        echo "[$PREFIX] full training already done, skipping"
        return
    fi
    echo "[$PREFIX] full training lr=$LR on GPU $GPU ($(date +%H:%M))"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train_sizegen.py \
        --conf "$CONF" \
        --duplication-equiv True \
        --direction "$DIRECTION" \
        --wandb True --lr "$LR" \
        --run_name "sizegen-v3-mpgmn-${PREFIX}" \
        > "$LOG" 2>&1
}

append_summary() {
    # $1 = human label, $2 = prefix, $3 = best lr
    local LABEL=$1 PREFIX=$2 LR=$3
    local LOG="${LOG_DIR}/full_${PREFIX}.log"
    local RUN_ID=$(grep -o "sizegen-v3-mpgmn-${PREFIX}_[a-z0-9]*" "$LOG" 2>/dev/null | tail -1 | sed "s/.*_//")
    local BEST_VAL=$(grep "Best val acc:" "$LOG" 2>/dev/null | tail -1 | awk '{print $NF}')
    local EPOCHS=$(grep -c "val_acc=" "$LOG" 2>/dev/null)
    {
        echo ""
        echo "### $LABEL"
        echo ""
        echo "- best LR: **$LR** (sweep val accs @ 6.25e-5 / 1.25e-4 / 2.5e-4 / 5e-4 / 1e-3 / 2e-3 / 4e-3: $(sweep_accs "$PREFIX"))"
        echo "- wandb run: \`${RUN_ID:-unknown}\`  (https://wandb.ai/yuxinma/mnist_cls/runs/${RUN_ID})"
        echo "- best val acc: ${BEST_VAL:-n/a}, epochs run: ${EPOCHS:-n/a}"
        echo "- log: \`${LOG}\`"
        echo ""
        echo "| Width | Test acc |"
        echo "|---|---|"
        sed -n '/Test acc\/loss @ best val:/,$p' "$LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+" \
            | sed -E 's/^ +w= *([0-9]+) \((IN|OUT)\): +acc=([0-9.]+).*/| w\1 (\2) | \3 |/'
        echo ""
        echo "OOD mean: $(sed -n '/Test acc\/loss @ best val:/,$p' "$LOG" 2>/dev/null \
            | grep -E "^ +w= *[0-9]+ \(OUT\)" | awk -F'acc=' '{split($2,a," "); s+=a[1]; n+=1} END {if(n) printf "%.4f", s/n; else print "n/a"}')"
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
        IFS='|' read -r CONF DIRECTION PREFIX LABEL <<< "$SPEC"
        echo "[GPU $GPU] picked up $PREFIX ($(date +%H:%M))"
        run_sweep "$CONF" "$DIRECTION" "$PREFIX" "$GPU"
        local LR=$(get_best_lr "$PREFIX")
        echo "[$PREFIX] best LR = $LR"
        run_full "$CONF" "$DIRECTION" "$PREFIX" "$GPU" "$LR"
        append_summary "$LABEL" "$PREFIX" "$LR"
        echo "[$PREFIX] done ($(date))"
    done
    echo "[GPU $GPU] queue empty, worker exiting ($(date +%H:%M))"
}

SP_CONF=configs/mnist_cls/mpgmn_sizegen_sp_v3.yml
MUP_CONF=configs/mnist_cls/mpgmn_sizegen_mup24_v3.yml

touch "$QUEUE_LOCK"
cat > "$QUEUE" <<EOF
$SP_CONF|forward|sp-fw|SP, forward
$MUP_CONF|forward|mup24-fw|muP24, forward
$SP_CONF|bidirectional|sp-bidir|SP, bidirectional
$MUP_CONF|bidirectional|mup24-bidir|muP24, bidirectional
EOF

{
    echo "# Matrix-product GMN — size generalization v3 (train w24, test w24–w256)"
    echo ""
    echo "Started: $(date). All conditions dup-equiv (fan-in rescale + mean aggr + layer-wise mean readout)."
    echo "Configs: \`configs/mnist_cls/mpgmn_sizegen_{sp,mup24}_v3.yml\`, \`--direction {forward,bidirectional}\`."
    echo "Conditions appear below as they finish; \"Finished:\" at the bottom means all four are done."
} > $SUMMARY

echo "=== Matrix-product GMN size-gen v3 started $(date) ==="

PIDS=()
for GPU in "${GPUS[@]}"; do
    worker "$GPU" &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

echo "" >> $SUMMARY
echo "Finished: $(date)" >> $SUMMARY

echo "=== All done $(date). Summary: $SUMMARY ==="

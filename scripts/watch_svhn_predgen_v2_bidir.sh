#!/bin/bash
# Finish the SVHN Stage 3f w512 extension unattended: wait for the six bidirectional
# re-scores of run_svhn_predgen_sizegen_v2_bidir.sh, audit them, and leave every number
# needed for the results doc in one place.
#
# Do NOT gate on the summary's "Finished:" line. That line is written by whichever worker
# set drains the queue, so when the runner is invoked a second time to add a worker (which
# is the documented way to grow it), an earlier invocation can stamp "Finished:" while a
# later worker is still scoring. Six `### ` sections is the real completion test.
#
# On completion it writes, into /tmp/predgen_svhn_sizegen_v2_bidir/:
#   RENDERED.md   doc-shaped tables (per-width tau, tau/tau_b^max, R2/L1/R2_recal) via
#                 render_predgen_v2_results.py --summary <this queue>
#   HANDOFF.md    audit verdicts + what is left to paste into the results doc
# and refreshes /tmp/diag4_svhn{,_compact}.md, whose SVHN bidirectional rows only reach
# w512 once this queue's preds_*.npz exist.
#
# Figures are deliberately NOT regenerated: plot_sizegen_results.py parses the results doc,
# so it has to run after the doc tables are extended, not after the numbers land.
#
# Launch:
#   screen -dmS svhn_bidir_watch bash -c 'bash scripts/watch_svhn_predgen_v2_bidir.sh'
# Log: /tmp/predgen_svhn_sizegen_v2_bidir/watch.log

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -z "${ANYDIM_DATA_ROOT:-}" ] && source .env

PYTHON=${PYTHON:-python}
DIR=/tmp/predgen_svhn_sizegen_v2_bidir
SUMMARY=$DIR/SUMMARY.md
RENDER=$DIR/RENDERED.md
NOTE=$DIR/HANDOFF.md
DOC=results/EXPERIMENTS_SVHN_CNN_accuracy_prediction.md
WANT=6
INTERVAL=${1:-60}
MAX_HOURS=${2:-6}

exec > >(tee -a "$DIR/watch.log") 2>&1
echo "=== SVHN v2 bidir watcher started $(date), polling every ${INTERVAL}s for ${WANT} conditions ==="
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

while true; do
    N=$(grep -c "^### " "$SUMMARY" 2>/dev/null || echo 0)
    if [ "$N" -ge "$WANT" ]; then
        echo "[$(date +%H:%M)] all ${WANT} conditions present"
        break
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[$(date +%H:%M)] giving up after ${MAX_HOURS}h with only ${N}/${WANT} conditions."
        echo "  Check $DIR/eval_*.log and whether any worker is still alive (screen -ls)."
        exit 1
    fi
    echo "[$(date +%H:%M)] ${N}/${WANT} conditions, waiting"
    sleep "$INTERVAL"
done

# --- audit -------------------------------------------------------------------------------
# Anchor every pattern: the summary's own header prose contains these phrases too.
AGREE=$(grep -c "^- v1 agreement: OK" "$SUMMARY")
IDENT=$(grep -c "^- w16 identity vs the other arm: \*\*OK\*\*" "$SUMMARY")
DUMPS=$(ls "$DIR"/preds_*.npz 2>/dev/null | wc -l)
VERDICT=PASS
[ "$AGREE" -eq "$WANT" ] || VERDICT=FAIL
[ "$IDENT" -eq "$WANT" ] || VERDICT=FAIL
[ "$DUMPS" -eq "$WANT" ] || VERDICT=FAIL
echo "[$(date +%H:%M)] audit ${VERDICT}: v1 agreement ${AGREE}/${WANT}, w16 identity ${IDENT}/${WANT}, dumps ${DUMPS}/${WANT}"

# --- artifacts ---------------------------------------------------------------------------
echo "[$(date +%H:%M)] rendering doc-shaped tables -> $RENDER"
$PYTHON scripts/render_predgen_v2_results.py --dataset svhn --summary "$SUMMARY" --stdout > "$RENDER"

echo "[$(date +%H:%M)] recomputing diagnostic 4 (now that this queue's dumps reach w512)"
$PYTHON scripts/summarize_predgen_hp_strata.py --dataset svhn --compact --out /tmp/diag4_svhn_compact.md
$PYTHON scripts/summarize_predgen_hp_strata.py --dataset svhn --out /tmp/diag4_svhn.md

{
    echo "# SVHN Stage 3f — w512 extension + diagnostic 4 — handoff ($(date))"
    echo ""
    echo "## Audit: **${VERDICT}**"
    echo ""
    echo "| Check | Got | Want |"
    echo "|---|---|---|"
    echo "| conditions scored | $(grep -c '^### ' "$SUMMARY") | ${WANT} |"
    echo "| \`v1 agreement: OK\` | ${AGREE} | ${WANT} |"
    echo "| \`w16 identity: OK\` | ${IDENT} | ${WANT} |"
    echo "| \`preds_*.npz\` dumps | ${DUMPS} | ${WANT} |"
    echo ""
    echo "\`v1 agreement\` is the audit that the checkpoint, data and splits are the ones the"
    echo "w16-w128 run measured; \`w16 identity\` that the two arms agree at the train width,"
    echo "where their inputs are byte-identical. Nothing was retrained."
    echo ""
    echo "## OOD means"
    echo ""
    echo '```'
    grep -E "^### |^- mean OOD tau|^- v2 eval run" "$SUMMARY"
    echo '```'
    echo ""
    echo "## What is already written"
    echo ""
    echo "- CIFAR-10 diagnostic 4: recorded in"
    echo "  \`results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md\`"
    echo "  § \"\$\\tau\$ within Hyperparameter Strata\", with item 4 of its mandatory-diagnostics"
    echo "  list and status row 3c flipped to done."
    echo ""
    echo "## What is left, in \`$DOC\`"
    echo ""
    echo "1. Extend the three in-scope \`#### Bidirectional\` subsections from w16-w128 to"
    echo "   w16-w512 (plain GMN's four rows, conv matrix-product GMN's two). Tables are in"
    echo "   \`$RENDER\`, already carrying the per-width \$\\tau/\\tau_b^{\\max}\$ row."
    echo "2. Retire the **\"These are w16-w128 only\"** sentence in the plain GMN"
    echo "   \`#### Bidirectional\` subsection, and the matching \"**w16-w128 only**, not"
    echo "   re-scored to w512\" in status row 3f."
    echo "3. Add the diagnostic-4 section (tables: \`/tmp/diag4_svhn_compact.md\`, per-width"
    echo "   detail in \`/tmp/diag4_svhn.md\`) and flip item 4 of the mandatory-diagnostics"
    echo "   list plus status row 3c, mirroring what CIFAR-10 now has."
    echo "4. Re-run \`$PYTHON scripts/plot_sizegen_results.py\` **after** step 1 — it parses the"
    echo "   doc, so the four bidirectional SVHN panels keep their short w128 x-axis until the"
    echo "   tables above are extended. The doc's § Figures says they stop at w128; update it."
    echo ""
    echo "wandb: entity \`yuxinma\`, project \`svhn_predgen\`. The six v2 eval run IDs are in the"
    echo "OOD-means block above; the three training runs (\`dcbl1fhp\`, \`sqno0nqf\`, \`bfeiloq3\`)"
    echo "are unchanged from the w16-w128 tables."
} > "$NOTE"

echo "[$(date +%H:%M)] handoff written to $NOTE"
[ "$VERDICT" = PASS ] || exit 1

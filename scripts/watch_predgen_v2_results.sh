#!/bin/bash
# Keep the CNN Stage 3 v2 results table in the results doc current while the ten re-scores
# of run_cifar10_predgen_sizegen_v2.sh drain, then stop.
#
# Every 5 min: re-render the marked block via scripts/render_predgen_v2_results.py. When
# /tmp/predgen_sizegen_v2/SUMMARY.md gains its "Finished:" line (all ten popped), do one last
# render -- which also retires the "running" phrasings -- write a handoff note, and exit.
#
# It only ever rewrites the region between the render script's markers, so it cannot clobber
# hand-written prose. Safe to run unattended and safe to run twice (the renderer is idempotent
# and prints "no change").
#
# Launch:
#   screen -dmS predgen_v2_render bash -c 'bash scripts/watch_predgen_v2_results.sh'
# Log: /tmp/predgen_sizegen_v2/render_watch.log

set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON=${PYTHON:-python}
DIR=/tmp/predgen_sizegen_v2
SUMMARY=$DIR/SUMMARY.md
NOTE=$DIR/HANDOFF.md
INTERVAL=${1:-300}
MAX_HOURS=${2:-12}       # give up rather than loop forever if the queue wedges

echo "=== v2 results watcher started $(date), polling every ${INTERVAL}s ==="
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))

while true; do
    $PYTHON scripts/render_predgen_v2_results.py 2>&1 | sed "s/^/[$(date +%H:%M)] /"

    if grep -q "^Finished:" "$SUMMARY" 2>/dev/null; then
        echo "[$(date +%H:%M)] queue drained -- final render done"
        N=$(grep -c "^### " "$SUMMARY")
        {
            echo "# CNN Stage 3 v2 re-score — handoff ($(date))"
            echo ""
            echo "All ten conditions finished; **${N} sections** in \`$SUMMARY\`."
            echo ""
            echo "The numbers are already in"
            echo "\`results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md\` § Results — All Ten"
            echo "Conditions, inside the \`predgen-v2-results\` markers, written by"
            echo "\`scripts/render_predgen_v2_results.py\`. What is NOT written:"
            echo ""
            echo "1. The prose analysis under that table (ranking, what moved between 8x and 32x,"
            echo "   how the ten compare to the conv matrix-product ScaleGMN in row 11)."
            echo "2. The corresponding experiment-log entry."
            echo "3. Every \`v1 agreement\` cell must read \`exact\`. Check that first:"
            echo "   \`grep -c '^- v1 agreement: OK' $SUMMARY\` should print 10 (anchor the"
            echo "   pattern -- the file's own header prose contains the phrase too)."
            echo "   Currently: $(grep -c '^- v1 agreement: OK' "$SUMMARY")."
            echo ""
            echo "Reminder when writing it up: SP columns above w128 are confounded (only 0.654 of"
            echo "w384 SP labels sit inside w16's label support), so SP rows measure a shifted"
            echo "target as much as a metanetwork. muP16 is the interpretable arm."
            echo ""
            echo "\`\`\`"
            grep -E "^### |mean OOD tau|v1 agreement" "$SUMMARY"
            echo "\`\`\`"
        } > "$NOTE"
        echo "[$(date +%H:%M)] handoff note written to $NOTE"
        exit 0
    fi

    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "[$(date +%H:%M)] giving up after ${MAX_HOURS}h without a Finished: line."
        echo "  The queue may be wedged -- check $DIR/worker_gpu*.log."
        exit 1
    fi
    sleep "$INTERVAL"
done

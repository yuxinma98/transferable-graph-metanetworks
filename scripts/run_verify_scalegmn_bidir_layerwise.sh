#!/bin/bash
# Re-runs the matrix-product verification checkers (scripts/check_matrix_product_gmn.py,
# scripts/check_matrix_product_gmn_cnn.py) to populate the (a) GMN / (b) ScaleGMN
# equivalence-class subtables in results/VERIFICATION_gmn_properties.md:
#   - MLP: adds the bidirectional ScaleGMN contrast (--include-bidir-scalegmn), which was
#     previously never built even though MatrixProductScale_GNN_layer_aggr exists
#     specifically for this. MLP has no readout choice (always layerwise_mean).
#   - CNN: switches Test 2's scoring readout from last_layer to layerwise_mean
#     (--readout layerwise_mean) and adds the same bidirectional ScaleGMN contrast.
# Untrained models, tiny CNNs/INRs -- each run finishes in well under a minute; logged to
# files rather than piped through a screen session since nothing here is long-running.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
PYTHON=${PYTHON:-python}
LOGDIR=/tmp/verify_scalegmn_bidir_layerwise
mkdir -p "$LOGDIR"

echo "[1/4] MLP, family=gmn (subtable a, no change in scope -- re-run for the record)"
$PYTHON scripts/check_matrix_product_gmn.py --family gmn --n-inrs 10 \
    > "$LOGDIR/mlp_gmn.log" 2>&1

echo "[2/4] MLP, family=scalegmn, --include-bidir-scalegmn (subtable b)"
$PYTHON scripts/check_matrix_product_gmn.py --family scalegmn --n-inrs 10 \
    --include-bidir-scalegmn > "$LOGDIR/mlp_scalegmn_bidir.log" 2>&1

echo "[3/4] CNN, family=gmn, --readout layerwise_mean (subtable a)"
$PYTHON scripts/check_matrix_product_gmn_cnn.py --family gmn \
    --readout layerwise_mean > "$LOGDIR/cnn_gmn_layerwise.log" 2>&1

echo "[4/4] CNN, family=scalegmn, --readout layerwise_mean --include-bidir-scalegmn (subtable b)"
$PYTHON scripts/check_matrix_product_gmn_cnn.py --family scalegmn \
    --readout layerwise_mean --include-bidir-scalegmn \
    > "$LOGDIR/cnn_scalegmn_bidir_layerwise.log" 2>&1

echo "Done. Logs in $LOGDIR/"
if grep -L "ALL CHECKS PASSED" "$LOGDIR"/*.log; then
    echo "^ logs above did NOT report ALL CHECKS PASSED"
else
    echo "All four logs report ALL CHECKS PASSED"
fi

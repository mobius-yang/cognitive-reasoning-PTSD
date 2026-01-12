#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-pipeline}
SEEDS=${2:-"42"}
WGRID=${3:-"1,1.5,2,3"}
DIMS=${4:-"temporal,coherence,reflection,language,perspective,sensory,arousal"}

TS=$(date +"%Y%m%d_%H%M%S")
OUTDIR="results/ablation_vam_sam_${TS}"
mkdir -p "$OUTDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_DISABLE_SHM=1
export KMP_USE_SHM=0
export KMP_SHM_DISABLE=1
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export OMP_PROC_BIND=false

python scripts/ablation_vam_sam.py \
  --mode "$MODE" \
  --seeds "$SEEDS" \
  --weight-grid "$WGRID" \
  --weight-dims "$DIMS" \
  --out-json "$OUTDIR/logreg.json" \
  --out-csv "$OUTDIR/logreg.csv" \
  | tee "$OUTDIR/stdout.log"

echo "Done. See $OUTDIR"

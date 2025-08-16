#!/usr/bin/env bash
set -euo pipefail
IFS=$' \n\t'

OUTDIR="/n/home09/annabelma/my_experiments/mnist_final_gridsearch"
EXP_FILE="/n/home09/annabelma/Dataset-Comparison-Underlying-Symmetries/cluster_experiments/final_mnist_exp/data_transform.py"
JOBLIST="$OUTDIR/joblist.txt"

mkdir -p "$OUTDIR"

if [[ ! -f "$EXP_FILE" ]]; then
  echo "ERROR: Runner file not found: $EXP_FILE" >&2
  exit 1
fi

FEATURES=( xs bs )
METHODS=( sinkhorn sinkhorn_stabilized sinkhorn_log greenkhorn )
REGS=( 0.0001 0.001 0.01 0.1 1 )

# fresh joblist in OUTDIR
: > "$JOBLIST"

while read -r D1 D2; do
  [[ -z "${D1:-}" || -z "${D2:-}" ]] && continue
  for F in "${FEATURES[@]}"; do
    for M in "${METHODS[@]}"; do
      for R in "${REGS[@]}"; do
        OUT="${OUTDIR}/${D1}__${D2}__${F}__${M}__${R}.json"
        echo "python -u $EXP_FILE $D1 $D2 $F $M $R $OUT" >> "$JOBLIST"
      done
    done
  done
done <<'EOF'
usps_0 mnist_usps_0
mnist_usps_rotated mnist_usps_0
fashion_rotated fashion_unrotated
fashion_baseline_tests fashion_baseline_labels
kmnist_rotated kmnist_unrotated 
kmnist_baseline_tests kmnist_baseline_labels 
mnist_rotated mnist_unrotated 
mnist_baseline_tests mnist_baseline_labels 
EOF

echo "Wrote $(wc -l < "$JOBLIST") jobs to $JOBLIST"


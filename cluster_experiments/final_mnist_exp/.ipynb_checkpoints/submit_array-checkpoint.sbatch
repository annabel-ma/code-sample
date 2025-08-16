#!/bin/bash
#SBATCH -J ot-bw-grid
#SBATCH -c 1
#SBATCH -t 0-12:00:00
#SBATCH -p shared
#SBATCH --mem=50G
#SBATCH --open-mode=append
#SBATCH -o /n/home09/annabelma/my_experiments/mnist_final_gridsearch/logs/ot_%A_%a.out
#SBATCH -e /n/home09/annabelma/my_experiments/mnist_final_gridsearch/logs/ot_%A_%a.err

set -euo pipefail
IFS=$'\n\t'

# --- Base directory for all outputs/logs/results ---
BASE_DIR="/n/home09/annabelma/my_experiments/mnist_final_gridsearch"

# Make sure logs & results exist
mkdir -p "$BASE_DIR/logs" "$BASE_DIR/results"

# Load environment
module load python/3.10.9-fasrc01
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

JOBLIST="${JOBLIST:-$BASE_DIR/joblist.txt}"
[[ -f "$JOBLIST" ]] || { echo "ERROR: $JOBLIST not found"; exit 1; }

# Append script
APPENDER="/n/home09/annabelma/Dataset-Comparison-Underlying-Symmetries/cluster_experiments/append_ndjson.py"

# Master merged results file & lock
RUNNER="/n/home09/annabelma/Dataset-Comparison-Underlying-Symmetries/cluster_experiments/final_mnist_exp/test_ot_once.py"
MERGED="$BASE_DIR/results/all_results.ndjson"
LOCK="$BASE_DIR/results/all_results.ndjson.lock"

# Figure out which line to run
NLINES=$(wc -l < "$JOBLIST")
OFFSET=${OFFSET:-0}
LIDX=${SLURM_ARRAY_TASK_ID:?need array id}
GIDX=$(( OFFSET + LIDX ))
(( GIDX>=1 && GIDX<=NLINES )) || { echo "Index $GIDX out of range (1..$NLINES)"; exit 0; }

read -r CMD < <(sed -n "${GIDX}p" "$JOBLIST")
echo "[CMD] $CMD"

# Split into tokens (need space in IFS for this read only)
IFS=$' \t\n' read -r -a TOKENS <<< "$CMD"

# Expect: python -u EXP_FILE D1 D2 F M R OUT
if (( ${#TOKENS[@]} < 9 )); then
  echo "ERROR: Parsed fewer than 9 tokens from joblist line: '$CMD'" >&2
  echo "TOKENS_COUNT=${#TOKENS[@]}" >&2
  exit 1
fi

RUN="${TOKENS[0]}"           # python
FLAG="${TOKENS[1]}"          # -u
EXP_FILE="${TOKENS[2]}"
D1="${TOKENS[3]}"
D2="${TOKENS[4]}"
F="${TOKENS[5]}"
M="${TOKENS[6]}"
R="${TOKENS[7]}"
OUT="${TOKENS[8]}"

echo "    Runner:  $EXP_FILE"
echo "    Dataset1:$D1"
echo "    Dataset2:$D2"
echo "    Feature: $F"
echo "    Method:  $M"
echo "    Reg:     $R"
echo "    Output:  $OUT"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

TMP="${OUT}.part.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID:-0}.$$"

if [[ -s "$OUT" ]]; then
  echo "[SKIP] $OUT exists"
else
  set +e
  "$RUN" "$FLAG" "$RUNNER" \
    --exp_file "$EXP_FILE" \
    --data1 "$D1" \
    --data2 "$D2" \
    --feature "$F" \
    --method "$M" \
    --reg "$R" \
    --out "$TMP"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[ERROR] test_ot_once.py failed (rc=$rc)"; exit $rc
  fi
  if [[ -s "$TMP" ]]; then
    mv -f -- "$TMP" "$OUT"
    echo "[WRITE] -> $OUT"
  else
    echo "[WARN] test_ot_once.py produced no data"
  fi
fi

echo "=== [FINISH] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID ($GIDX/$NLINES) ==="
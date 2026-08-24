#!/bin/bash
# Absolute animations of one field for several runs, on a colour scale shared across them.
#
#   FIELD=q_850 RUNS="a b c" sbatch diagnostics/run_field_animations.sh
#
# Two stages: compare_runs.py first reports the exact min/max over ALL the runs (and writes
# the drift curve as a by-product), then the animations are rendered with that fixed range.
# Without the shared range each run is scaled to its own extremes, which re-normalises a real
# signal away and makes the videos look identical.

#SBATCH --job-name=field_anim
#SBATCH --exclusive --mem=450G
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH -A e-ext-2025e01-128
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export MPLBACKEND=Agg

read -r -a _runs <<< "${RUNS:-sst_exp1_1y_0K sst_exp1_1y_2K sst_exp1_1y_4K}"
FIELD="${FIELD:-q_850}"
REF="${REF:-${_runs[0]}}"
OUT_DIR="${OUT_DIR:-results/comparison}"
WORKERS="${WORKERS:-64}"

_cmp_args=(); for r in "${_runs[@]}"; do [ "$r" != "$REF" ] && _cmp_args+=(--run "$r"); done

echo "===== $(date +%H:%M) [1/2] shared range for $FIELD over ${_runs[*]} ====="
RANGE_LOG=$(mktemp)
.venv/bin/python3 diagnostics/compare_runs.py --ref "$REF" "${_cmp_args[@]}" \
  --field "$FIELD" --out-dir "$OUT_DIR" 2>&1 | tee "$RANGE_LOG"

# "SHARED COLOUR RANGE over [...]: --vmin 1.2e-05 --vmax 0.0203"
VMIN=$(grep -m1 "SHARED COLOUR RANGE" "$RANGE_LOG" | sed -n 's/.*--vmin \([^ ]*\).*/\1/p')
VMAX=$(grep -m1 "SHARED COLOUR RANGE" "$RANGE_LOG" | sed -n 's/.*--vmax \([^ ]*\).*/\1/p')
rm -f "$RANGE_LOG"
if [ -z "$VMIN" ] || [ -z "$VMAX" ]; then
  echo "FAILED: could not determine a shared colour range; refusing to render per-run scales"
  exit 1
fi
echo "===== $(date +%H:%M) [2/2] animating $FIELD on [$VMIN, $VMAX] ====="

FIELDS="$FIELD" RUNS="${_runs[*]}" VMIN="$VMIN" VMAX="$VMAX" WORKERS="$WORKERS" \
  bash diagnostics/run_animations.sh
echo "===== $(date +%H:%M) all done ====="

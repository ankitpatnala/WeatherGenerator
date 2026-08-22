#!/bin/bash
# Cross-run comparison for a set of rollouts: drift curve + shared-scale side-by-side video.
#
#   RUNS="a b c" REF=a sbatch diagnostics/run_comparison.sh
#
# Each stage is gated behind an 8-frame smoke run, so a bug shows up in ~2 min instead of
# after an hour of loading. Reading the 58GB validation zips is I/O heavy -- run it here, on a
# compute node, never on the login node.

#SBATCH --job-name=run_comparison
#SBATCH --exclusive --mem=450G
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH -A hclimrep
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export MPLBACKEND=Agg

read -r -a _runs <<< "${RUNS:-sst_exp1_1y_0K sst_exp1_1y_2K sst_exp1_1y_4K}"
REF="${REF:-${_runs[0]}}"
FIELD="${FIELD:-2t}"
WORKERS="${WORKERS:-64}"
OUT_DIR="${OUT_DIR:-results/comparison}"

# --run args for compare_runs.py = every run except the baseline
_cmp_args=()
for r in "${_runs[@]}"; do
  [ "$r" != "$REF" ] && _cmp_args+=(--run "$r")
done
# --run args for the side-by-side = all runs, panels left-to-right
_sbs_args=()
for r in "${_runs[@]}"; do _sbs_args+=(--run "$r"); done

echo "===== $(date +%H:%M) runs=${_runs[*]} ref=$REF field=$FIELD ====="

echo "===== $(date +%H:%M) [1/2] drift curve (smoke) ====="
.venv/bin/python3 diagnostics/compare_runs.py --ref "$REF" "${_cmp_args[@]}" \
  --field "$FIELD" --max-frames 8 --out-dir "$OUT_DIR/_smoke" || { echo "SMOKE FAILED: compare_runs"; exit 1; }

echo "===== $(date +%H:%M) [1/2] drift curve (full) ====="
.venv/bin/python3 diagnostics/compare_runs.py --ref "$REF" "${_cmp_args[@]}" \
  --field "$FIELD" --out-dir "$OUT_DIR" || echo "FAILED: compare_runs"

echo "===== $(date +%H:%M) [2/2] side-by-side (smoke) ====="
.venv/bin/python3 diagnostics/animate_side_by_side.py "${_sbs_args[@]}" \
  --field "$FIELD" --workers 8 --max-frames 8 --no-video || { echo "SMOKE FAILED: side_by_side"; exit 1; }

echo "===== $(date +%H:%M) [2/2] side-by-side (full) ====="
.venv/bin/python3 diagnostics/animate_side_by_side.py "${_sbs_args[@]}" \
  --field "$FIELD" --workers "$WORKERS" --fps 8 || echo "FAILED: side_by_side"

echo "===== $(date +%H:%M) all done ====="

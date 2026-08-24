#!/bin/bash
# Global-mean time-series plot (target + each run) for a field. Smoke-gated like
# run_comparison.sh; reading the 58GB zips belongs on a compute node, not the login node.
#
#   RUNS="a b c" FIELD=2t sbatch diagnostics/run_series.sh

#SBATCH --job-name=global_mean_series
#SBATCH --exclusive --mem=450G
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH -A e-ext-2025e01-128
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv/bin/activate
export MPLBACKEND=Agg

read -r -a _runs <<< "${RUNS:-sst_exp1_1y_0K sst_exp1_1y_2K sst_exp1_1y_4K}"
FIELD="${FIELD:-2t}"
_args=(); for r in "${_runs[@]}"; do _args+=(--run "$r"); done
# LABELS="run=Display;run2=Display2" renames series in the figures (CSV keeps the run ids)
if [ -n "${LABELS:-}" ]; then
  IFS=';' read -r -a _labels <<< "$LABELS"
  for l in "${_labels[@]}"; do _args+=(--label "$l"); done
fi

echo "===== $(date +%H:%M) series: runs=${_runs[*]} field=$FIELD ====="
OUT_DIR="${OUT_DIR:-results/comparison}"
# smoke output lives under OUT_DIR, not a fixed path -- two jobs running concurrently for
# different experiments would otherwise overwrite each other's smoke files
.venv/bin/python3 diagnostics/plot_global_mean_series.py "${_args[@]}" --field "$FIELD" \
  --max-frames 8 --out-dir "$OUT_DIR/_smoke" || { echo "SMOKE FAILED"; exit 1; }

echo "===== $(date +%H:%M) full ====="
.venv/bin/python3 diagnostics/plot_global_mean_series.py "${_args[@]}" --field "$FIELD" \
  --out-dir "$OUT_DIR" || echo "FAILED: series"
echo "===== $(date +%H:%M) all done ====="

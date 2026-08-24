#!/bin/bash
# Generate eval-style decadal animations for several fields, over one or more runs (sequential).
#
# Run it, don't source it:
#     bash diagnostics/run_animations.sh                        # default runs x default fields
#     bash diagnostics/run_animations.sh msl sst                # explicit fields (highest precedence)
#     FIELDS="msl sst" bash diagnostics/run_animations.sh
#     RUNS="sst_exp1_1y_0K sst_exp1_1y_2K" bash diagnostics/run_animations.sh 2t
#     WORKERS=16 bash diagnostics/run_animations.sh             # be gentler on a shared node
#
# Set REF=<run> to animate run-vs-run differences instead of the raw field: each run in RUNS
# is rendered as <run> MINUS <REF>, e.g. the 2K/4K rollouts minus the 0K control, isolating
# the response to the prescribed SST warming. (Not the same as prediction-minus-target, which
# is animate_decadal.py's --source bias.)
#     REF=sst_exp1_1y_0K RUNS="sst_exp1_1y_2K sst_exp1_1y_4K" bash diagnostics/run_animations.sh
#
# Each run name is resolved to results/<run>/validation_chkpt00000_rank0000.zip; the mp4 lands
# next to it as results/<run>/decadal_<field>_animation.mp4.
#
# Sourcing leaves FIELDS set in your interactive shell, so the *next* run silently reuses the
# previous field list instead of the default. Hence the positional-args form above, and the
# local `_fields` below (this script never assigns to FIELDS itself).

cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
source .venv/bin/activate
export MPLBACKEND=Agg

# precedence: command-line args > FIELDS env var > default
if [ "$#" -gt 0 ]; then
  _fields=("$@")
else
  read -r -a _fields <<< "${FIELDS:-10u 10v 2t}"
fi
read -r -a _runs <<< "${RUNS:-sst_exp1_1y_0K sst_exp1_1y_2K sst_exp1_1y_4K}"
_workers="${WORKERS:-24}"
_ref="${REF:-}"

# Fixed colour range shared across runs. Without it every run is scaled to its own min/max,
# which re-normalises a real warming away and makes the videos look identical.
# diagnostics/compare_runs.py prints the range to use over a set of runs.
# SOURCE=prediction (default) | target | bias  -- which field animate_decadal.py renders
_source_args=()
[ -n "${SOURCE:-}" ] && _source_args+=(--source "$SOURCE")

_range_args=()
[ -n "${VMIN:-}" ] && _range_args+=(--vmin "$VMIN")
[ -n "${VMAX:-}" ] && _range_args+=(--vmax "$VMAX")

_ref_args=()
if [ -n "$_ref" ]; then
  _ref_args=(--ref-zip "results/$_ref/validation_chkpt00000_rank0000.zip")
  if [ ! -f "results/$_ref/validation_chkpt00000_rank0000.zip" ]; then
    echo "MISSING reference: results/$_ref/validation_chkpt00000_rank0000.zip" >&2
    exit 1
  fi
fi

echo "===== runs: ${_runs[*]} | fields: ${_fields[*]} | workers: $_workers${_ref:+ | minus $_ref}${VMIN:+ | range ${VMIN}..${VMAX}}${SOURCE:+ | source $SOURCE} ====="

# fail fast on a missing/misspelled run rather than after the first hour of rendering
for r in "${_runs[@]}"; do
  if [ ! -f "results/$r/validation_chkpt00000_rank0000.zip" ]; then
    echo "MISSING: results/$r/validation_chkpt00000_rank0000.zip" >&2
    exit 1
  fi
done

for r in "${_runs[@]}"; do
  for f in "${_fields[@]}"; do
    echo "===== $(date +%H:%M) generating $r / $f ====="
    .venv/bin/python3 diagnostics/animate_decadal.py \
      --zip "results/$r/validation_chkpt00000_rank0000.zip" \
      "${_ref_args[@]}" "${_range_args[@]}" "${_source_args[@]}" \
      --field "$f" --fps 8 --workers "$_workers" || echo "FAILED: $r / $f"
  done
done
echo "===== $(date +%H:%M) all done ====="

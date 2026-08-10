#!/bin/bash
# Generate eval-style decadal animations for several fields, with parallel frame rendering.
#   sbatch weathergen_animate.sh                          # default 5 fields
#   FIELDS="2t msl" sbatch weathergen_animate.sh          # custom fields

#SBATCH --job-name=decadal_anim
#SBATCH --exclusive --mem=450G
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -A ch17
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
source .venv/bin/activate
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FIELDS="${FIELDS:-t_850 z_500 z_850 u_850 v_850}"
WORKERS="${WORKERS:-64}"

for f in $FIELDS; do
  echo "===== $(date +%H:%M:%S) $f (workers=$WORKERS) ====="
  srun .venv/bin/python3 diagnostics/animate_decadal.py --field "$f" --fps 8 --workers "$WORKERS" --max-frames "${MAXFRAMES:-0}" \
    || echo "FAILED: $f"
done
echo "===== $(date +%H:%M:%S) all done ====="

#!/bin/bash
# Generate eval-style animations on a compute node, with parallel frame rendering.
#
#   FIELDS=2t sbatch weathergen_animate.sh                      # raw field, default runs
#   FIELDS=2t REF=sst_exp1_1y_0K RUNS="sst_exp1_1y_2K sst_exp1_1y_4K" \
#       sbatch weathergen_animate.sh                            # run-vs-run difference
#
# All knobs (FIELDS / RUNS / REF / WORKERS) are read by diagnostics/run_animations.sh, which
# this script just wraps with a compute-node allocation -- the loop lives in one place only.
# sbatch exports the submitting environment by default, so the prefix assignments above reach
# the job.

#SBATCH --job-name=decadal_anim
#SBATCH --exclusive --mem=450G
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH -A e-ext-2025e01-128
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export MPLBACKEND=Agg

# frame rendering is pure CPU (matplotlib/cartopy); no GPU is requested or used
export WORKERS="${WORKERS:-64}"

bash diagnostics/run_animations.sh

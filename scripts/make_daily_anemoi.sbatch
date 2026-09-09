#!/bin/bash
#SBATCH --job-name=daily_anemoi
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -A e-ext-2025e01-128
#SBATCH --array=0-23
#SBATCH --output=logs/daily-anemoi-%A_%a.out
#SBATCH --error=logs/daily-anemoi-%A_%a.err

# Bake the [-24, 0, "frequency"] rolling average + 24h subsampling of the hourly ERA5 zarr into
# a standalone daily anemoi dataset, so training reads one chunk per sample instead of 25.
#
# Run `init` once on the login node first (it only writes metadata, takes seconds):
#
#   .venv/bin/python scripts/make_daily_anemoi.py init --src "$SRC" --dst "$DST"
#   sbatch scripts/make_daily_anemoi.sbatch "$DST"
#   .venv/bin/python scripts/make_daily_anemoi.py status --dst "$DST"   # after the array ends
#   .venv/bin/python scripts/make_daily_anemoi.py verify --dst "$DST"
#
# Each array task owns a contiguous, chunk-aligned slice of the output, so the tasks never write
# the same chunk and can be re-run individually if one fails.

set -euo pipefail
REPO=/e/project1/weatherai/patnala1/WeatherGen/WeatherGenerator
cd "$REPO"

DST=${1:?"usage: sbatch scripts/make_daily_anemoi.sbatch <output.zarr>"}

source .venv/bin/activate

srun python scripts/make_daily_anemoi.py fill \
  --dst "$DST" \
  --shard "$SLURM_ARRAY_TASK_ID" \
  --nshards "$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))" \
  --log-every 100

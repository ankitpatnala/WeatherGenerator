#!/bin/bash
# Dump FE latents per forecast step, CHUNKED (one .npy per 50 steps to avoid an inode blow-up
# on long rollouts), under results/<ckpt>/latents/set{A,B,C}/.
#
#   sbatch weathergen_latent_dump.sh A    # forecast : free-running rollout,      POST-FE
#   sbatch weathergen_latent_dump.sh B    # encoder  : consecutive 1-step from truth, POST-FE
#   sbatch weathergen_latent_dump.sh C    # encoder  : consecutive 1-step from truth, PRE-FE
#                                         #            (pre-FE = encode(real(t)), assimilation latent)
#
# All self-chain on the 30-min debug partition and are resume-safe (chunk files never clobber).
# Set C is a SEPARATE set so it never overrides Set B. Length via STEPS (default 500). oq9o0t86, 2015.

#SBATCH --job-name=latent_dump
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SET=$1
MAX=${2:-25}
CKPT=${CKPT:-oq9o0t86}
START=201501010000
END=201507050000
NUM_STEPS=${STEPS:-500}
CHUNK=50
export LATENT_FLUSH_EVERY=$CHUNK      # one .npy per CHUNK steps (aligned to rollout chunk_size)
DIAG="$REPO/diagnostics/dump_latents.py"
SCRIPT="$REPO/weathergen_latent_dump.sh"

COMMON=(
  test_config.start_date=$START
  test_config.end_date=$END
  test_config.compute_loss=false
  test_config.output.num_samples=0
)

if [ "$SET" = "A" ]; then
  # forecast / free-running: single IC, one resumable long rollout, chained across debug links.
  # Latents live under the model that produced them; the small rollout_state uses a separate
  # run-id so it can't clobber the model's own results dir.
  RUN_ID=lat_${CKPT}_setA
  export LATENT_CAPTURE=post
  export LATENT_DUMP_DIR="$REPO/results/$CKPT/latents/setA"
  export LATENT_INDEX_MODE=rollout           # chunk file index = global forecast step
  mkdir -p "$LATENT_DUMP_DIR"
  STATE="$REPO/results/$RUN_ID/rollout_state.pt"

  DONE=$(.venv/bin/python3 -c "import torch,glob;f=glob.glob('$STATE');print(torch.load(f[0],map_location='cpu',weights_only=False)['done_steps'] if f else 0)" 2>/dev/null || echo 0)
  echo "[chain] setA done=$DONE / $NUM_STEPS  links_left=$MAX  job=${SLURM_JOB_ID:-local}"
  if [ "$DONE" -ge "$NUM_STEPS" ]; then echo "[chain] setA complete."; exit 0; fi
  if [ "$MAX" -le 0 ]; then echo "[chain] max links reached at $DONE/$NUM_STEPS."; exit 0; fi

  sbatch --export=ALL,CKPT="$CKPT" --dependency=afterany:"${SLURM_JOB_ID:-0}" "$SCRIPT" A $((MAX - 1)) \
    || echo "[chain] WARNING: failed to submit successor"

  srun uv --offline run python "$DIAG" \
    --from-run-id "$CKPT" --run-id "$RUN_ID" \
    --options "${COMMON[@]}" \
      test_config.samples_per_mini_epoch=1 \
      test_config.forecast.num_steps=$NUM_STEPS \
      test_config.forecast.chunk_size=$CHUNK \
      test_config.forecast.free_running=true \
      test_config.forecast.resume=true

elif [ "$SET" = "B" ] || [ "$SET" = "C" ]; then
  # encoder / truth-anchored: consecutive ICs, each a single 1-step forecast.
  #   B = POST-FE latent  FE(encode(real(t)))
  #   C = PRE-FE  latent  encode(real(t))         (the FE input; a SEPARATE set, never touches B)
  # Chain by DATE OFFSET: resume from the last COMPLETE chunk boundary (any partial buffer lost
  # on a kill is simply recomputed), shifting start_date and LATENT_START_INDEX.
  RUN_ID=lat_${CKPT}_set$SET
  [ "$SET" = "C" ] && export LATENT_CAPTURE=pre || export LATENT_CAPTURE=post
  export LATENT_DUMP_DIR="$REPO/results/$CKPT/latents/set$SET"
  export LATENT_INDEX_MODE=counter
  mkdir -p "$LATENT_DUMP_DIR"

  NCHUNKS=$(ls "$LATENT_DUMP_DIR"/z_*.npy 2>/dev/null | wc -l)
  RESUME=$((NCHUNKS * CHUNK))
  echo "[chain] set$SET (capture=$LATENT_CAPTURE) chunks=$NCHUNKS -> resume at $RESUME / $NUM_STEPS  links_left=$MAX  job=${SLURM_JOB_ID:-local}"
  if [ "$RESUME" -ge "$NUM_STEPS" ]; then echo "[chain] set$SET complete."; exit 0; fi
  if [ "$MAX" -le 0 ]; then echo "[chain] max links reached at $RESUME/$NUM_STEPS."; exit 0; fi

  export LATENT_START_INDEX=$RESUME
  RSTART=$(.venv/bin/python3 -c "import datetime;print((datetime.datetime(2015,1,1)+datetime.timedelta(hours=6*$RESUME)).strftime('%Y%m%d%H%M'))")
  REMAIN=$((NUM_STEPS - RESUME))
  echo "[chain] set$SET resume start_date=$RSTART remaining=$REMAIN"

  sbatch --export=ALL,CKPT="$CKPT" --dependency=afterany:"${SLURM_JOB_ID:-0}" "$SCRIPT" "$SET" $((MAX - 1)) \
    || echo "[chain] WARNING: failed to submit successor"

  srun uv --offline run python "$DIAG" \
    --from-run-id "$CKPT" --run-id "$RUN_ID" \
    --options \
      test_config.start_date=$RSTART test_config.end_date=$END \
      test_config.compute_loss=false test_config.output.num_samples=0 \
      test_config.samples_per_mini_epoch=$REMAIN \
      test_config.shuffle=false test_config.forecast.num_steps=1
else
  echo "usage: sbatch weathergen_latent_dump.sh [A|B|C] [MAX_LINKS]"; exit 1
fi

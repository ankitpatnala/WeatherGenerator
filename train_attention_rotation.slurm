#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --job-name=attn_rot_train
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=develbooster
#SBATCH --chdir=.
#SBATCH --output=logs/attn_rot-%x.%j.out
#SBATCH --error=logs/attn_rot-%x.%j.err

mkdir -p logs

module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3
if [ "${WEATHERGEN_NSYS_PROFILING:-0}" = "1" ]; then
    ml Nsight-Systems/2024.7.1
fi


# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── GPU visibility ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── NCCL tuning for multi-GPU ─────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# ── Activate environment ──────────────────────────────────────────────────────
source .venv/bin/activate

# ── Launch with torchrun (one process per GPU) ────────────────────────────────

module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3
if [ "${WEATHERGEN_NSYS_PROFILING:-0}" = "1" ]; then
    ml Nsight-Systems/2024.7.1
fi


# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


srun uv --offline run python train_attention_rotation.py

#!/bin/bash -x
#SBATCH --account=ab1412
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --output=logs/atmorep-%x.%j.out
#SBATCH --error=logs/atmorep-%x.%j.err

# import modules at dkrz
module load python3/2023.01-gcc-11.2.0 nvhpc/23.9-gcc-11.2.0
source pyenv/bin/activate

export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=TRACE
echo "nccl_debug: $NCCL_DEBUG"

# work-around for flipping links issue on JUWELS-BOOSTER
export NCCL_IB_TIMEOUT=250
export UCX_RC_TIMEOUT=16s
export NCCL_IB_RETRY_CNT=50

echo "Starting job."
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
date

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

CONFIG_DIR=${SLURM_SUBMIT_DIR}/atmorep_models_slurmid/atmorep_train_${SLURM_JOBID}
mkdir ${CONFIG_DIR}
cp ${SLURM_SUBMIT_DIR}/atmorep/core/train.py ${CONFIG_DIR}
echo "${CONFIG_DIR}/train.py"
srun --label --cpu-bind=v --accel-bind=v ${SLURM_SUBMIT_DIR}/pyenv/bin/python -u ${CONFIG_DIR}/train.py > output/output_${SLURM_JOBID}.txt

echo "Finished job."
sstat -j $SLURM_JOB_ID.batch   --format=JobID,MaxVMSize
date
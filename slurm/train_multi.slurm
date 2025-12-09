#!/bin/bash
#SBATCH --job-name=llm-train-multi
#SBATCH --account=YOUR_ACCOUNT          # Replace with your account
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --nodes=4                       # 4 nodes
#SBATCH --ntasks-per-node=4             # 4 tasks per node (one per GPU)
#SBATCH --cpus-per-task=8               # 8 CPUs per GPU
#SBATCH --gres=gpu:4                    # 4 GPUs per node
#SBATCH --mem=256G                      # 256GB RAM per node
#SBATCH --time=12:00:00                 # 12 hour time limit
#SBATCH --output=logs/slurm/train-multi-%j.out
#SBATCH --error=logs/slurm/train-multi-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# Multi-node training requires shared storage
#SBATCH --constraint=shared_storage     # Ensure shared filesystem

# ============================================================================
# SLURM Environment Information
# ============================================================================
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of Nodes: $SLURM_NNODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_TASKS_PER_NODE"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Nodelist: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# ============================================================================
# Get Master Node Information
# ============================================================================
# Get the hostname of the first node (master node)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Distributed Training Configuration:"
echo "  Master Address: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  World Size: $SLURM_NTASKS"
echo "  Nodes: $SLURM_NNODES"
echo "  GPUs per Node: 4"
echo "  Total GPUs: $((SLURM_NNODES * 4))"
echo ""

# ============================================================================
# Load Modules
# ============================================================================
module purge
module load python/3.9
module load cuda/12.1
module load nccl/2.18
module load openmpi/4.1  # If using MPI backend

echo "Loaded modules:"
module list
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
source venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL settings for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                    # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=5                # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=ib0              # Network interface
export NCCL_IB_HCA=mlx5_0                  # InfiniBand adapter
export NCCL_IB_GID_INDEX=3                 # InfiniBand GID index
export NCCL_SOCKET_TIMEOUT=3600            # Increase timeout for slow networks

# Additional performance settings
export NCCL_ALGO=Ring                      # NCCL algorithm
export NCCL_CROSS_NIC=1                    # Enable cross-NIC communication
export NCCL_MIN_NCHANNELS=4                # Minimum channels

# DeepSpeed settings
export DEEPSPEED_DISABLE_NUMA_AFFINITY=1

echo "Environment configured for multi-node training"
echo ""

# ============================================================================
# Training Configuration
# ============================================================================
MODEL_CONFIG="configs/models/smollm_1.7b.yaml"
TRAINING_CONFIG="configs/training/cpt.yaml"
DEEPSPEED_CONFIG="configs/deepspeed/zero3.json"
OUTPUT_DIR="models/checkpoints/cpt/smollm_1.7b_${SLURM_JOB_ID}"
RUN_NAME="smollm_1.7b_cpt_multi_${SLURM_JOB_ID}"

mkdir -p $OUTPUT_DIR
mkdir -p logs/slurm

echo "Training Configuration:"
echo "  Model Config: $MODEL_CONFIG"
echo "  Training Config: $TRAINING_CONFIG"
echo "  DeepSpeed Config: $DEEPSPEED_CONFIG (ZeRO-3 for multi-node)"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Run Name: $RUN_NAME"
echo ""

# ============================================================================
# Launch Training with SRUN
# ============================================================================
echo "=========================================="
echo "Starting Multi-Node Training at $(date)"
echo "=========================================="

# Method 1: Using torchrun directly (recommended)
srun --jobid=$SLURM_JOB_ID bash -c "
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_backend=c10d \
        scripts/train.py \
        --config $TRAINING_CONFIG \
        --model-config $MODEL_CONFIG \
        --deepspeed $DEEPSPEED_CONFIG \
        --output-dir $OUTPUT_DIR \
        --run-name $RUN_NAME
"

# Method 2: Using DeepSpeed launcher (alternative)
# deepspeed --hostfile=/tmp/hostfile_${SLURM_JOB_ID} \
#     --num_nodes=$SLURM_NNODES \
#     --num_gpus=4 \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     scripts/train.py \
#     --config $TRAINING_CONFIG \
#     --model-config $MODEL_CONFIG \
#     --deepspeed $DEEPSPEED_CONFIG \
#     --output-dir $OUTPUT_DIR \
#     --run-name $RUN_NAME

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training Completed at $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

# ============================================================================
# Post-Training
# ============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo "Multi-node training succeeded!"
    
    # Save job info
    echo "Job completed successfully on $(date)" > $OUTPUT_DIR/job_info.txt
    echo "Nodes used: $SLURM_NODELIST" >> $OUTPUT_DIR/job_info.txt
    echo "Total GPUs: $((SLURM_NNODES * 4))" >> $OUTPUT_DIR/job_info.txt
else
    echo "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

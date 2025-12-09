#!/bin/bash
#SBATCH --job-name=llm-train-single
#SBATCH --account=YOUR_ACCOUNT          # Replace with your account
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks-per-node=4             # 4 tasks (one per GPU)
#SBATCH --cpus-per-task=8               # 8 CPUs per GPU
#SBATCH --gres=gpu:4                    # 4 GPUs
#SBATCH --mem=256G                      # 256GB RAM
#SBATCH --time=04:00:00                 # 4 hour time limit
#SBATCH --output=logs/slurm/train-single-%j.out
#SBATCH --error=logs/slurm/train-single-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications
#SBATCH --mail-user=your.email@example.com

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
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Nodelist: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# ============================================================================
# Load Modules (adjust for your cluster)
# ============================================================================
module purge
module load python/3.9
module load cuda/12.1
module load nccl/2.18

# Show loaded modules
echo "Loaded modules:"
module list
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL settings for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                 # Enable InfiniBand (if available)
export NCCL_NET_GDR_LEVEL=5             # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=ib0           # Network interface (adjust for your cluster)

# Prevent CPU affinity issues
export DEEPSPEED_DISABLE_NUMA_AFFINITY=1

echo "Environment variables set"
echo "Python: $(which python)"
echo "CUDA: $CUDA_VISIBLE_DEVICES"
echo ""

# ============================================================================
# Training Configuration
# ============================================================================
MODEL_CONFIG="configs/models/smollm_360m.yaml"
TRAINING_CONFIG="configs/training/sft.yaml"
DEEPSPEED_CONFIG="configs/deepspeed/zero2.json"
OUTPUT_DIR="models/checkpoints/sft/smollm_360m_${SLURM_JOB_ID}"
RUN_NAME="smollm_360m_sft_single_${SLURM_JOB_ID}"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs/slurm

echo "Training Configuration:"
echo "  Model Config: $MODEL_CONFIG"
echo "  Training Config: $TRAINING_CONFIG"
echo "  DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Run Name: $RUN_NAME"
echo ""

# ============================================================================
# Launch Training
# ============================================================================
echo "=========================================="
echo "Starting Training at $(date)"
echo "=========================================="

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    scripts/train.py \
    --config $TRAINING_CONFIG \
    --model-config $MODEL_CONFIG \
    --deepspeed $DEEPSPEED_CONFIG \
    --output-dir $OUTPUT_DIR \
    --run-name $RUN_NAME

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training Completed at $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

# ============================================================================
# Post-Training Tasks
# ============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training succeeded!"
    
    # Optional: Run evaluation
    # python scripts/evaluate.py \
    #     --checkpoint $OUTPUT_DIR/final \
    #     --benchmarks humaneval \
    #     --output-dir results/${RUN_NAME}
    
else
    echo "Training failed with exit code $EXIT_CODE"
fi

# Cleanup (optional)
# rm -rf $OUTPUT_DIR/checkpoint-*

exit $EXIT_CODE

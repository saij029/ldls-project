#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/eval-%j.out
#SBATCH --error=logs/slurm/eval-%j.err

# ============================================================================
# Evaluation Job
# ============================================================================
echo "=========================================="
echo "Evaluation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Load modules
module purge
module load python/3.9
module load cuda/12.1

# Activate environment
source venv/bin/activate

# Configuration
CHECKPOINT="models/final/smollm-360m-sft"
OUTPUT_DIR="results/eval_${SLURM_JOB_ID}"

mkdir -p $OUTPUT_DIR

echo "Evaluating checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation
python scripts/evaluate.py \
    --checkpoint $CHECKPOINT \
    --benchmarks humaneval mbpp \
    --output-dir $OUTPUT_DIR \
    --num-samples 1 \
    --temperature 0.0

EXIT_CODE=$?

echo ""
echo "Evaluation completed with exit code: $EXIT_CODE"

exit $EXIT_CODE

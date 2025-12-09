
# SLURM Job Templates

Production-ready SLURM scripts for distributed training.

---

## Available Templates

| Script | Purpose | Resources | Time |
|--------|---------|-----------|------|
| `train_single.slurm` | Single-node training | 1 node, 4 GPUs | 4 hours |
| `train_multi.slurm` | Multi-node training | 4 nodes, 16 GPUs | 12 hours |
| `eval.slurm` | Model evaluation | 1 node, 1 GPU | 2 hours |

---

## Quick Start

### 1. Configure Your Account

Edit the SLURM scripts and replace:
- `YOUR_ACCOUNT` with your cluster account
- `your.email@example.com` with your email
- Partition names (e.g., `gpu`, `a100`, etc.)

### 2. Submit Jobs

```
# Single-node training
sbatch slurm/train_single.slurm

# Multi-node training
sbatch slurm/train_multi.slurm

# Evaluation
sbatch slurm/eval.slurm
```

### 3. Monitor Jobs

```
# Check job status
squeue -u $USER

# View detailed job info
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View output logs
tail -f logs/slurm/train-single-<job_id>.out
```

---

## Customization Guide

### Adjust Resources

#### GPUs per Node
```
#SBATCH --gres=gpu:4              # Request 4 GPUs
#SBATCH --ntasks-per-node=4        # One task per GPU
```

#### Number of Nodes
```
#SBATCH --nodes=4                  # 4 nodes for multi-node
```

#### Memory
```
#SBATCH --mem=256G                 # 256GB RAM per node
```

#### Time Limit
```
#SBATCH --time=12:00:00            # 12 hours
#SBATCH --time=1-00:00:00          # 1 day
```

### Specific GPU Types

```
# Request specific GPU model
#SBATCH --gres=gpu:a100:4          # 4x A100 GPUs
#SBATCH --gres=gpu:v100:4          # 4x V100 GPUs
#SBATCH --gres=gpu:h100:4          # 4x H100 GPUs
```

### Priority and QOS

```
# High priority queue
#SBATCH --qos=high

# Low priority (preemptible) queue
#SBATCH --qos=low
#SBATCH --requeue                  # Auto-requeue if preempted
```

---

## Cluster-Specific Settings

### Network Interface

Update for your cluster's network:

```
# InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0

# Ethernet
export NCCL_SOCKET_IFNAME=eth0

# Check available interfaces
ip addr show
```

### Module Names

Update module names for your cluster:

```
# Example: NERSC
module load python/3.9-anaconda-2021.11
module load cuda/11.7
module load nccl/2.15.5-cuda11

# Example: TACC
module load python3/3.9.2
module load cuda/12.0
module load nccl/2.16.2

# Check available modules
module avail
```

---

## Multi-Node Troubleshooting

### Connection Issues

**Problem:** Nodes can't communicate

**Solutions:**
```
# 1. Test connectivity
srun --nodes=2 hostname
srun --nodes=2 --ntasks-per-node=1 ping -c 3 <master_node>

# 2. Check firewall
srun --nodes=2 nc -zv <master_node> 29500

# 3. Increase timeout
export NCCL_SOCKET_TIMEOUT=7200

# 4. Enable debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### NCCL Errors

**Problem:** NCCL initialization fails

**Solutions:**
```
# 1. Disable InfiniBand if not available
export NCCL_IB_DISABLE=1

# 2. Force specific interface
export NCCL_SOCKET_IFNAME=eth0

# 3. Use Ethernet fallback
export NCCL_NET_GDR_LEVEL=0
```

### Shared Storage Issues

**Problem:** Checkpoints not visible across nodes

**Solution:**
```
# Ensure output directory is on shared storage
# Check with:
df -h | grep $OUTPUT_DIR

# Use absolute paths in scripts
OUTPUT_DIR="/shared/storage/models/checkpoints/..."
```

---

## Performance Optimization

### NCCL Tuning for InfiniBand

```
# Optimal settings for IB
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Multiple adapters
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=4
```

### CPU Affinity

```
# Bind tasks to specific CPUs (cluster-dependent)
#SBATCH --cpu-bind=cores

# Or in script
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

### GPU Direct RDMA

```
# Enable GPU Direct (requires kernel support)
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Verify with
nvidia-smi topo -m
```

---

## Job Arrays

Submit multiple experiments:

```
#!/bin/bash
#SBATCH --array=0-4                    # 5 experiments
#SBATCH --output=logs/slurm/array-%A_%a.out

# Different learning rates
LR_ARRAY=(5e-5 1e-4 2e-4 5e-4 1e-3)
LR=${LR_ARRAY[$SLURM_ARRAY_TASK_ID]}

echo "Running experiment with LR=$LR"

# Update config with LR and run
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --learning-rate $LR
```

Submit:
```
sbatch slurm/array_experiment.slurm
```

---

## Job Dependencies

Chain jobs (e.g., train → eval):

```
# Submit training
TRAIN_JOB=$(sbatch --parsable slurm/train_single.slurm)

# Submit evaluation (runs after training succeeds)
sbatch --dependency=afterok:$TRAIN_JOB slurm/eval.slurm
```

---

## Example Workflows

### Full Pipeline

```
# 1. Download data (login node or small job)
python scripts/download_data.py --datasets code_alpaca

# 2. Submit training
TRAIN_JOB=$(sbatch --parsable slurm/train_single.slurm)

# 3. Submit evaluation (after training)
EVAL_JOB=$(sbatch --dependency=afterok:$TRAIN_JOB slurm/eval.slurm)

# 4. Monitor
watch -n 10 squeue -u $USER
```

### Checkpointed Long Job

```
#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --requeue                      # Requeue if preempted
#SBATCH --open-mode=append             # Append to logs

# Enable auto-restart from checkpoint
CHECKPOINT_DIR="models/checkpoints/cpt/smollm_1.7b"

if [ -d "$CHECKPOINT_DIR/latest" ]; then
    echo "Resuming from checkpoint"
    RESUME_FLAG="--resume-from-checkpoint $CHECKPOINT_DIR/latest"
else
    echo "Starting fresh training"
    RESUME_FLAG=""
fi

torchrun ... scripts/train.py $RESUME_FLAG ...
```

---

## Monitoring Tips

### Real-time GPU Usage

```
# On compute node (via ssh or srun)
watch -n 1 nvidia-smi

# Or from submission
srun --jobid=<job_id> --pty bash
watch nvidia-smi
```

### Training Progress

```
# Follow training logs
tail -f logs/slurm/train-single-<job_id>.out

# Grep for metrics
grep "loss:" logs/slurm/train-single-*.out

# TensorBoard on cluster
sbatch --wrap="tensorboard --logdir=models/checkpoints/sft/tensorboard --port=6006"
# Then forward port: ssh -L 6006:compute_node:6006 cluster
```

---

## Additional Resources

- [SLURM Documentation](https://slurm.schedmd.com/)
- [DeepSpeed Multi-Node Setup](https://www.deepspeed.ai/getting-started/#multi-node-training)
- [PyTorch Distributed Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Best Practices](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/troubleshooting.html)



### 🚀 **Ready to Use:**
```bash
# Download & prepare
python scripts/download_data.py --models smollm_360m --datasets code_alpaca
python scripts/prepare_data.py --dataset sft --input data/raw/code_alpaca/train.jsonl

# Train locally
torchrun --nproc_per_node=4 scripts/train.py --config configs/training/sft.yaml

# Or submit to cluster
sbatch slurm/train_multi.slurm
```


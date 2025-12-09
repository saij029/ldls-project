Perfect! Here's the properly formatted `scripts/README.md`:

***

## **Create `scripts/README.md`:**

```markdown
# Training Scripts

Complete guide for using the training pipeline scripts.

---

## Quick Start

### 1. Download Data and Models

#### Download Models

```
# Download a specific model
python scripts/download_data.py --models smollm_135m --model-dir models/base

# Download multiple models
python scripts/download_data.py --models smollm_135m smollm_360m --model-dir models/base

# Download all models
python scripts/download_data.py --models all --model-dir models/base
```

#### Download Datasets

```
# Download CodeParrot for CPT (with sample limit for testing)
python scripts/download_data.py \
    --datasets codeparrot \
    --data-dir data/raw \
    --max-samples 10000

# Download CodeAlpaca for SFT
python scripts/download_data.py \
    --datasets code_alpaca \
    --data-dir data/raw

# Download evaluation datasets
python scripts/download_data.py \
    --datasets humaneval mbpp \
    --data-dir data/raw

# Download all datasets (warning: large!)
python scripts/download_data.py \
    --datasets all \
    --data-dir data/raw \
    --max-samples 100000
```

---

### 2. Prepare Data

#### Prepare CPT Data

```
# Prepare CodeParrot for continuous pretraining
python scripts/prepare_data.py \
    --dataset cpt \
    --input data/raw/codeparrot/train.jsonl \
    --output data/processed/cpt/train \
    --tokenizer HuggingFaceTB/SmolLM-135M \
    --max-length 2048

# With custom text column
python scripts/prepare_data.py \
    --dataset cpt \
    --input data/raw/codeparrot/train.jsonl \
    --output data/processed/cpt/train \
    --tokenizer HuggingFaceTB/SmolLM-135M \
    --text-column content
```

#### Prepare SFT Data

```
# Prepare CodeAlpaca for supervised finetuning
python scripts/prepare_data.py \
    --dataset sft \
    --input data/raw/code_alpaca/train.jsonl \
    --output data/processed/sft/train \
    --tokenizer HuggingFaceTB/SmolLM-135M \
    --max-length 2048

# Prepare validation data
python scripts/prepare_data.py \
    --dataset sft \
    --input data/raw/code_alpaca/val.jsonl \
    --output data/processed/sft/val \
    --tokenizer HuggingFaceTB/SmolLM-135M
```

---

### 3. Train

#### Single GPU Training

```
# Basic single GPU training
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/sft/smollm_135m \
    --run-name test_run

# CPT training
python scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/cpt/smollm_135m \
    --run-name cpt_run
```

#### Multi-GPU Training (Single Node)

```
# Training with 4 GPUs using DeepSpeed ZeRO-2
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/sft/smollm_360m \
    --run-name multi_gpu_run

# Training with 8 GPUs
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/cpt/smollm_1.7b
```

#### Multi-Node Training

```
# Node 0 (master node)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml \
    --deepspeed configs/deepspeed/zero3.json \
    --output-dir models/checkpoints/cpt/smollm_1.7b

# Node 1 (worker node)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml \
    --deepspeed configs/deepspeed/zero3.json \
    --output-dir models/checkpoints/cpt/smollm_1.7b
```

#### Using DeepSpeed Launcher

```
# Alternative to torchrun
deepspeed --num_gpus=4 scripts/train.py \
    --deepspeed configs/deepspeed/zero2.json \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --output-dir models/checkpoints/sft/smollm_360m

# Multi-node with DeepSpeed launcher
deepspeed \
    --num_nodes=2 \
    --num_gpus=4 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train.py \
    --deepspeed configs/deepspeed/zero3.json \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml
```

---

## Environment Variables

### GPU Selection

```
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/train.py ...

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py ...
```

### Distributed Training

```
# Master node configuration (for multi-node)
export MASTER_ADDR=192.168.1.100  # IP of master node
export MASTER_PORT=29500           # Port for communication

# Local rank (usually set automatically)
export LOCAL_RANK=0
```

### DeepSpeed Configuration

```
# Enable DeepSpeed debug logging
export DEEPSPEED_LOG_LEVEL=DEBUG

# Disable CPU affinity (may help with SLURM)
export DEEPSPEED_DISABLE_NUMA_AFFINITY=1
```

### NCCL Configuration

```
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Set NCCL timeout (useful for slow networks)
export NCCL_SOCKET_TIMEOUT=3600

# Use InfiniBand (if available)
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

### Logging

```
# Set log level
export LOGLEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# PyTorch distributed debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Script Reference

### download_data.py

**Purpose:** Download models and datasets from HuggingFace Hub.

**Arguments:**
- `--models`: Models to download (smollm_135m, smollm_360m, smollm_1.7b, all)
- `--datasets`: Datasets to download (codeparrot, code_alpaca, humaneval, mbpp, all)
- `--model-dir`: Directory to save models (default: `models/base`)
- `--data-dir`: Directory to save datasets (default: `data/raw`)
- `--max-samples`: Maximum samples per dataset (optional, for testing)

**Examples:**
```
# Download single model
python scripts/download_data.py --models smollm_135m

# Download multiple datasets with sample limit
python scripts/download_data.py \
    --datasets codeparrot code_alpaca \
    --max-samples 50000
```

---

### prepare_data.py

**Purpose:** Tokenize and prepare datasets for training.

**Arguments:**
- `--dataset`: Dataset type (`cpt` or `sft`)
- `--input`: Input JSONL file path
- `--output`: Output directory
- `--tokenizer`: Tokenizer name or path (default: `HuggingFaceTB/SmolLM-135M`)
- `--max-length`: Maximum sequence length (default: 2048)
- `--text-column`: Name of text column for CPT (default: `content`)

**Examples:**
```
# Prepare CPT data
python scripts/prepare_data.py \
    --dataset cpt \
    --input data/raw/codeparrot/train.jsonl \
    --output data/processed/cpt/train \
    --max-length 2048

# Prepare SFT data with custom tokenizer
python scripts/prepare_data.py \
    --dataset sft \
    --input data/raw/code_alpaca/train.jsonl \
    --output data/processed/sft/train \
    --tokenizer models/base/smollm_360m
```

---

### train.py

**Purpose:** Main training script with DeepSpeed support.

**Arguments:**
- `--config`: Path to training config YAML (required)
- `--model-config`: Path to model config YAML (required)
- `--deepspeed`: Path to DeepSpeed config JSON (optional)
- `--output-dir`: Output directory (overrides config)
- `--run-name`: Run name for logging
- `--local_rank`: Local rank for distributed training (auto-set)

**Examples:**
```
# Single GPU
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/test \
    --run-name test_run

# Multi-GPU with DeepSpeed
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --deepspeed configs/deepspeed/zero2.json
```

---

## Common Workflows

### Quick Test Run (Small Dataset)

```
# 1. Download small dataset
python scripts/download_data.py \
    --datasets code_alpaca \
    --data-dir data/raw

# 2. Prepare data
python scripts/prepare_data.py \
    --dataset sft \
    --input data/raw/code_alpaca/train.jsonl \
    --output data/processed/sft/train \
    --tokenizer HuggingFaceTB/SmolLM-135M

# 3. Train on single GPU
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/test
```

### Production CPT Pipeline

```
# 1. Download large model and dataset
python scripts/download_data.py \
    --models smollm_1.7b \
    --datasets codeparrot \
    --max-samples 1000000

# 2. Prepare CPT data
python scripts/prepare_data.py \
    --dataset cpt \
    --input data/raw/codeparrot/train.jsonl \
    --output data/processed/cpt/train

# 3. Multi-GPU training with ZeRO-3
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml \
    --deepspeed configs/deepspeed/zero3.json \
    --output-dir models/checkpoints/cpt/smollm_1.7b_prod
```

### Resume Training from Checkpoint

```
# Update config to include checkpoint path
# Edit configs/training/sft.yaml:
# training:
#   resume_from_checkpoint: "models/checkpoints/sft/smollm_135m/checkpoint-1000"

# Run training
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml
```

---

## Troubleshooting

### Issue: "No module named 'llm_trainer'"

**Solution:**
```
pip install -e .
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size in config
2. Increase gradient accumulation steps
3. Use ZeRO-3 with CPU offloading
4. Enable gradient checkpointing

```
# In configs/training/sft.yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
```

### Issue: "Connection timeout in multi-node"

**Solutions:**
```
# Increase NCCL timeout
export NCCL_SOCKET_TIMEOUT=7200

# Check connectivity
ping $MASTER_ADDR
nc -zv $MASTER_ADDR $MASTER_PORT
```

### Issue: "DeepSpeed import error"

**Solution:**
```
pip install deepspeed
# Or with specific CUDA version
DS_BUILD_OPS=1 pip install deepspeed
```

---

## Performance Tips

1. **Use BF16 mixed precision** (enabled by default in configs)
2. **Increase num_workers** for faster data loading (4-8 per GPU)
3. **Use NVMe/SSD** for data storage
4. **Enable Flash Attention 2** for supported models
5. **Profile first** on small dataset before full training

---

## Monitoring Training

### View Logs

```
# Training logs
tail -f logs/training/run_*/train_rank_0.log

# SLURM logs
tail -f logs/slurm-*.out
```

### TensorBoard

```
# Start TensorBoard
tensorboard --logdir models/checkpoints/sft/smollm_135m/tensorboard

# Open browser at http://localhost:6006
```

### GPU Monitoring

```
# Watch GPU usage
watch -n 1 nvidia-smi

# More detailed monitoring
nvtop  # Install: sudo apt install nvtop
```

---

## Additional Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

---

**Need help?** Check the main [README.md](../README.md) or open an issue on GitHub.
```

***

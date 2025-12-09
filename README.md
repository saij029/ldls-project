# LLM Code Trainer

Production-grade continuous pretraining and finetuning pipeline for code language models using DeepSpeed and multi-node GPU clusters.

## Features

- 🚀 **Multi-node distributed training** with DeepSpeed ZeRO-2/3
- 📚 **Continuous pretraining** on CodeParrot Clean (1B tokens)
- 🎯 **Supervised finetuning** on CodeAlpaca-20K
- 📊 **Comprehensive evaluation** on HumanEval, MBPP, LiveCodeBench
- ⚡ **Efficient inference** with vLLM and INT8 quantization
- 📈 **Production monitoring** with DCGM, W&B, TensorBoard
- 🔄 **Gradient checkpointing** and mixed precision (BF16/FP16)
- 💾 **Smart checkpointing** with automatic cleanup

## Supported Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| SmolLM-135M | 135M | Ultra-small, fast iteration |
| SmolLM-360M | 360M | Small, balanced performance |
| SmolLM-1.7B | 1.7B | Medium-small, best quality |

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [SLURM Cluster](#slurm-cluster)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ or 12.1+ (for GPU training)
- 4+ GPUs recommended for multi-node training

### Step 1: Clone Repository

```
git clone https://github.com/saij029/ldls-project.git
cd ldls-project
```

### Step 2: Create Virtual Environment

```
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows Git Bash)
source venv/Scripts/activate

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Optional: Install inference dependencies
pip install vllm ray

# Optional: Install evaluation dependencies
pip install human-eval evaluate
```

### Verify Installation

```
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

---

## Quick Start

### 1. Download Models and Data

```
# Download SmolLM-135M model
python scripts/download_data.py \
    --models smollm_135m \
    --model-dir models/base

# Download CodeAlpaca dataset (small, for testing)
python scripts/download_data.py \
    --datasets code_alpaca \
    --data-dir data/raw

# Optional: Download CodeParrot (large dataset, use --max-samples for subset)
python scripts/download_data.py \
    --datasets codeparrot \
    --data-dir data/raw \
    --max-samples 100000
```

### 2. Prepare Datasets

```
# Prepare SFT data (CodeAlpaca)
python scripts/prepare_data.py \
    --dataset sft \
    --input data/raw/code_alpaca/train.jsonl \
    --output data/processed/sft/train \
    --tokenizer HuggingFaceTB/SmolLM-135M

# Optional: Prepare CPT data (CodeParrot)
python scripts/prepare_data.py \
    --dataset cpt \
    --input data/raw/codeparrot/train.jsonl \
    --output data/processed/cpt/train \
    --tokenizer HuggingFaceTB/SmolLM-135M
```

### 3. Train Model

#### Single GPU Training

```
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/sft/smollm_135m \
    --run-name my_first_run
```

#### Multi-GPU Training (Single Node)

```
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/sft/smollm_135m \
    --run-name multi_gpu_run
```

#### Multi-Node Training (SLURM)

```
sbatch slurm/train_multi.slurm
```

---

## Configuration

### Model Configurations

Located in `configs/models/`:

- `smollm_135m.yaml` - 135M parameter model
- `smollm_360m.yaml` - 360M parameter model
- `smollm_1.7b.yaml` - 1.7B parameter model

**Example (`smollm_135m.yaml`):**

```
model:
  name: "smollm_135m"
  model_name_or_path: "HuggingFaceTB/SmolLM-135M"
  num_parameters: 135000000
  max_seq_length: 2048
  use_flash_attention: false
  gradient_checkpointing: true
  torch_dtype: "bfloat16"
```

### Training Configurations

Located in `configs/training/`:

- `cpt.yaml` - Continuous pretraining settings
- `sft.yaml` - Supervised finetuning settings

**Example (`sft.yaml`):**

```
data:
  dataset_type: "sft"
  train_path: "data/processed/sft/train"
  val_path: "data/processed/sft/val"
  num_workers: 4

training:
  output_dir: "models/checkpoints/sft"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
  bf16: true
  logging_steps: 10
  save_steps: 200
  eval_steps: 200
```

### DeepSpeed Configurations

Located in `configs/deepspeed/`:

- `zero2.json` - ZeRO Stage 2 (single-node, recommended)
- `zero3.json` - ZeRO Stage 3 (multi-node, CPU offloading)

**When to use which:**
- **ZeRO-2**: Single node, up to 8 GPUs, models <2B parameters
- **ZeRO-3**: Multi-node, large models, CPU/NVMe offloading needed

---

## Training

### Training Modes

#### 1. Continuous Pretraining (CPT)

Train on raw code data to adapt model to your code domain.

```
# Update configs/training/cpt.yaml with your data paths
python scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/cpt/smollm_360m
```

#### 2. Supervised Finetuning (SFT)

Train on instruction-response pairs for code generation tasks.

```
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/sft/smollm_360m
```

### Resume Training

```
python scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml \
    --output-dir models/checkpoints/sft/smollm_135m \
    --resume-from-checkpoint models/checkpoints/sft/smollm_135m/checkpoint-1000
```

### Monitoring Training

#### TensorBoard

```
tensorboard --logdir models/checkpoints/sft/smollm_135m/tensorboard
```

#### Weights & Biases

Edit `configs/training/sft.yaml`:

```
monitoring:
  wandb_project: "llm-code-trainer"
  wandb_entity: "your-username"
  wandb_run_name: "smollm-135m-sft"
```

### Training Time Estimates

On 4x A100 GPUs (80GB):

| Model | Dataset | Epochs | Time |
|-------|---------|--------|------|
| SmolLM-135M | CodeAlpaca-20K (SFT) | 3 | 10-15 min |
| SmolLM-360M | CodeAlpaca-20K (SFT) | 3 | 15-20 min |
| SmolLM-1.7B | CodeAlpaca-20K (SFT) | 3 | 30-40 min |
| SmolLM-135M | CodeParrot-1B (CPT) | 3 | 40-50 min |
| SmolLM-360M | CodeParrot-1B (CPT) | 3 | 60-80 min |

---

## Evaluation

### Run Benchmarks

```
# Evaluate on HumanEval
python scripts/evaluate.py \
    --checkpoint models/final/smollm-135m-sft \
    --benchmarks humaneval \
    --output results/humaneval_results.json

# Evaluate on MBPP
python scripts/evaluate.py \
    --checkpoint models/final/smollm-135m-sft \
    --benchmarks mbpp \
    --output results/mbpp_results.json

# Evaluate on both
python scripts/evaluate.py \
    --checkpoint models/final/smollm-135m-sft \
    --benchmarks humaneval mbpp \
    --output results/combined_results.json
```

### Expected Results

| Model | HumanEval pass@1 | MBPP pass@1 |
|-------|------------------|-------------|
| SmolLM-135M (base) | 2-4% | 5-8% |
| SmolLM-135M (SFT) | 8-12% | 10-15% |
| SmolLM-360M (SFT) | 15-20% | 18-25% |
| SmolLM-1.7B (SFT) | 25-32% | 28-35% |

---

## Inference

### Using vLLM (Recommended)

```
# Start vLLM server
python scripts/serve.py \
    --model models/final/smollm-135m-sft \
    --port 8000 \
    --tensor-parallel-size 1

# Query the server
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "smollm-135m-sft",
        "prompt": "def fibonacci(n):",
        "max_tokens": 100,
        "temperature": 0.2
    }'
```

### Batch Inference

```
python scripts/batch_generate.py \
    --model models/final/smollm-135m-sft \
    --input prompts.txt \
    --output generated.txt \
    --batch-size 16
```

### INT8 Quantization

```
# Quantize model
python scripts/quantize.py \
    --model models/final/smollm-360m-sft \
    --output models/final/smollm-360m-sft-int8 \
    --quantization int8

# Use quantized model
python scripts/serve.py \
    --model models/final/smollm-360m-sft-int8 \
    --port 8000
```

---

## SLURM Cluster

### Single Node Training

```
sbatch slurm/train_single.slurm
```

**`slurm/train_single.slurm`:**
```
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

module load python/3.9
module load cuda/12.1

source venv/bin/activate

torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_360m.yaml \
    --deepspeed configs/deepspeed/zero2.json \
    --output-dir models/checkpoints/sft/smollm_360m
```

### Multi-Node Training

```
sbatch slurm/train_multi.slurm
```

**`slurm/train_multi.slurm`:**
```
#!/bin/bash
#SBATCH --job-name=llm-train-multi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out

module load python/3.9
module load cuda/12.1

source venv/bin/activate

# Get master node
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config configs/training/cpt.yaml \
    --model-config configs/models/smollm_1.7b.yaml \
    --deepspeed configs/deepspeed/zero3.json \
    --output-dir models/checkpoints/cpt/smollm_1.7b
```

---

## Project Structure

```
llm-code-trainer/
├── src/llm_trainer/           # Core library
│   ├── config.py              # Configuration management
│   ├── model.py               # Model loading
│   ├── dataset.py             # Dataset classes
│   ├── distributed.py         # Distributed training setup
│   ├── trainer.py             # Main training loop
│   ├── evaluator.py           # Evaluation logic
│   ├── inference.py           # Inference utilities
│   ├── monitor.py             # Resource monitoring
│   └── utils.py               # Helper functions
│
├── scripts/                   # Executable scripts
│   ├── download_data.py       # Download models/datasets
│   ├── prepare_data.py        # Tokenize datasets
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── serve.py               # Inference server
│
├── configs/                   # Configuration files
│   ├── models/                # Model configs
│   ├── training/              # Training configs
│   └── deepspeed/             # DeepSpeed configs
│
├── slurm/                     # SLURM job templates
│   ├── train_single.slurm
│   └── train_multi.slurm
│
├── data/                      # Datasets
│   ├── raw/                   # Downloaded data
│   └── processed/             # Tokenized data
│
├── models/                    # Model weights
│   ├── base/                  # Downloaded models
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained models
│
├── logs/                      # Training logs
├── results/                   # Evaluation results
└── tests/                     # Unit tests
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solutions:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use ZeRO-3 with CPU offloading
- Enable gradient checkpointing

```
# In training config
training:
  per_device_train_batch_size: 2  # Reduce this
  gradient_accumulation_steps: 16  # Increase this
```

#### 2. Slow Training

**Solutions:**
- Use BF16 mixed precision
- Increase `num_workers` in data config
- Use faster storage (SSD/NVMe)
- Enable Flash Attention 2 (for supported models)

```
model:
  use_flash_attention: true

data:
  num_workers: 8  # Increase this
```

#### 3. Multi-Node Connection Issues

**Solutions:**
- Check firewall settings
- Verify `MASTER_ADDR` and `MASTER_PORT`
- Increase timeout in distributed setup
- Use InfiniBand if available

```
# Check connectivity
ping $MASTER_ADDR
nc -zv $MASTER_ADDR $MASTER_PORT
```

#### 4. DeepSpeed Errors

**Solutions:**
- Ensure all nodes have same DeepSpeed version
- Check `--local_rank` is set correctly
- Verify CUDA versions match
- Use `deepspeed` launcher instead of `torchrun`

```
# Use DeepSpeed launcher
deepspeed --num_gpus=4 scripts/train.py \
    --deepspeed configs/deepspeed/zero2.json \
    --config configs/training/sft.yaml \
    --model-config configs/models/smollm_135m.yaml
```

### Debug Mode

Enable detailed logging:

```
export LOGLEVEL=DEBUG
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python scripts/train.py ...
```

---

## Performance Tips

### Data Loading

- Use SSD/NVMe for data storage
- Increase `num_workers` (4-8 per GPU)
- Pre-tokenize large datasets
- Use memory-mapped files for huge datasets

### Memory Optimization

- Enable gradient checkpointing
- Use BF16 instead of FP32
- ZeRO-3 for models >1B parameters
- CPU offloading for very large models

### Speed Optimization

- Use Flash Attention 2
- Increase batch size with gradient accumulation
- Compile model with `torch.compile()` (PyTorch 2.0+)
- Profile with PyTorch Profiler to find bottlenecks

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

---

## Citation

```
@software{llm_code_trainer,
  title = {LDLS Project},
  author = {Sai Varun},
  year = {2025},
  url = {https://github.com/saij029/ldls-project}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [SmolLM](https://huggingface.co/blog/smollm) by HuggingFace
- [DeepSpeed](https://www.deepspeed.ai/) by Microsoft
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- [CodeParrot](https://huggingface.co/codeparrot) dataset
- [CodeAlpaca](https://github.com/sahil280114/codealpaca) dataset

---

## Support

- 💬 Issues: [GitHub Issues](https://github.com/saij029/ldls-project/issues)
- 📖 Docs: [Documentation](https://github.com/saij029/ldls-project/wiki)

---

**Happy Training! 🚀**
```


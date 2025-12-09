"""Main training script."""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.llm_trainer.config import FullConfig, ModelConfig, DataConfig, TrainingConfig
from src.llm_trainer.model import load_model_and_tokenizer
from src.llm_trainer.dataset import create_dataset, get_dataloader
from src.llm_trainer.distributed import initialize_training_environment, cleanup_distributed
from src.llm_trainer.trainer import Trainer
from src.llm_trainer.utils import setup_logging, create_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM with DeepSpeed")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed training
    rank, local_rank, world_size, local_world_size = initialize_training_environment()
    
    # Setup logging
    log_dir = os.path.join(args.output_dir or "logs/training", args.run_name or "default")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_rank_{rank}.log")
    
    logger = setup_logging(
        log_level="INFO",
        log_file=log_file if rank == 0 else None,
        rank=rank
    )
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Load configurations
    logger.info(f"Loading config from: {args.config}")
    logger.info(f"Loading model config from: {args.model_config}")
    
    # Parse configs manually (simplified version)
    import yaml
    
    with open(args.model_config, 'r') as f:
        model_config_dict = yaml.safe_load(f)
    model_config = ModelConfig(**model_config_dict['model'])
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    data_config = DataConfig(**config_dict['data'])
    training_config = TrainingConfig(**config_dict['training'])
    
    # Override output dir if specified
    if args.output_dir:
        training_config.output_dir = args.output_dir
    
    # Set DeepSpeed config
    training_config.deepspeed = args.deepspeed
    training_config.local_rank = local_rank
    
    # Create full config
    config = FullConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = create_dataset(
        data_config,
        tokenizer,
        split="train",
        max_length=model_config.max_seq_length
    )
    
    eval_dataset = None
    if data_config.val_path:
        eval_dataset = create_dataset(
            data_config,
            tokenizer,
            split="val",
            max_length=model_config.max_seq_length
        )
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=training_config.per_device_train_batch_size,
        num_workers=data_config.num_workers,
        shuffle=True
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = get_dataloader(
            eval_dataset,
            batch_size=training_config.per_device_eval_batch_size,
            num_workers=data_config.num_workers,
            shuffle=False
        )
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        use_deepspeed=args.deepspeed is not None
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        cleanup_distributed()
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

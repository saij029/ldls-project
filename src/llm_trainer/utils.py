"""Utility functions for distributed training, logging, and checkpointing."""

import os
import sys
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0
) -> logging.Logger:
    """Setup distributed-aware logging.
    
    Only rank 0 logs to console and file. Other ranks log only errors.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to save logs
        rank: Process rank in distributed training
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("llm_trainer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (only rank 0 for INFO/DEBUG, all ranks for WARNING/ERROR)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if rank == 0:
        console_handler.setLevel(getattr(logging, log_level.upper()))
    else:
        console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)
    
    # File handler (only rank 0)
    if log_file and rank == 0:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "llm_trainer") -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        rank: Process rank (used to vary seed per rank)
    """
    # Add rank to seed for data loading diversity
    effective_seed = seed + rank
    
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    
    # Deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = get_logger()
    logger.info(f"Set seed to {effective_seed} (base: {seed}, rank: {rank})")


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    output_dir: str,
    is_best: bool = False,
    save_optimizer: bool = True,
    deepspeed: bool = False,
    **kwargs
) -> str:
    """Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        step: Current step
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
        save_optimizer: Whether to save optimizer state
        deepspeed: Whether using DeepSpeed (handles saving differently)
        **kwargs: Additional metadata to save
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if deepspeed:
        # DeepSpeed handles model/optimizer saving
        model.save_checkpoint(checkpoint_dir)
    else:
        # Save model state
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if hasattr(model, "module"):  # Unwrap DDP
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        
        # Save optimizer and scheduler
        if save_optimizer:
            optim_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(optimizer.state_dict(), optim_path)
            
            sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
            torch.save(scheduler.state_dict(), sched_path)
    
    # Create/update "latest" symlink
    latest_link = os.path.join(output_dir, "latest")
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(os.path.basename(checkpoint_dir), latest_link)
    
    # Save best model copy
    if is_best:
        best_dir = os.path.join(output_dir, "best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
    
    logger = get_logger()
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    deepspeed: bool = False
) -> Dict[str, Any]:
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        deepspeed: Whether using DeepSpeed
        
    Returns:
        Dictionary with metadata (epoch, step, etc.)
    """
    logger = get_logger()
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    if deepspeed:
        # DeepSpeed handles loading
        model.load_checkpoint(checkpoint_path)
    else:
        # Load model state
        model_path = os.path.join(checkpoint_path, "model.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
        # Load optimizer and scheduler
        if optimizer is not None:
            optim_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optim_path):
                optimizer.load_state_dict(torch.load(optim_path))
        
        if scheduler is not None:
            sched_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(sched_path):
                scheduler.load_state_dict(torch.load(sched_path))
    
    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}, step {metadata['step']}")
    return metadata


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """Remove old checkpoints, keeping only recent ones.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to preserve "best" checkpoint
    """
    if not os.path.exists(output_dir):
        return
    
    # Get all checkpoint directories
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                # Extract step number
                step = int(name.split("-")[1])
                checkpoints.append((step, path))
    
    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    
    # Remove old checkpoints
    to_remove = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
    
    for _, path in to_remove:
        shutil.rmtree(path)
    
    logger = get_logger()
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old checkpoints")


# ============================================================================
# File I/O Helpers
# ============================================================================

def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(input_path: str) -> Dict[str, Any]:
    """Load JSON file to dictionary.
    
    Args:
        input_path: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(input_path, "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable
    }


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., "1.5M", "3.2B")
    """
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(num)


def get_gpu_memory_usage() -> Dict[int, Dict[str, float]]:
    """Get current GPU memory usage.
    
    Returns:
        Dictionary mapping GPU ID to memory stats (allocated, reserved, free)
    """
    if not torch.cuda.is_available():
        return {}
    
    gpu_stats = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - reserved
        
        gpu_stats[i] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "total_gb": total
        }
    
    return gpu_stats


# ============================================================================
# Training Helpers
# ============================================================================

def create_output_dir(base_dir: str, run_name: Optional[str] = None) -> str:
    """Create timestamped output directory.
    
    Args:
        base_dir: Base directory path
        run_name: Optional run name (otherwise uses timestamp)
        
    Returns:
        Created directory path
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    output_dir = os.path.join(base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def is_main_process(rank: int = -1) -> bool:
    """Check if current process is main process (rank 0).
    
    Args:
        rank: Process rank (-1 means not distributed)
        
    Returns:
        True if main process
    """
    if rank == -1:
        return True
    return rank == 0


def wait_for_everyone() -> None:
    """Synchronization barrier for distributed training."""
    if dist.is_initialized():
        dist.barrier()

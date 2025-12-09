"""Main training loop with DeepSpeed integration."""

import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_scheduler
from tqdm import tqdm

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import FullConfig
from .distributed import (
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    all_reduce_mean
)
from .utils import (
    get_logger,
    save_checkpoint,
    load_checkpoint,
    cleanup_checkpoints,
    get_gpu_memory_usage,
    count_parameters,
    format_number
)


class Trainer:
    """Main trainer class for LLM training."""
    
    def __init__(
        self,
        config: FullConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        use_deepspeed: bool = False
    ):
        """Initialize trainer.
        
        Args:
            config: Full training configuration
            model: Model to train
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            optimizer: Optimizer (created automatically if None)
            lr_scheduler: LR scheduler (created automatically if None)
            use_deepspeed: Whether to use DeepSpeed
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.use_deepspeed = use_deepspeed and DEEPSPEED_AVAILABLE
        
        self.logger = get_logger()
        self.rank = get_rank()
        self.world_size = get_world_size()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup output directory
        os.makedirs(config.training.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup optimizer and scheduler
        if not self.use_deepspeed:
            if optimizer is None:
                self.optimizer = self._create_optimizer()
            else:
                self.optimizer = optimizer
            
            if lr_scheduler is None:
                self.lr_scheduler = self._create_scheduler()
            else:
                self.lr_scheduler = lr_scheduler
        
        # Log model info
        if is_main_process():
            param_counts = count_parameters(model)
            self.logger.info(f"Model parameters: {format_number(param_counts['total'])}")
            self.logger.info(f"Trainable parameters: {format_number(param_counts['trainable'])}")
    
    def _setup_logging(self) -> None:
        """Setup TensorBoard and W&B logging."""
        self.tb_writer = None
        self.use_wandb = False
        
        if not is_main_process():
            return
        
        # TensorBoard
        if self.config.monitoring.use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.config.monitoring.tensorboard_dir or os.path.join(
                self.config.training.output_dir, "tensorboard"
            )
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.logger.info(f"TensorBoard logging to: {tb_dir}")
        
        # Weights & Biases
        if self.config.monitoring.wandb_project and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.monitoring.wandb_project,
                entity=self.config.monitoring.wandb_entity,
                name=self.config.monitoring.wandb_run_name,
                config=self.config.to_dict()
            )
            self.use_wandb = True
            self.logger.info("W&B logging initialized")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No weight decay for bias and layer norm
            if "bias" in name or "layernorm" in name.lower() or "layer_norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.training.weight_decay
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            }
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
        
        self.logger.info(f"Created optimizer: {self.config.training.optim}")
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.training.num_train_epochs
        num_training_steps = num_training_steps // self.config.training.gradient_accumulation_steps
        
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        scheduler = get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"Created scheduler: {self.config.training.lr_scheduler_type}")
        self.logger.info(f"Total training steps: {num_training_steps}")
        self.logger.info(f"Warmup steps: {num_warmup_steps}")
        
        return scheduler
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info("=" * 80)
        self.logger.info("Starting training")
        self.logger.info("=" * 80)
        
        # Resume from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self._resume_from_checkpoint(self.config.training.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(self.epoch, self.config.training.num_train_epochs):
            self.epoch = epoch
            
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_train_epochs}")
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Log epoch metrics
            if is_main_process():
                self._log_metrics(train_metrics, prefix="train_epoch")
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                
                if is_main_process():
                    self._log_metrics(eval_metrics, prefix="eval")
                    
                    # Save best model
                    eval_loss = eval_metrics.get("loss", float('inf'))
                    is_best = eval_loss < self.best_eval_loss
                    if is_best:
                        self.best_eval_loss = eval_loss
                        self.logger.info(f"New best model! Eval loss: {eval_loss:.4f}")
            
            barrier()
        
        # Final save
        if is_main_process():
            self._save_final_model()
        
        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        self.logger.info("=" * 80)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Setup progress bar
        if is_main_process():
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        else:
            pbar = self.train_dataloader
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
            
            # Backward pass
            if self.use_deepspeed:
                self.model.backward(loss)
            else:
                loss.backward()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Optimizer step
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_deepspeed:
                    # DeepSpeed handles clipping internally
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_deepspeed:
                    self.model.step()
                else:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    avg_loss = total_loss / num_batches
                    
                    metrics = {
                        "loss": avg_loss * self.config.training.gradient_accumulation_steps,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0] if not self.use_deepspeed else self.config.training.learning_rate,
                        "epoch": self.epoch,
                        "step": self.global_step
                    }
                    
                    if is_main_process():
                        self._log_metrics(metrics, prefix="train")
                        
                        # Update progress bar
                        if isinstance(pbar, tqdm):
                            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                
                # Checkpointing
                if (self.config.training.save_steps and 
                    self.global_step % self.config.training.save_steps == 0):
                    
                    if is_main_process():
                        self._save_checkpoint()
                    
                    barrier()
                
                # Evaluation
                if (self.config.training.eval_steps and 
                    self.eval_dataloader is not None and
                    self.global_step % self.config.training.eval_steps == 0):
                    
                    eval_metrics = self.evaluate()
                    
                    if is_main_process():
                        self._log_metrics(eval_metrics, prefix="eval")
                    
                    self.model.train()
        
        # Epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            "loss": avg_loss * self.config.training.gradient_accumulation_steps,
            "epoch": self.epoch
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not is_main_process()):
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Average across processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.model.device)
        avg_loss_tensor = all_reduce_mean(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item()
        
        return {
            "loss": avg_loss,
            "step": self.global_step
        }
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to console, TensorBoard, and W&B.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names
        """
        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        self.logger.info(f"[{prefix}] {metric_str}")
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}/{key}", value, self.global_step)
        
        # W&B
        if self.use_wandb:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=self.global_step)
        
        # GPU metrics
        if self.config.monitoring.log_gpu_metrics:
            gpu_stats = get_gpu_memory_usage()
            for gpu_id, stats in gpu_stats.items():
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        f"gpu_{gpu_id}/allocated_gb",
                        stats["allocated_gb"],
                        self.global_step
                    )
    
    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        self.logger.info(f"Saving checkpoint at step {self.global_step}")
        
        checkpoint_dir = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer if not self.use_deepspeed else None,
            scheduler=self.lr_scheduler if not self.use_deepspeed else None,
            epoch=self.epoch,
            step=self.global_step,
            output_dir=self.config.training.output_dir,
            deepspeed=self.use_deepspeed
        )
        
        # Cleanup old checkpoints
        if self.config.training.save_total_limit:
            cleanup_checkpoints(
                self.config.training.output_dir,
                keep_last_n=self.config.training.save_total_limit
            )
    
    def _save_final_model(self) -> None:
        """Save final trained model."""
        final_dir = os.path.join(
            self.config.training.output_dir.replace("/checkpoints/", "/final/"),
            self.config.model.name
        )
        
        os.makedirs(final_dir, exist_ok=True)
        
        if self.use_deepspeed:
            # DeepSpeed saves differently
            self.model.save_checkpoint(final_dir)
        else:
            # Save model state
            if hasattr(self.model, "module"):
                self.model.module.save_pretrained(final_dir)
            else:
                self.model.save_pretrained(final_dir)
        
        self.logger.info(f"Saved final model to {final_dir}")
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        metadata = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer if not self.use_deepspeed else None,
            scheduler=self.lr_scheduler if not self.use_deepspeed else None,
            deepspeed=self.use_deepspeed
        )
        
        self.epoch = metadata["epoch"]
        self.global_step = metadata["step"]
        
        self.logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")

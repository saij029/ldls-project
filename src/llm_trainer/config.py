"""Configuration management for LLM training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import json


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str  # e.g., "smollm_135m"
    model_name_or_path: str  # HuggingFace model ID or local path
    num_parameters: int  # Number of parameters (for logging)
    max_seq_length: int = 2048
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    torch_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_type: str  # "cpt" or "sft"
    train_path: str
    val_path: Optional[str] = None
    max_samples: Optional[int] = None  # Limit samples for testing
    num_workers: int = 4
    preprocessing_num_workers: int = 8


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    
    # Logging
    logging_steps: int = 10
    eval_steps: Optional[int] = 500
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Optimization
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Distributed training
    local_rank: int = -1
    deepspeed: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Resumption
    resume_from_checkpoint: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True
    tensorboard_dir: Optional[str] = None
    log_gpu_metrics: bool = True
    gpu_metrics_interval: int = 10  # seconds


@dataclass
class FullConfig:
    """Complete configuration for training pipeline."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "FullConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            FullConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': vars(self.model),
            'data': vars(self.data),
            'training': vars(self.training),
            'monitoring': vars(self.monitoring)
        }
    
    def save(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML file
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_deepspeed_config(config_path: str) -> Dict[str, Any]:
    """Load DeepSpeed configuration from JSON file.
    
    Args:
        config_path: Path to DeepSpeed JSON config
        
    Returns:
        DeepSpeed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

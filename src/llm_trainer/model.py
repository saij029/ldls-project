"""Model loading and configuration for LLM training."""

from typing import Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .config import ModelConfig
from .utils import get_logger, count_parameters, format_number


def load_model_and_tokenizer(
    model_config: ModelConfig,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from HuggingFace or local path.
    
    Args:
        model_config: Model configuration
        device: Device to load model on (None for auto)
        load_in_8bit: Load model in 8-bit precision
        load_in_4bit: Load model in 4-bit precision
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger = get_logger()
    logger.info(f"Loading model: {model_config.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_config)
    
    # Load model
    model = load_model(
        model_config=model_config,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    
    # Log model info
    param_counts = count_parameters(model)
    logger.info(f"Model loaded: {format_number(param_counts['total'])} parameters")
    logger.info(f"  Trainable: {format_number(param_counts['trainable'])}")
    logger.info(f"  Non-trainable: {format_number(param_counts['non_trainable'])}")
    
    return model, tokenizer


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """Load tokenizer with proper configuration.
    
    Args:
        model_config: Model configuration
        
    Returns:
        Configured tokenizer
    """
    logger = get_logger()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Set model max length
    if tokenizer.model_max_length > 100000:  # Default is often too large
        tokenizer.model_max_length = model_config.max_seq_length
        logger.info(f"Set tokenizer max_length to {model_config.max_seq_length}")
    
    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    return tokenizer


def load_model(
    model_config: ModelConfig,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> PreTrainedModel:
    """Load model with specified configuration.
    
    Args:
        model_config: Model configuration
        device: Device to load model on
        load_in_8bit: Load in 8-bit precision
        load_in_4bit: Load in 4-bit precision
        
    Returns:
        Loaded model
    """
    logger = get_logger()
    
    # Determine dtype
    torch_dtype = get_torch_dtype(model_config.torch_dtype)
    
    # Load model config
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=True
    )
    
    # Configure model features
    if hasattr(config, "use_cache"):
        config.use_cache = False  # Disable for training
    
    # Model loading kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": model_config.model_name_or_path,
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    
    # Quantization
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        logger.info("Loading model in 8-bit precision")
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        logger.info("Loading model in 4-bit precision")
    
    # Device map
    if device:
        model_kwargs["device_map"] = device
    elif not (load_in_8bit or load_in_4bit):
        model_kwargs["device_map"] = None  # Manual device placement
    else:
        model_kwargs["device_map"] = "auto"  # Auto for quantization
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Enable gradient checkpointing
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Flash Attention 2 (if supported)
    if model_config.use_flash_attention:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        else:
            logger.warning("Flash Attention 2 not supported for this model")
    
    return model


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype.
    
    Args:
        dtype_str: String representation ("float32", "float16", "bfloat16")
        
    Returns:
        torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(f"Unknown dtype: {dtype_str}. Options: {list(dtype_map.keys())}")
    
    return dtype


def prepare_model_for_training(
    model: PreTrainedModel,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None
) -> PreTrainedModel:
    """Prepare model for training with optional LoRA.
    
    Args:
        model: Pre-trained model
        use_lora: Whether to use LoRA adapters
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA (None for auto)
        
    Returns:
        Prepared model
    """
    logger = get_logger()
    
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Default target modules for causal LM
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            logger.info("LoRA adapters added to model")
            
        except ImportError:
            logger.error("peft library not installed. Install with: pip install peft")
            raise
    
    # Enable input require grads for gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model


def get_model_memory_footprint(model: PreTrainedModel) -> dict:
    """Calculate model memory footprint.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with memory statistics
    """
    # Get parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    
    # Get buffer memory
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**3
    
    total_memory = param_memory + buffer_memory
    
    return {
        "parameters_gb": param_memory,
        "buffers_gb": buffer_memory,
        "total_gb": total_memory
    }


def freeze_model_layers(
    model: PreTrainedModel,
    num_layers_to_freeze: int = 0,
    freeze_embeddings: bool = False
) -> PreTrainedModel:
    """Freeze specific layers of the model.
    
    Args:
        model: Model to freeze layers in
        num_layers_to_freeze: Number of initial layers to freeze
        freeze_embeddings: Whether to freeze embedding layer
        
    Returns:
        Model with frozen layers
    """
    logger = get_logger()
    
    # Freeze embeddings
    if freeze_embeddings:
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            for param in model.model.embed_tokens.parameters():
                param.requires_grad = False
            logger.info("Froze embedding layer")
    
    # Freeze transformer layers
    if num_layers_to_freeze > 0:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            for i in range(min(num_layers_to_freeze, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
            logger.info(f"Froze {num_layers_to_freeze} transformer layers")
    
    # Log trainable parameters
    param_counts = count_parameters(model)
    logger.info(f"Trainable parameters: {format_number(param_counts['trainable'])}")
    
    return model

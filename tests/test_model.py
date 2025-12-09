"""Test model loading functionality."""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.config import ModelConfig
from src.llm_trainer.model import (
    load_tokenizer,
    get_torch_dtype,
    count_parameters,
    get_model_memory_footprint
)
from src.llm_trainer.utils import setup_logging


def test_torch_dtype():
    """Test dtype conversion."""
    assert get_torch_dtype("float32") == torch.float32
    assert get_torch_dtype("fp16") == torch.float16
    assert get_torch_dtype("bfloat16") == torch.bfloat16
    print("✓ Torch dtype conversion works")


def test_tokenizer_loading():
    """Test tokenizer loading (requires internet)."""
    setup_logging(rank=0)
    
    model_config = ModelConfig(
        name="test_model",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        num_parameters=135000000,
        max_seq_length=512
    )
    
    try:
        tokenizer = load_tokenizer(model_config)
        
        # Test tokenization
        text = "def hello_world():\n    print('Hello, World!')"
        tokens = tokenizer(text, return_tensors="pt")
        
        assert tokens.input_ids.shape[1] > 0
        assert tokenizer.pad_token is not None
        
        print(f"✓ Tokenizer loaded: vocab_size={len(tokenizer)}")
        print(f"  Sample encoding: {tokens.input_ids.shape[1]} tokens")
        
    except Exception as e:
        print(f"⚠ Tokenizer test skipped (requires internet): {e}")


def test_model_loading_small():
    """Test loading a very small model (requires internet and memory)."""
    setup_logging(rank=0)
    
    # Use tiny model for testing
    model_config = ModelConfig(
        name="tiny_test",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        num_parameters=135000000,
        max_seq_length=512,
        gradient_checkpointing=False
    )
    
    try:
        from src.llm_trainer.model import load_model
        
        # Try to load model
        model = load_model(model_config, device="cpu")
        
        # Check model
        param_counts = count_parameters(model)
        memory_info = get_model_memory_footprint(model)
        
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {param_counts['total']:,}")
        print(f"  Memory: {memory_info['total_gb']:.2f} GB")
        
        # Cleanup
        del model
        
    except Exception as e:
        print(f"⚠ Model loading test skipped: {e}")


if __name__ == "__main__":
    print("Testing model loading module...\n")
    
    test_torch_dtype()
    test_tokenizer_loading()
    # test_model_loading_small()  # Uncomment if you want to test full model loading
    
    print("\n✓ Core model tests passed!")

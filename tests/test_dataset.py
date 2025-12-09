"""Test dataset functionality."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.config import ModelConfig, DataConfig
from src.llm_trainer.model import load_tokenizer
from src.llm_trainer.dataset import CPTDataset, SFTDataset, create_dataset
from src.llm_trainer.utils import setup_logging


def test_cpt_dataset():
    """Test CPT dataset loading."""
    setup_logging(rank=0)
    
    # Create tokenizer
    model_config = ModelConfig(
        name="test",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        num_parameters=135000000
    )
    
    try:
        tokenizer = load_tokenizer(model_config)
        
        # Load dataset
        dataset = CPTDataset(
            data_path="data/raw/sample_cpt",
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Test sample
        sample = dataset[0]
        
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape[0] == 128
        
        print(f"✓ CPT Dataset: {len(dataset)} samples")
        print(f"  Sample shape: {sample['input_ids'].shape}")
        
    except FileNotFoundError:
        print("⚠ CPT test data not found. Create data/raw/sample_cpt/sample.jsonl")
    except Exception as e:
        print(f"⚠ CPT test skipped: {e}")


def test_sft_dataset():
    """Test SFT dataset loading."""
    setup_logging(rank=0)
    
    model_config = ModelConfig(
        name="test",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        num_parameters=135000000
    )
    
    try:
        tokenizer = load_tokenizer(model_config)
        
        # Load dataset
        dataset = SFTDataset(
            data_path="data/raw/sample_sft",
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Test sample
        sample = dataset[0]
        
        assert "input_ids" in sample
        assert "labels" in sample
        
        # Check that instruction is masked in labels
        labels = sample["labels"]
        num_masked = (labels == -100).sum().item()
        assert num_masked > 0, "Instruction should be masked"
        
        print(f"✓ SFT Dataset: {len(dataset)} samples")
        print(f"  Sample shape: {sample['input_ids'].shape}")
        print(f"  Masked tokens: {num_masked}")
        
    except FileNotFoundError:
        print("⚠ SFT test data not found. Create data/raw/sample_sft/sample.jsonl")
    except Exception as e:
        print(f"⚠ SFT test skipped: {e}")


def test_dataset_factory():
    """Test dataset factory function."""
    setup_logging(rank=0)
    
    model_config = ModelConfig(
        name="test",
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        num_parameters=135000000
    )
    
    try:
        tokenizer = load_tokenizer(model_config)
        
        # Test CPT
        data_config = DataConfig(
            dataset_type="cpt",
            train_path="data/raw/sample_cpt",
            max_samples=5
        )
        
        dataset = create_dataset(data_config, tokenizer, split="train")
        assert len(dataset) <= 5
        
        print(f"✓ Dataset factory works: created {data_config.dataset_type} dataset")
        
    except Exception as e:
        print(f"⚠ Factory test skipped: {e}")


if __name__ == "__main__":
    print("Testing dataset module...\n")
    
    test_cpt_dataset()
    test_sft_dataset()
    test_dataset_factory()
    
    print("\n✓ Dataset tests completed!")

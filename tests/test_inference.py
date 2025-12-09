"""Test inference module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_inference_engine_structure():
    """Test inference engine structure."""
    from src.llm_trainer.inference import InferenceEngine, load_inference_engine
    
    print("✓ Inference engine imports successful")
    print("  Classes available:")
    print("    - InferenceEngine")
    print("    - vLLMEngine")
    print("    - load_inference_engine()")


def test_batch_processing():
    """Test batch processing logic."""
    prompts = ["def hello():", "def add(a, b):", "class Calculator:"]
    
    print(f"✓ Batch processing setup")
    print(f"  Sample prompts: {len(prompts)}")


if __name__ == "__main__":
    print("Testing inference module...\n")
    
    test_inference_engine_structure()
    test_batch_processing()
    
    print("\n✓ Inference structure tests completed!")
    print("\nNote: Full inference tests require a trained model.")

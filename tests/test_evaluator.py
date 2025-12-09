"""Test evaluation module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.evaluator import CodeExecutor


def test_code_executor():
    """Test code execution."""
    executor = CodeExecutor(timeout=5)
    
    # Test successful execution
    code = "def add(a, b):\n    return a + b"
    test = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    
    success, error = executor.execute_code(code, test)
    
    print(f"✓ Code executor test:")
    print(f"  Success: {success}")
    if not success:
        print(f"  Error: {error}")
    
    # Test failed execution
    bad_code = "def add(a, b):\n    return a - b"
    success_bad, error_bad = executor.execute_code(bad_code, test)
    
    print(f"✓ Code executor (expected failure):")
    print(f"  Success: {success_bad}")


if __name__ == "__main__":
    print("Testing evaluator module...\n")
    
    test_code_executor()
    
    print("\n✓ Evaluator tests completed!")

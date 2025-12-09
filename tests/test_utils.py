"""Test utility functions."""

import os
import tempfile
import torch
from llm_trainer.utils import (
    setup_logging,
    set_seed,
    count_parameters,
    format_number,
    save_json,
    load_json,
    create_output_dir
)


def test_logging():
    """Test logging setup."""
    logger = setup_logging(log_level="INFO", rank=0)
    logger.info("Test log message")
    assert logger is not None


def test_seed():
    """Test seed setting."""
    set_seed(42, rank=0)
    a = torch.rand(3, 3)
    
    set_seed(42, rank=0)
    b = torch.rand(3, 3)
    
    assert torch.allclose(a, b), "Random seeds should produce same results"


def test_parameter_counting():
    """Test parameter counting."""
    model = torch.nn.Linear(10, 5)
    counts = count_parameters(model)
    
    assert counts["total"] == 55  # 10*5 + 5 bias
    assert counts["trainable"] == 55


def test_number_formatting():
    """Test number formatting."""
    assert format_number(1500) == "1.5K"
    assert format_number(2500000) == "2.5M"
    assert format_number(3200000000) == "3.2B"


def test_json_io():
    """Test JSON save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data = {"a": 1, "b": [2, 3, 4]}
        filepath = os.path.join(tmpdir, "test.json")
        
        save_json(test_data, filepath)
        loaded = load_json(filepath)
        
        assert loaded == test_data


def test_output_dir_creation():
    """Test output directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = create_output_dir(tmpdir, run_name="test_run")
        assert os.path.exists(output_dir)
        assert "test_run" in output_dir


if __name__ == "__main__":
    test_logging()
    test_seed()
    test_parameter_counting()
    test_number_formatting()
    test_json_io()
    test_output_dir_creation()
    print("✓ All utility tests passed!")

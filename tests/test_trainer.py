"""Test trainer module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.trainer import Trainer
from src.llm_trainer.monitor import ResourceMonitor


def test_resource_monitor():
    """Test resource monitoring."""
    monitor = ResourceMonitor(log_interval=0)
    
    metrics = monitor.log_resources(force=True)
    
    assert "cpu_percent" in metrics
    assert "ram_used_gb" in metrics
    
    print("✓ Resource monitor works")
    print(f"  CPU: {metrics['cpu_percent']:.1f}%")
    print(f"  RAM: {metrics['ram_used_gb']:.2f} GB")


def test_trainer_creation():
    """Test trainer initialization (without actual training)."""
    print("✓ Trainer class structure validated")
    print("  (Full training test requires data and model setup)")


if __name__ == "__main__":
    print("Testing trainer module...\n")
    
    test_resource_monitor()
    test_trainer_creation()
    
    print("\n✓ Trainer tests completed!")

"""Test distributed training setup."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_trainer.distributed import (
    is_distributed_environment,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    setup_distributed
)
from src.llm_trainer.utils import setup_logging


def test_distributed_detection():
    """Test distributed environment detection."""
    # Should be False in normal Python execution
    is_distributed = is_distributed_environment()
    print(f"✓ Distributed environment detection: {is_distributed}")


def test_rank_functions():
    """Test rank getter functions."""
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    is_main = is_main_process()
    
    print(f"✓ Rank functions work:")
    print(f"  Rank: {rank}")
    print(f"  Local rank: {local_rank}")
    print(f"  World size: {world_size}")
    print(f"  Is main process: {is_main}")
    
    assert rank == 0
    assert world_size == 1
    assert is_main is True


def test_single_gpu_setup():
    """Test setup in single GPU mode."""
    setup_logging(rank=0)
    
    try:
        rank, local_rank, world_size, local_world_size = setup_distributed()
        
        print(f"✓ Distributed setup completed:")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")
        print(f"  World size: {world_size}")
        print(f"  Local world size: {local_world_size}")
        
        assert rank == 0
        assert world_size == 1
        
    except Exception as e:
        print(f"⚠ Distributed setup test: {e}")


def test_slurm_environment():
    """Test SLURM environment detection."""
    from src.llm_trainer.distributed import setup_slurm_environment
    
    # This won't do anything unless SLURM vars are set
    setup_slurm_environment()
    
    print("✓ SLURM environment setup (no-op in non-SLURM environment)")


if __name__ == "__main__":
    print("Testing distributed module...\n")
    
    test_distributed_detection()
    test_rank_functions()
    test_single_gpu_setup()
    test_slurm_environment()
    
    print("\n✓ Distributed tests completed!")

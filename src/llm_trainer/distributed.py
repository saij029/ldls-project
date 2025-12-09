"""Distributed training setup for multi-node/multi-GPU training."""

import os
import socket
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from .utils import get_logger, set_seed


def setup_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30
) -> Tuple[int, int, int, int]:
    """Setup distributed training environment.
    
    Supports multiple launchers:
    - torchrun/torch.distributed.launch
    - DeepSpeed launcher
    - SLURM
    - Single GPU (no distribution)
    
    Args:
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        timeout_minutes: Timeout for distributed operations
        
    Returns:
        Tuple of (rank, local_rank, world_size, local_world_size)
    """
    logger = get_logger()
    
    # Check if already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = torch.cuda.device_count()
        
        logger.info(f"Distributed already initialized: rank={rank}, world_size={world_size}")
        return rank, local_rank, world_size, local_world_size
    
    # Detect distributed environment
    if is_distributed_environment():
        # Get environment variables
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        
        # Set LOCAL_RANK if not set (needed by some libraries)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Initialize process group
        timeout = torch.distributed.timedelta(minutes=timeout_minutes)
        
        # Get master address and port
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        
        # Set environment variables if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = master_port
        
        logger.info(f"Initializing distributed: rank={rank}, local_rank={local_rank}, "
                   f"world_size={world_size}")
        logger.info(f"Master: {master_addr}:{master_port}")
        
        # Initialize
        dist.init_process_group(
            backend=backend,
            init_method=f"env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            local_world_size = torch.cuda.device_count()
        else:
            local_world_size = 1
        
        logger.info(f"Distributed initialized successfully on {socket.gethostname()}")
        
    else:
        # Single GPU/CPU mode
        rank = 0
        local_rank = 0
        world_size = 1
        local_world_size = 1
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info("Running in single GPU mode")
        else:
            logger.info("Running in CPU mode")
    
    return rank, local_rank, world_size, local_world_size


def is_distributed_environment() -> bool:
    """Check if running in distributed environment.
    
    Returns:
        True if distributed environment detected
    """
    # Check for torchrun/torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    
    # Check for SLURM
    if "SLURM_PROCID" in os.environ:
        return True
    
    # Check for DeepSpeed
    if "DEEPSPEED_WORLD_RANK" in os.environ:
        return True
    
    return False


def setup_for_distributed(rank: int) -> None:
    """Disable printing for non-master processes.
    
    Args:
        rank: Process rank
    """
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if rank == 0 or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger = get_logger()
        logger.info("Distributed training cleaned up")


def get_rank() -> int:
    """Get current process rank.
    
    Returns:
        Process rank (0 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size.
    
    Returns:
        World size (1 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank (GPU index on current node).
    
    Returns:
        Local rank
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is main process (rank 0).
    
    Returns:
        True if main process
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronization barrier for all processes."""
    if dist.is_initialized():
        dist.barrier()


def setup_slurm_environment() -> None:
    """Setup environment variables for SLURM.
    
    This function configures MASTER_ADDR and MASTER_PORT for SLURM jobs.
    """
    if "SLURM_PROCID" not in os.environ:
        return
    
    # Get SLURM info
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    
    # Set RANK and LOCAL_RANK
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # Get master node
    if "MASTER_ADDR" not in os.environ:
        # Get first node in nodelist
        nodelist = os.environ.get("SLURM_NODELIST", "")
        if nodelist:
            # Parse nodelist (e.g., "node[01-04]" -> "node01")
            master_node = nodelist.split(",")[0].split("[")[0]
            if "[" in nodelist:
                master_node += nodelist.split("[")[1].split("-")[0].split("]")[0]
            os.environ["MASTER_ADDR"] = master_node
        else:
            os.environ["MASTER_ADDR"] = "localhost"
    
    # Set master port
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    logger = get_logger()
    logger.info(f"SLURM environment configured: MASTER_ADDR={os.environ['MASTER_ADDR']}, "
               f"MASTER_PORT={os.environ['MASTER_PORT']}")


def print_distributed_info() -> None:
    """Print distributed training information."""
    logger = get_logger()
    
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Distributed Training Configuration")
        logger.info("=" * 80)
        logger.info(f"Hostname: {socket.gethostname()}")
        logger.info(f"World Size: {world_size}")
        logger.info(f"Backend: {dist.get_backend() if dist.is_initialized() else 'N/A'}")
        
        if torch.cuda.is_available():
            logger.info(f"GPUs per node: {torch.cuda.device_count()}")
            logger.info(f"Total GPUs: {world_size}")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(local_rank)}")
            
            # GPU memory
            total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1024**3
            logger.info(f"GPU Memory: {total_mem:.2f} GB")
        
        logger.info("=" * 80)
    
    barrier()


def setup_deepspeed_distributed() -> Tuple[int, int, int]:
    """Setup distributed training for DeepSpeed.
    
    DeepSpeed handles initialization internally, but we can extract info.
    
    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    # DeepSpeed sets these automatically
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger = get_logger()
    logger.info(f"DeepSpeed distributed: rank={rank}, local_rank={local_rank}, "
               f"world_size={world_size}")
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all processes.
    
    Args:
        tensor: Tensor to average
        
    Returns:
        Averaged tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    
    return tensor


def gather_tensor(tensor: torch.Tensor) -> Optional[list]:
    """Gather tensor from all processes to rank 0.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all ranks (only on rank 0, None on others)
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    if get_rank() == 0:
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list, dst=0)
        return gather_list
    else:
        dist.gather(tensor, dst=0)
        return None


def initialize_training_environment(
    seed: int = 42,
    backend: str = "nccl"
) -> Tuple[int, int, int, int]:
    """Initialize complete training environment.
    
    Sets up distributed training, CUDA, and seeds.
    
    Args:
        seed: Random seed
        backend: Distributed backend
        
    Returns:
        Tuple of (rank, local_rank, world_size, local_world_size)
    """
    # Setup SLURM if needed
    setup_slurm_environment()
    
    # Setup distributed
    rank, local_rank, world_size, local_world_size = setup_distributed(backend=backend)
    
    # Set seed
    set_seed(seed, rank=rank)
    
    # Print info
    if rank == 0:
        print_distributed_info()
    
    return rank, local_rank, world_size, local_world_size

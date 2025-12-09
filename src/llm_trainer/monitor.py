"""Resource monitoring utilities."""

import time
import psutil
import torch
from typing import Dict, Any

from .utils import get_logger


class ResourceMonitor:
    """Monitor CPU, GPU, and memory usage during training."""
    
    def __init__(self, log_interval: int = 60):
        """Initialize resource monitor.
        
        Args:
            log_interval: Logging interval in seconds
        """
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.logger = get_logger()
    
    def log_resources(self, force: bool = False) -> Dict[str, Any]:
        """Log current resource usage.
        
        Args:
            force: Force logging even if interval not reached
            
        Returns:
            Dictionary of resource metrics
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_log_time) < self.log_interval:
            return {}
        
        self.last_log_time = current_time
        
        metrics = {}
        
        # CPU usage
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["cpu_count"] = psutil.cpu_count()
        
        # RAM usage
        ram = psutil.virtual_memory()
        metrics["ram_used_gb"] = ram.used / 1024**3
        metrics["ram_total_gb"] = ram.total / 1024**3
        metrics["ram_percent"] = ram.percent
        
        # GPU usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                metrics[f"gpu_{i}_allocated_gb"] = allocated
                metrics[f"gpu_{i}_reserved_gb"] = reserved
        
        # Log
        self.logger.info(f"Resources: CPU={metrics['cpu_percent']:.1f}% | "
                        f"RAM={metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}GB")
        
        return metrics

import os
import resource

import torch


def get_system_metrics():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_scale = 1024.0
    if os.uname().sysname == "Darwin":
        rss_scale = 1024.0 * 1024.0
    metrics = {
        "cpu_load_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0,
        "cpu_rss_gb": usage.ru_maxrss / rss_scale / 1024.0 / 1024.0,
        "gpu_memory_allocated_gb": 0.0,
        "gpu_memory_reserved_gb": 0.0,
        "gpu_memory_peak_gb": 0.0,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024.0 ** 3)
        metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024.0 ** 3)
        metrics["gpu_memory_peak_gb"] = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
    return metrics

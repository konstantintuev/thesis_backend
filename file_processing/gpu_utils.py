import logging
import torch

logger = logging.getLogger(__name__)

number_of_models_in_parallel = 2

# Around 4 GB for batch size 8 on GTX 1080
#  -> 1 batch per 0.5 GB free, safe is 1 batch per 1 GB free
def get_batch_size():
    if torch.cuda.is_available():
        # One, first GPU
        gpu_index = 0

        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu_index)
        allocated_memory = torch.cuda.memory_allocated(gpu_index)
        free_memory = reserved_memory - allocated_memory
        total_memory_in_gb = total_memory / (1024 ** 3)
        free_memory_in_gb = free_memory / (1024 ** 3)

        logger.info(f"Total VRAM: {total_memory_in_gb:.2f} GB")
        logger.info(f"Free VRAM: {free_memory_in_gb:.2f} GB")

        return max(round(total_memory_in_gb/2) - 1, 1) # Max - 1 batch of slack
    else:
        logger.info("CUDA is not available. No GPU detected.")
        # Good baseline
        return 3
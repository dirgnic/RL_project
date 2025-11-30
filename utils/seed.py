"""
Seed management for reproducibility.
"""
import random
import numpy as np


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch seeds (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"PyTorch seed set to {seed}")
    except ImportError:
        pass
    
    print(f"Random seeds set to {seed}")

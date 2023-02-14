import numpy
from copy import deepcopy
import torch

def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, torch.nn.Module)
    return torch.nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).int() # upper triangle matrix where lower triangle are zeros, k determines where triangle starts
    
    return mask == 0
"""
Model pruning for mobile deployment
"""
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

from app.core.config import settings


class ChannelPruning:
    """
    Prune model channels for mobile deployment
    """
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def apply_pruning(self):
        """Apply magnitude-based pruning to model"""
        print(f"Pruning {self.pruning_ratio * 100}% of model weights...")
        
        # Collect all weights
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global pruning
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=self.pruning_ratio
        )
        
        print("✓ Pruning applied")
        return self.model
    
    def remove_masks(self):
        """Remove pruning masks permanently"""
        for module in self.model.modules():
            if hasattr(module, 'weight_mask'):
                torch.nn.utils.prune.remove(module, 'weight')


def prune_model(model: nn.Module, ratio: float = 0.3) -> nn.Module:
    """
    Prune model for mobile deployment
    
    Args:
        model: PyTorch model
        ratio: Pruning ratio (0.0 to 1.0)
    
    Returns:
        Pruned model
    """
    pruner = ChannelPruning(model, ratio)
    model = pruner.apply_pruning()
    return model


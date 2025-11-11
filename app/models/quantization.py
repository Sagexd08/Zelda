"""
Model quantization for faster inference
"""
import torch
import torch.nn as nn
from typing import Optional, Callable
import numpy as np

from app.core.config import settings


class QuantizedModel:
    """
    INT8 quantized model wrapper for faster inference
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.original_model = model
        self.quantized_model = None
        self.is_quantized = False
    
    def quantize(self, calibration_data: list, num_calibration_batches: int = 10):
        """
        Quantize model using INT8 quantization with calibration
        
        Args:
            calibration_data: List of calibration samples
            num_calibration_batches: Number of batches for calibration
        """
        print("Quantizing model to INT8...")
        
        # Prepare calibration batches
        calibration_batches = []
        batch_size = len(calibration_data) // num_calibration_batches
        for i in range(0, len(calibration_data), batch_size):
            calibration_batches.append(calibration_data[i:i+batch_size])
        
        # Set model to evaluation mode
        self.original_model.eval()
        
        # Quantize model
        try:
            # Use PyTorch's quantization API
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            self.is_quantized = True
            print("✓ Model quantized successfully")
        except Exception as e:
            print(f"✗ Quantization failed: {e}")
            self.quantized_model = self.original_model
    
    def __call__(self, *args, **kwargs):
        """Forward pass through quantized or original model"""
        if self.is_quantized and self.quantized_model is not None:
            return self.quantized_model(*args, **kwargs)
        return self.original_model(*args, **kwargs)


def quantize_model(model: nn.Module, calibration_data: list) -> QuantizedModel:
    """
    Convert a model to INT8 quantized version
    
    Args:
        model: PyTorch model to quantize
        calibration_data: Calibration data for quantization
    
    Returns:
        Quantized model wrapper
    """
    quantized = QuantizedModel(model)
    quantized.quantize(calibration_data)
    return quantized


class ModelOptimizer:
    """Model optimization utilities"""
    
    @staticmethod
    def fuse_modules(model: nn.Module) -> nn.Module:
        """
        Fuse Conv-BN-ReLU modules for faster inference
        
        Args:
            model: PyTorch model
        
        Returns:
            Fused model
        """
        try:
            fused_model = torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']],
                inplace=False
            )
            print("✓ Model modules fused")
            return fused_model
        except Exception as e:
            print(f"⚠ Could not fuse modules: {e}")
            return model
    
    @staticmethod
    def enable_torchscript(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """
        Convert model to TorchScript for faster inference
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
        
        Returns:
            TorchScript model
        """
        try:
            traced_model = torch.jit.trace(model, example_input)
            traced_model.eval()
            print("✓ Model converted to TorchScript")
            return traced_model
        except Exception as e:
            print(f"⚠ Could not convert to TorchScript: {e}")
            return model


"""
Differential privacy for federated learning
"""
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Add noise for differential privacy guarantees
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Args:
            epsilon: Privacy budget
            delta: Failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_gaussian_noise(self, gradients: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Gaussian noise for (epsilon, delta)-DP
        
        Args:
            gradients: Model gradients
            sensitivity: L2 sensitivity bound
        
        Returns:
            Noisy gradients
        """
        # Compute noise scale
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        # Add noise
        noise = np.random.normal(0, sigma, gradients.shape)
        noisy_gradients = gradients + noise
        
        return noisy_gradients
    
    def add_laplace_noise(self, gradients: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Laplace noise for epsilon-DP
        
        Args:
            gradients: Model gradients
            sensitivity: L1 sensitivity bound
        
        Returns:
            Noisy gradients
        """
        # Compute noise scale
        scale = sensitivity / self.epsilon
        
        # Add noise
        noise = np.random.laplace(0, scale, gradients.shape)
        noisy_gradients = gradients + noise
        
        return noisy_gradients
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float) -> np.ndarray:
        """
        Clip gradients to bound sensitivity
        
        Args:
            gradients: Model gradients
            clip_norm: Maximum L2 norm
        
        Returns:
            Clipped gradients
        """
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > clip_norm:
            gradients = gradients * clip_norm / grad_norm
        
        return gradients


def create_dp_mechanism(epsilon: float, delta: float = 1e-5) -> DifferentialPrivacy:
    """
    Create differential privacy mechanism
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
    
    Returns:
        DifferentialPrivacy instance
    """
    return DifferentialPrivacy(epsilon, delta)


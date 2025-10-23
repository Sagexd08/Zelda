"""
Fusion Model Module
Learned weighted fusion of multiple embeddings using MLP
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings


class FusionMLP(nn.Module):
    """
    Multi-Layer Perceptron for embedding fusion.
    Takes concatenated embeddings from multiple models and outputs weighted fusion.
    """
    
    def __init__(self, input_dim: int = 1536, embedding_dim: int = 512):
        """
        Initialize fusion MLP.
        
        Args:
            input_dim: Input dimension (3 x 512 for 3 models)
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, embedding_dim)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Concatenated embeddings (B, 1536)
            
        Returns:
            torch.Tensor: Fused embeddings (B, 512)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x


class FusionModel:
    """
    Fusion model for combining multiple face embeddings.
    Can use simple averaging or learned MLP fusion.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize fusion model.
        
        Args:
            device: Device for inference
        """
        self.device = device or settings.get_device()
        self.mlp_model = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize fusion MLP if weights available"""
        fusion_path = settings.get_model_path(settings.FUSION_MODEL)
        
        if fusion_path.exists():
            try:
                self.mlp_model = FusionMLP()
                state_dict = torch.load(fusion_path, map_location=self.device)
                self.mlp_model.load_state_dict(state_dict)
                self.mlp_model = self.mlp_model.to(self.device)
                self.mlp_model.eval()
                print("✓ Fusion MLP model loaded")
            except Exception as e:
                print(f"✗ Failed to load Fusion MLP: {e}")
                self.mlp_model = None
        else:
            print("Fusion MLP weights not found, using simple averaging")
    
    @torch.no_grad()
    def fuse_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray],
        use_mlp: bool = True
    ) -> np.ndarray:
        """
        Fuse multiple embeddings into single representation.
        
        Args:
            embeddings: Dictionary of embeddings from different models
            use_mlp: Use learned MLP fusion if available
            
        Returns:
            np.ndarray: Fused embedding (512-D)
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")
        
        # If only one embedding, return it
        if len(embeddings) == 1:
            return list(embeddings.values())[0]
        
        # Try MLP fusion if available and requested
        if use_mlp and self.mlp_model is not None and len(embeddings) == 3:
            try:
                return self._mlp_fusion(embeddings)
            except Exception as e:
                print(f"MLP fusion failed: {e}, falling back to averaging")
        
        # Fallback to simple averaging
        return self._average_fusion(embeddings)
    
    def _mlp_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Perform learned fusion using MLP.
        
        Args:
            embeddings: Dictionary with 'arcface', 'facenet', 'mobilefacenet'
            
        Returns:
            np.ndarray: Fused embedding
        """
        # Concatenate embeddings in fixed order
        concat_embedding = np.concatenate([
            embeddings.get('arcface', np.zeros(512)),
            embeddings.get('facenet', np.zeros(512)),
            embeddings.get('mobilefacenet', np.zeros(512))
        ])
        
        # Convert to tensor
        tensor = torch.from_numpy(concat_embedding).float().unsqueeze(0).to(self.device)
        
        # Forward pass
        fused = self.mlp_model(tensor)
        fused = fused.cpu().numpy().flatten()
        
        return fused
    
    def _average_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple averaging fusion.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            np.ndarray: Averaged embedding
        """
        embedding_list = list(embeddings.values())
        fused = np.mean(embedding_list, axis=0)
        
        # L2 normalization
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        
        return fused
    
    def weighted_fusion(
        self, 
        embeddings: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Weighted average fusion with custom weights.
        
        Args:
            embeddings: Dictionary of embeddings
            weights: Dictionary of weights for each model
            
        Returns:
            np.ndarray: Weighted fused embedding
        """
        if weights is None:
            # Default weights (can be tuned based on validation performance)
            weights = {
                'arcface': 0.4,
                'facenet': 0.35,
                'mobilefacenet': 0.25
            }
        
        # Normalize weights
        total_weight = sum(weights.get(k, 1.0) for k in embeddings.keys())
        
        # Weighted sum
        fused = np.zeros(512, dtype=np.float32)
        for model_name, embedding in embeddings.items():
            weight = weights.get(model_name, 1.0) / total_weight
            fused += weight * embedding
        
        # L2 normalization
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        
        return fused


# Singleton instance
_fusion_instance: Optional[FusionModel] = None


def get_fusion_model() -> FusionModel:
    """
    Get singleton fusion model instance.
    
    Returns:
        FusionModel: Fusion model
    """
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = FusionModel()
    return _fusion_instance


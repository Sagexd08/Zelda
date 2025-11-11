"""
Attention-based multi-model fusion for face recognition
"""
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings


class AttentionFusionModel(nn.Module):
    """
    Attention-based fusion for combining embeddings from multiple models.
    
    Uses self-attention mechanism to learn optimal weights for each embedding
    model dynamically based on input characteristics.
    """
    
    def __init__(self, embedding_dim: int = 512, num_heads: int = 4):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Multi-head self-attention
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, num_models, embedding_dim)
        
        Returns:
            fused_embedding: (batch_size, embedding_dim)
        """
        batch_size, num_models, emb_dim = embeddings.shape
        
        # Apply layer normalization
        x = self.layer_norm1(embeddings)
        
        # Multi-head attention
        Q = self.query_proj(x).view(batch_size, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x).view(batch_size, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x).view(batch_size, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_models, emb_dim)
        
        # Residual connection
        x = embeddings + attn_output
        
        # Feed-forward with residual
        x = self.layer_norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output
        
        # Pool across models (average)
        x = torch.mean(x, dim=1)
        
        # Final projection
        x = self.output_proj(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x


class AttentionFusion:
    """
    Wrapper class for attention-based fusion model.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None
        self.embedding_dim = 512
        self.num_models = 3  # ArcFace, FaceNet, MobileFaceNet
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the attention fusion model"""
        try:
            self.model = AttentionFusionModel(embedding_dim=self.embedding_dim).to(self.device)
            self.model.eval()
            print("✓ Attention fusion model initialized")
        except Exception as e:
            print(f"⚠ Could not initialize attention fusion model: {e}")
    
    @torch.no_grad()
    def fuse_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        use_attention: bool = True
    ) -> np.ndarray:
        """
        Fuse embeddings using attention mechanism.
        
        Args:
            embeddings: Dictionary of model_name -> embedding array
            use_attention: Whether to use attention or simple average
        
        Returns:
            Fused embedding array
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")
        
        if len(embeddings) == 1:
            return list(embeddings.values())[0]
        
        if use_attention and self.model is not None:
            try:
                return self._attention_fusion(embeddings)
            except Exception as e:
                print(f"Attention fusion failed: {e}, falling back to averaging")
        
        # Fallback to average
        return self._average_fusion(embeddings)
    
    def _attention_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply attention-based fusion"""
        # Convert to tensor
        emb_list = [
            embeddings.get('arcface', np.zeros(self.embedding_dim)),
            embeddings.get('facenet', np.zeros(self.embedding_dim)),
            embeddings.get('mobilefacenet', np.zeros(self.embedding_dim))
        ]
        
        emb_tensor = torch.from_numpy(np.array(emb_list)).float().unsqueeze(0).to(self.device)
        
        # Apply attention fusion
        fused = self.model(emb_tensor)
        fused = fused.cpu().numpy().flatten()
        
        return fused
    
    def _average_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average fusion"""
        embedding_list = list(embeddings.values())
        fused = np.mean(embedding_list, axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✓ Loaded attention fusion weights from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")


_attention_fusion_instance: Optional[AttentionFusion] = None


def get_attention_fusion() -> AttentionFusion:
    """Get singleton instance of AttentionFusion"""
    global _attention_fusion_instance
    if _attention_fusion_instance is None:
        _attention_fusion_instance = AttentionFusion()
    return _attention_fusion_instance


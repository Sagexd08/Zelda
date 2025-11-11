"""
Vision Transformer (ViT) for face recognition
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings


class PatchEmbedding(nn.Module):
    """Split image into patches and embed"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        # (batch_size, embed_dim, num_patches_sqrt, num_patches_sqrt)
        x = self.projection(x)
        # (batch_size, embed_dim, num_patches)
        x = x.flatten(2)
        # (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        Q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)
        
        output = self.output(attn_output)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for face recognition
    
    Architecture based on "An Image is Worth 16x16 Words"
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        num_classes: int = 512  # Embedding dimension
    ):
        super().__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Head for embedding extraction
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract cls token
        x = x[:, 0]  # (B, embed_dim)
        
        # Project to embedding
        x = self.head(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ViTFaceRecognizer:
    """
    Vision Transformer-based face recognition
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None
        self.input_size = 224
        
        self._init_model()
    
    def _init_model(self):
        """Initialize ViT model"""
        try:
            self.model = VisionTransformer(
                img_size=self.input_size,
                embed_dim=768,
                depth=12,
                num_heads=12
            ).to(self.device)
            self.model.eval()
            print("✓ Vision Transformer initialized")
        except Exception as e:
            print(f"⚠ Could not initialize ViT: {e}")
    
    @torch.no_grad()
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using Vision Transformer
        
        Args:
            face_image: (H, W, C) face image array
        
        Returns:
            embedding: (512,) face embedding
        """
        if self.model is None:
            raise RuntimeError("ViT model not initialized")
        
        # Preprocess
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = face_image[:, :, ::-1]  # BGR to RGB
        
        # Transform and add batch dimension
        tensor = transform(face_image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        embedding = self.model(tensor)
        embedding = embedding.cpu().numpy().flatten()
        
        return embedding
    
    def load_pretrained(self, model_path: str):
        """Load pretrained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✓ Loaded ViT weights from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")


_vit_instance: Optional[ViTFaceRecognizer] = None


def get_vit_recognizer() -> ViTFaceRecognizer:
    """Get singleton instance of ViT recognizer"""
    global _vit_instance
    if _vit_instance is None:
        _vit_instance = ViTFaceRecognizer()
    return _vit_instance


"""
Embedding Extraction Module
Multiple backbone models: ArcFace, FaceNet, MobileFaceNet
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: facenet-pytorch not available")

from app.core.config import settings


class ArcFaceResNet(nn.Module):
    """
    ArcFace ResNet100 backbone for face recognition.
    Extracts 512-dimensional embeddings.
    """
    
    def __init__(self, pretrained_path: Optional[str] = None):
        """
        Initialize ArcFace model.
        
        Args:
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        
        # Use torchvision ResNet as base
        from torchvision.models import resnet101
        
        # Load base model
        base_model = resnet101(pretrained=False)
        
        # Modify for face recognition (512-D embeddings)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.bn = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, 512)
        self.bn_out = nn.BatchNorm1d(512)
        
        # Load pretrained weights if available
        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded ArcFace weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load ArcFace weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Embeddings (B, 512)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)
        x = self.bn_out(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet for efficient face recognition on mobile devices.
    Lightweight model with 512-D embeddings.
    """
    
    def __init__(self, pretrained_path: Optional[str] = None):
        """
        Initialize MobileFaceNet.
        
        Args:
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        
        # Use MobileNetV2 as base
        from torchvision.models import mobilenet_v2
        
        base_model = mobilenet_v2(pretrained=False)
        self.features = base_model.features
        
        # Custom classifier for 512-D embeddings
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, 512)
        self.bn = nn.BatchNorm1d(512)
        
        # Load pretrained weights if available
        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded MobileFaceNet weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load MobileFaceNet weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 160, 160)
            
        Returns:
            torch.Tensor: Embeddings (B, 512)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class MultiBackboneEmbeddingExtractor:
    """
    Multi-backbone embedding extractor combining ArcFace, FaceNet, and MobileFaceNet.
    Provides both individual and fused embeddings.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize multi-backbone extractor.
        
        Args:
            device: Device for inference
        """
        self.device = device or settings.get_device()
        
        # Initialize models
        self.arcface_model = None
        self.facenet_model = None
        self.mobilefacenet_model = None
        
        self._init_models()
        self._init_transforms()
    
    def _init_models(self):
        """Initialize all backbone models"""
        print(f"Initializing embedding models on device: {self.device}")
        
        # Initialize ArcFace
        try:
            arcface_path = settings.get_model_path(settings.ARCFACE_MODEL)
            self.arcface_model = ArcFaceResNet(arcface_path if arcface_path.exists() else None)
            self.arcface_model = self.arcface_model.to(self.device)
            self.arcface_model.eval()
            print("✓ ArcFace model loaded")
        except Exception as e:
            print(f"✗ Failed to load ArcFace: {e}")
        
        # Initialize FaceNet
        if FACENET_AVAILABLE:
            try:
                self.facenet_model = InceptionResnetV1(pretrained='vggface2').to(self.device)
                self.facenet_model.eval()
                print("✓ FaceNet model loaded")
            except Exception as e:
                print(f"✗ Failed to load FaceNet: {e}")
        
        # Initialize MobileFaceNet
        try:
            mobilefacenet_path = settings.get_model_path(settings.MOBILEFACENET_MODEL)
            self.mobilefacenet_model = MobileFaceNet(
                mobilefacenet_path if mobilefacenet_path.exists() else None
            )
            self.mobilefacenet_model = self.mobilefacenet_model.to(self.device)
            self.mobilefacenet_model.eval()
            print("✓ MobileFaceNet model loaded")
        except Exception as e:
            print(f"✗ Failed to load MobileFaceNet: {e}")
        
        # Verify at least one model is loaded
        if not any([self.arcface_model, self.facenet_model, self.mobilefacenet_model]):
            raise RuntimeError("No embedding model could be loaded")
    
    def _init_transforms(self):
        """Initialize image preprocessing transforms"""
        # Transform for 160x160 models (FaceNet, MobileFaceNet)
        self.transform_160 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Transform for 224x224 models (ArcFace)
        self.transform_224 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_arcface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace embedding.
        
        Args:
            face_image: Aligned face image (224x224, BGR)
            
        Returns:
            Optional[np.ndarray]: 512-D embedding or None
        """
        if self.arcface_model is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if rgb_image.shape[:2] != (224, 224):
                rgb_image = cv2.resize(rgb_image, (224, 224))
            
            # Convert to PIL and apply transforms
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_224(pil_image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            embedding = self.arcface_model(tensor)
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"ArcFace embedding extraction failed: {e}")
            return None
    
    @torch.no_grad()
    def extract_facenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract FaceNet embedding.
        
        Args:
            face_image: Aligned face image (160x160, BGR)
            
        Returns:
            Optional[np.ndarray]: 512-D embedding or None
        """
        if self.facenet_model is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if rgb_image.shape[:2] != (160, 160):
                rgb_image = cv2.resize(rgb_image, (160, 160))
            
            # Convert to PIL and apply transforms
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_160(pil_image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            embedding = self.facenet_model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"FaceNet embedding extraction failed: {e}")
            return None
    
    @torch.no_grad()
    def extract_mobilefacenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract MobileFaceNet embedding.
        
        Args:
            face_image: Aligned face image (160x160, BGR)
            
        Returns:
            Optional[np.ndarray]: 512-D embedding or None
        """
        if self.mobilefacenet_model is None:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if rgb_image.shape[:2] != (160, 160):
                rgb_image = cv2.resize(rgb_image, (160, 160))
            
            # Convert to PIL and apply transforms
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_160(pil_image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            embedding = self.mobilefacenet_model(tensor)
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"MobileFaceNet embedding extraction failed: {e}")
            return None
    
    def extract_all_embeddings(
        self, 
        face_image_160: np.ndarray,
        face_image_224: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all available models.
        
        Args:
            face_image_160: Aligned face (160x160, BGR)
            face_image_224: Aligned face (224x224, BGR) for ArcFace
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of embeddings
        """
        embeddings = {}
        
        # Extract from each model
        if self.facenet_model is not None:
            emb = self.extract_facenet_embedding(face_image_160)
            if emb is not None:
                embeddings['facenet'] = emb
        
        if self.mobilefacenet_model is not None:
            emb = self.extract_mobilefacenet_embedding(face_image_160)
            if emb is not None:
                embeddings['mobilefacenet'] = emb
        
        if self.arcface_model is not None:
            # Use 224 image if provided, otherwise resize 160
            face_224 = face_image_224 if face_image_224 is not None else face_image_160
            emb = self.extract_arcface_embedding(face_224)
            if emb is not None:
                embeddings['arcface'] = emb
        
        return embeddings
    
    def compute_simple_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple averaging fusion of multiple embeddings.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            np.ndarray: Fused embedding (512-D)
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")
        
        # Stack and average
        embedding_list = list(embeddings.values())
        fused = np.mean(embedding_list, axis=0)
        
        # L2 normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        
        return fused


# Singleton instance
_extractor_instance: Optional[MultiBackboneEmbeddingExtractor] = None


def get_embedding_extractor() -> MultiBackboneEmbeddingExtractor:
    """
    Get singleton embedding extractor instance.
    
    Returns:
        MultiBackboneEmbeddingExtractor: Embedding extractor
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MultiBackboneEmbeddingExtractor()
    return _extractor_instance


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Cosine similarity score
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        float: Euclidean distance
    """
    return float(np.linalg.norm(embedding1 - embedding2))


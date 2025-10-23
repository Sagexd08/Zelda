
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

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        from torchvision.models import resnet101

        base_model = resnet101(pretrained=False)

        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.bn = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, 512)
        self.bn_out = nn.BatchNorm1d(512)

        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded ArcFace weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load ArcFace weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)
        x = self.bn_out(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class MobileFaceNet(nn.Module):

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        from torchvision.models import mobilenet_v2

        base_model = mobilenet_v2(pretrained=False)
        self.features = base_model.features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, 512)
        self.bn = nn.BatchNorm1d(512)

        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded MobileFaceNet weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load MobileFaceNet weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class MultiBackboneEmbeddingExtractor:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()

        self.arcface_model = None
        self.facenet_model = None
        self.mobilefacenet_model = None

        self._init_models()
        self._init_transforms()

    def _init_models(self):
        print(f"Initializing embedding models on device: {self.device}")

        try:
            arcface_path = settings.get_model_path(settings.ARCFACE_MODEL)
            self.arcface_model = ArcFaceResNet(arcface_path if arcface_path.exists() else None)
            self.arcface_model = self.arcface_model.to(self.device)
            self.arcface_model.eval()
            print("✓ ArcFace model loaded")
        except Exception as e:
            print(f"✗ Failed to load ArcFace: {e}")

        if FACENET_AVAILABLE:
            try:
                self.facenet_model = InceptionResnetV1(pretrained='vggface2').to(self.device)
                self.facenet_model.eval()
                print("✓ FaceNet model loaded")
            except Exception as e:
                print(f"✗ Failed to load FaceNet: {e}")

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

        if not any([self.arcface_model, self.facenet_model, self.mobilefacenet_model]):
            raise RuntimeError("No embedding model could be loaded")

    def _init_transforms(self):
        self.transform_160 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.transform_224 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_arcface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        if self.arcface_model is None:
            return None

        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            if rgb_image.shape[:2] != (224, 224):
                rgb_image = cv2.resize(rgb_image, (224, 224))

            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_224(pil_image).unsqueeze(0).to(self.device)

            embedding = self.arcface_model(tensor)
            embedding = embedding.cpu().numpy().flatten()

            return embedding
        except Exception as e:
            print(f"ArcFace embedding extraction failed: {e}")
            return None

    @torch.no_grad()
    def extract_facenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        if self.facenet_model is None:
            return None

        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            if rgb_image.shape[:2] != (160, 160):
                rgb_image = cv2.resize(rgb_image, (160, 160))

            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_160(pil_image).unsqueeze(0).to(self.device)

            embedding = self.facenet_model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.cpu().numpy().flatten()

            return embedding
        except Exception as e:
            print(f"FaceNet embedding extraction failed: {e}")
            return None

    @torch.no_grad()
    def extract_mobilefacenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        if self.mobilefacenet_model is None:
            return None

        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            if rgb_image.shape[:2] != (160, 160):
                rgb_image = cv2.resize(rgb_image, (160, 160))

            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform_160(pil_image).unsqueeze(0).to(self.device)

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
        embeddings = {}

        if self.facenet_model is not None:
            emb = self.extract_facenet_embedding(face_image_160)
            if emb is not None:
                embeddings['facenet'] = emb

        if self.mobilefacenet_model is not None:
            emb = self.extract_mobilefacenet_embedding(face_image_160)
            if emb is not None:
                embeddings['mobilefacenet'] = emb

        if self.arcface_model is not None:
            face_224 = face_image_224 if face_image_224 is not None else face_image_160
            emb = self.extract_arcface_embedding(face_224)
            if emb is not None:
                embeddings['arcface'] = emb

        return embeddings

    def compute_simple_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("No embeddings to fuse")

        embedding_list = list(embeddings.values())
        fused = np.mean(embedding_list, axis=0)

        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

_extractor_instance: Optional[MultiBackboneEmbeddingExtractor] = None

def get_embedding_extractor() -> MultiBackboneEmbeddingExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MultiBackboneEmbeddingExtractor()
    return _extractor_instance

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return float(np.linalg.norm(embedding1 - embedding2))

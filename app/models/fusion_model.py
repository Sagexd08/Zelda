
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings

class FusionMLP(nn.Module):

    def __init__(self, input_dim: int = 1536, embedding_dim: int = 512):
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
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)

        x = F.normalize(x, p=2, dim=1)

        return x

class FusionModel:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.mlp_model = None

        self._init_model()

    def _init_model(self):
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
        if not embeddings:
            raise ValueError("No embeddings to fuse")

        if len(embeddings) == 1:
            return list(embeddings.values())[0]

        if use_mlp and self.mlp_model is not None and len(embeddings) == 3:
            try:
                return self._mlp_fusion(embeddings)
            except Exception as e:
                print(f"MLP fusion failed: {e}, falling back to averaging")

        return self._average_fusion(embeddings)

    def _mlp_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        concat_embedding = np.concatenate([
            embeddings.get('arcface', np.zeros(512)),
            embeddings.get('facenet', np.zeros(512)),
            embeddings.get('mobilefacenet', np.zeros(512))
        ])

        tensor = torch.from_numpy(concat_embedding).float().unsqueeze(0).to(self.device)

        fused = self.mlp_model(tensor)
        fused = fused.cpu().numpy().flatten()

        return fused

    def _average_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        embedding_list = list(embeddings.values())
        fused = np.mean(embedding_list, axis=0)

        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def weighted_fusion(
        self,
        embeddings: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        if weights is None:
            weights = {
                'arcface': 0.4,
                'facenet': 0.35,
                'mobilefacenet': 0.25
            }

        total_weight = sum(weights.get(k, 1.0) for k in embeddings.keys())

        fused = np.zeros(512, dtype=np.float32)
        for model_name, embedding in embeddings.items():
            weight = weights.get(model_name, 1.0) / total_weight
            fused += weight * embedding

        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

_fusion_instance: Optional[FusionModel] = None

def get_fusion_model() -> FusionModel:
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = FusionModel()
    return _fusion_instance

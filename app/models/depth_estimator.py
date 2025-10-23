
from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn

from app.core.config import settings

class MiDaSSmall(nn.Module):

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            print("✓ MiDaS depth estimator loaded from torch hub")
        except Exception as e:
            print(f"✗ Failed to load MiDaS: {e}")
            self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(x.shape[0], x.shape[2], x.shape[3])
        return self.model(x)

class DepthEstimator:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None
        self.transform = None

        self._init_model()

    def _init_model(self):
        if not settings.ENABLE_DEPTH_ESTIMATION:
            print("Depth estimation disabled in config")
            return

        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model = self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            print("✓ Depth estimator loaded")
        except Exception as e:
            print(f"✗ Failed to load depth estimator: {e}")
            self.model = None

    @torch.no_grad()
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        if self.model is None or self.transform is None:
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_batch = self.transform(rgb_image).to(self.device)

            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            return depth_map
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

    def analyze_depth_for_liveness(
        self,
        face_image: np.ndarray,
        face_bbox: np.ndarray
    ) -> Tuple[bool, float, dict]:
        depth_map = self.estimate_depth(face_image)

        x1, y1, x2, y2 = face_bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(face_image.shape[1], x2), min(face_image.shape[0], y2)

        face_depth = depth_map[y1:y2, x1:x2]

        depth_mean = np.mean(face_depth)
        depth_std = np.std(face_depth)
        depth_variance = np.var(face_depth)

        bg_depth = self._get_background_depth(depth_map, face_bbox)
        bg_mean = np.mean(bg_depth) if len(bg_depth) > 0 else depth_mean

        depth_contrast = abs(depth_mean - bg_mean)

        analysis = {
            'depth_mean': float(depth_mean),
            'depth_std': float(depth_std),
            'depth_variance': float(depth_variance),
            'depth_contrast': float(depth_contrast),
            'background_mean': float(bg_mean)
        }

        is_live = depth_variance > settings.DEPTH_THRESHOLD
        confidence = min(1.0, depth_variance / 0.3)

        return is_live, confidence, analysis

    def _get_background_depth(
        self,
        depth_map: np.ndarray,
        face_bbox: np.ndarray,
        margin: int = 20
    ) -> np.ndarray:
        x1, y1, x2, y2 = face_bbox.astype(int)

        x1_outer = max(0, x1 - margin)
        y1_outer = max(0, y1 - margin)
        x2_outer = min(depth_map.shape[1], x2 + margin)
        y2_outer = min(depth_map.shape[0], y2 + margin)

        outer_region = depth_map[y1_outer:y2_outer, x1_outer:x2_outer].copy()

        face_y1 = y1 - y1_outer
        face_y2 = y2 - y1_outer
        face_x1 = x1 - x1_outer
        face_x2 = x2 - x1_outer

        mask = np.ones_like(outer_region, dtype=bool)
        mask[face_y1:face_y2, face_x1:face_x2] = False

        bg_depth = outer_region[mask]

        return bg_depth

    def visualize_depth(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if depth_map is None:
            depth_map = self.estimate_depth(image)

        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO
        )

        blended = cv2.addWeighted(image, 0.5, depth_colored, 0.5, 0)

        return blended

_depth_estimator_instance: Optional[DepthEstimator] = None

def get_depth_estimator() -> DepthEstimator:
    global _depth_estimator_instance
    if _depth_estimator_instance is None:
        _depth_estimator_instance = DepthEstimator()
    return _depth_estimator_instance

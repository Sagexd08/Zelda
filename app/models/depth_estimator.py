"""
Depth Estimation Module
Monocular depth estimation for 3D face consistency validation
"""

from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn

from app.core.config import settings


class MiDaSSmall(nn.Module):
    """
    MiDaS Small model for monocular depth estimation.
    Used to detect 2D face spoofs (photos, screens) by analyzing depth map.
    """
    
    def __init__(self, pretrained_path: Optional[str] = None):
        """
        Initialize MiDaS model.
        
        Args:
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        
        try:
            # Try to load MiDaS from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            print("✓ MiDaS depth estimator loaded from torch hub")
        except Exception as e:
            print(f"✗ Failed to load MiDaS: {e}")
            self.model = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            torch.Tensor: Depth map (B, H, W)
        """
        if self.model is None:
            return torch.zeros(x.shape[0], x.shape[2], x.shape[3])
        return self.model(x)


class DepthEstimator:
    """
    Depth estimation for anti-spoofing.
    Validates 3D consistency of face to detect 2D photo/video attacks.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize depth estimator.
        
        Args:
            device: Device for inference
        """
        self.device = device or settings.get_device()
        self.model = None
        self.transform = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize depth estimation model"""
        if not settings.ENABLE_DEPTH_ESTIMATION:
            print("Depth estimation disabled in config")
            return
        
        try:
            # Load MiDaS
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            
            print("✓ Depth estimator loaded")
        except Exception as e:
            print(f"✗ Failed to load depth estimator: {e}")
            self.model = None
    
    @torch.no_grad()
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Depth map (normalized to 0-1)
        """
        if self.model is None or self.transform is None:
            # Return dummy depth map
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_batch = self.transform(rgb_image).to(self.device)
            
            # Predict depth
            prediction = self.model(input_batch)
            
            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            # Convert to numpy and normalize
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
        """
        Analyze depth map to determine if face is 3D (live) or 2D (spoof).
        
        Args:
            face_image: Full frame image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple[bool, float, dict]: (is_live, confidence, analysis_details)
        """
        # Estimate depth
        depth_map = self.estimate_depth(face_image)
        
        # Extract face region from depth map
        x1, y1, x2, y2 = face_bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(face_image.shape[1], x2), min(face_image.shape[0], y2)
        
        face_depth = depth_map[y1:y2, x1:x2]
        
        # Compute depth statistics
        depth_mean = np.mean(face_depth)
        depth_std = np.std(face_depth)
        depth_variance = np.var(face_depth)
        
        # Get background depth (regions around face)
        bg_depth = self._get_background_depth(depth_map, face_bbox)
        bg_mean = np.mean(bg_depth) if len(bg_depth) > 0 else depth_mean
        
        # Depth difference (face should be closer/different from background)
        depth_contrast = abs(depth_mean - bg_mean)
        
        # Analysis
        analysis = {
            'depth_mean': float(depth_mean),
            'depth_std': float(depth_std),
            'depth_variance': float(depth_variance),
            'depth_contrast': float(depth_contrast),
            'background_mean': float(bg_mean)
        }
        
        # Decision logic:
        # - Live faces have higher depth variance (3D surface)
        # - Spoofs (photos) have lower variance (flat surface)
        is_live = depth_variance > settings.DEPTH_THRESHOLD
        confidence = min(1.0, depth_variance / 0.3)  # Normalize confidence
        
        return is_live, confidence, analysis
    
    def _get_background_depth(
        self, 
        depth_map: np.ndarray, 
        face_bbox: np.ndarray,
        margin: int = 20
    ) -> np.ndarray:
        """
        Extract background depth around face region.
        
        Args:
            depth_map: Full depth map
            face_bbox: Face bounding box
            margin: Margin around face
            
        Returns:
            np.ndarray: Background depth values
        """
        x1, y1, x2, y2 = face_bbox.astype(int)
        
        # Expand bbox with margin
        x1_outer = max(0, x1 - margin)
        y1_outer = max(0, y1 - margin)
        x2_outer = min(depth_map.shape[1], x2 + margin)
        y2_outer = min(depth_map.shape[0], y2 + margin)
        
        # Extract outer region
        outer_region = depth_map[y1_outer:y2_outer, x1_outer:x2_outer].copy()
        
        # Mask out face region
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
        """
        Visualize depth map overlaid on image.
        
        Args:
            image: Original image
            depth_map: Depth map (computed if not provided)
            
        Returns:
            np.ndarray: Visualization image
        """
        if depth_map is None:
            depth_map = self.estimate_depth(image)
        
        # Convert depth to color map
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_INFERNO
        )
        
        # Blend with original image
        blended = cv2.addWeighted(image, 0.5, depth_colored, 0.5, 0)
        
        return blended


# Singleton instance
_depth_estimator_instance: Optional[DepthEstimator] = None


def get_depth_estimator() -> DepthEstimator:
    """
    Get singleton depth estimator instance.
    
    Returns:
        DepthEstimator: Depth estimator
    """
    global _depth_estimator_instance
    if _depth_estimator_instance is None:
        _depth_estimator_instance = DepthEstimator()
    return _depth_estimator_instance


"""
Liveness Detection Module
CNN-based anti-spoofing with Grad-CAM visualization
"""

from typing import Optional, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from app.core.config import settings


class LivenessResNet(nn.Module):
    """
    ResNet-18 based liveness detector.
    Binary classification: live (1) vs spoof (0)
    """
    
    def __init__(self, pretrained_path: Optional[str] = None):
        """
        Initialize liveness detector.
        
        Args:
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        
        # Load ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=False)
        
        # Modify classifier for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary: live, spoof
        )
        
        # Load pretrained weights if available
        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"Loaded liveness weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load liveness weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        return self.backbone(x)
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature maps for Grad-CAM visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Feature maps from last conv layer
        """
        # Get features from layer before final pooling
        for name, module in self.backbone.named_children():
            x = module(x)
            if name == 'avgpool':
                return x
        return x


class LivenessDetector:
    """
    Liveness detection system with multiple anti-spoofing techniques.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize liveness detector.
        
        Args:
            device: Device for inference
        """
        self.device = device or settings.get_device()
        self.model = None
        
        self._init_model()
        self._init_transform()
    
    def _init_model(self):
        """Initialize liveness detection model"""
        liveness_path = settings.get_model_path(settings.LIVENESS_MODEL)
        
        try:
            self.model = LivenessResNet(liveness_path if liveness_path.exists() else None)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ Liveness detector loaded")
        except Exception as e:
            print(f"✗ Failed to load liveness detector: {e}")
    
    def _init_transform(self):
        """Initialize image preprocessing"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if face is live or spoof.
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Tuple[bool, float]: (is_live, confidence_score)
        """
        if self.model is None:
            # If model not loaded, assume live (fail-open for testing)
            return True, 1.0
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL and apply transforms
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Get live probability
            live_prob = probabilities[0, 1].item()
            
            # Determine if live based on threshold
            is_live = live_prob >= settings.LIVENESS_THRESHOLD
            
            return is_live, live_prob
        except Exception as e:
            print(f"Liveness prediction failed: {e}")
            return True, 0.5
    
    def predict_with_gradcam(
        self, 
        face_image: np.ndarray
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Predict liveness with Grad-CAM visualization.
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Tuple[bool, float, np.ndarray]: (is_live, confidence, heatmap)
        """
        if self.model is None:
            return True, 1.0, np.zeros_like(face_image)
        
        try:
            # Convert and preprocess
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            tensor.requires_grad = True
            
            # Forward pass
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            live_prob = probabilities[0, 1].item()
            is_live = live_prob >= settings.LIVENESS_THRESHOLD
            
            # Compute Grad-CAM
            heatmap = self._compute_gradcam(tensor, logits, target_class=1)
            
            # Overlay heatmap on original image
            heatmap_resized = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
            heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(face_image, 0.6, heatmap_colored, 0.4, 0)
            
            return is_live, live_prob, overlay
        except Exception as e:
            print(f"Liveness with Grad-CAM failed: {e}")
            return True, 0.5, face_image
    
    def _compute_gradcam(
        self, 
        input_tensor: torch.Tensor, 
        logits: torch.Tensor, 
        target_class: int
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor
            logits: Model output logits
            target_class: Target class for gradient
            
        Returns:
            np.ndarray: Heatmap (H, W)
        """
        # Backward pass for target class
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()
        
        # Get gradients and activations from last conv layer
        # For ResNet-18, we use layer4 output
        gradients = None
        activations = None
        
        # Hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output
        
        # Register hooks
        target_layer = self.model.backbone.layer4
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Run forward and backward
        logits = self.model(input_tensor)
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Compute Grad-CAM
        if gradients is not None and activations is not None:
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination of activation maps
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().cpu().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam
        
        return np.zeros((7, 7))  # Default size for ResNet layer4 output
    
    def texture_analysis(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image texture features for spoof detection.
        
        Args:
            face_image: Face image
            
        Returns:
            Dict[str, float]: Texture analysis metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Compute local binary patterns variance
        lbp_var = self._compute_lbp_variance(gray)
        
        # Compute high-frequency content
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq = np.var(laplacian)
        
        # Compute color diversity (spoofs often have less color variation)
        color_std = np.mean([np.std(face_image[:, :, i]) for i in range(3)])
        
        return {
            'lbp_variance': float(lbp_var),
            'high_frequency': float(high_freq),
            'color_diversity': float(color_std)
        }
    
    def _compute_lbp_variance(self, gray_image: np.ndarray) -> float:
        """
        Compute LBP (Local Binary Pattern) variance.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            float: LBP variance
        """
        # Simple LBP implementation (8 neighbors)
        lbp = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                code = 0
                
                # 8 neighbors
                code |= (gray_image[i-1, j-1] >= center) << 7
                code |= (gray_image[i-1, j] >= center) << 6
                code |= (gray_image[i-1, j+1] >= center) << 5
                code |= (gray_image[i, j+1] >= center) << 4
                code |= (gray_image[i+1, j+1] >= center) << 3
                code |= (gray_image[i+1, j] >= center) << 2
                code |= (gray_image[i+1, j-1] >= center) << 1
                code |= (gray_image[i, j-1] >= center) << 0
                
                lbp[i, j] = code
        
        return float(np.var(lbp))


# Singleton instance
_liveness_instance: Optional[LivenessDetector] = None


def get_liveness_detector() -> LivenessDetector:
    """
    Get singleton liveness detector instance.
    
    Returns:
        LivenessDetector: Liveness detector
    """
    global _liveness_instance
    if _liveness_instance is None:
        _liveness_instance = LivenessDetector()
    return _liveness_instance


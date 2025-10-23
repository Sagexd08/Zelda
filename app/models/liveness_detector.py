
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

    def __init__(self, pretrained_path: Optional[str] = None):
        super().__init__()

        self.backbone = models.resnet18(pretrained=False)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"Loaded liveness weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load liveness weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        for name, module in self.backbone.named_children():
            x = module(x)
            if name == 'avgpool':
                return x
        return x

class LivenessDetector:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None

        self._init_model()
        self._init_transform()

    def _init_model(self):
        liveness_path = settings.get_model_path(settings.LIVENESS_MODEL)

        try:
            self.model = LivenessResNet(liveness_path if liveness_path.exists() else None)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ Liveness detector loaded")
        except Exception as e:
            print(f"✗ Failed to load liveness detector: {e}")

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, face_image: np.ndarray) -> Tuple[bool, float]:
        if self.model is None:
            return True, 1.0

        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)

            live_prob = probabilities[0, 1].item()

            is_live = live_prob >= settings.LIVENESS_THRESHOLD

            return is_live, live_prob
        except Exception as e:
            print(f"Liveness prediction failed: {e}")
            return True, 0.5

    def predict_with_gradcam(
        self,
        face_image: np.ndarray
    ) -> Tuple[bool, float, np.ndarray]:
        if self.model is None:
            return True, 1.0, np.zeros_like(face_image)

        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            tensor.requires_grad = True

            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            live_prob = probabilities[0, 1].item()
            is_live = live_prob >= settings.LIVENESS_THRESHOLD

            heatmap = self._compute_gradcam(tensor, logits, target_class=1)

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
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        gradients = None
        activations = None

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        target_layer = self.model.backbone.layer4
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        logits = self.model(input_tensor)
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        forward_handle.remove()
        backward_handle.remove()

        if gradients is not None and activations is not None:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = cam.squeeze().cpu().detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam

        return np.zeros((7, 7))

    def texture_analysis(self, face_image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        lbp_var = self._compute_lbp_variance(gray)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq = np.var(laplacian)

        color_std = np.mean([np.std(face_image[:, :, i]) for i in range(3)])

        return {
            'lbp_variance': float(lbp_var),
            'high_frequency': float(high_freq),
            'color_diversity': float(color_std)
        }

    def _compute_lbp_variance(self, gray_image: np.ndarray) -> float:
        lbp = np.zeros_like(gray_image)

        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                code = 0

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

_liveness_instance: Optional[LivenessDetector] = None

def get_liveness_detector() -> LivenessDetector:
    global _liveness_instance
    if _liveness_instance is None:
        _liveness_instance = LivenessDetector()
    return _liveness_instance

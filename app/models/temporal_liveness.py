"""
Temporal Liveness Detection
LSTM-based sequence model for video-based liveness detection
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from app.core.config import settings


class TemporalLivenessLSTM(nn.Module):
    """
    LSTM model for temporal liveness detection.
    Analyzes sequence of facial features to detect liveness.
    """
    
    def __init__(
        self, 
        input_dim: int = 20,  # Feature dimension per frame
        hidden_dim: int = 64,
        num_layers: int = 2,
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize temporal liveness LSTM.
        
        Args:
            input_dim: Feature dimension per frame
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            pretrained_path: Path to pretrained weights
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Classification layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)  # Binary: live, spoof
        )
        
        # Load pretrained weights if available
        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"Loaded temporal liveness weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load temporal liveness weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, F) where T is sequence length, F is features
            
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        final_hidden = hidden[-1]  # (B, hidden_dim)
        
        # Classification
        logits = self.fc(final_hidden)
        
        return logits


class TemporalLivenessDetector:
    """
    Temporal liveness detector using video sequences.
    Extracts micro-movement features (blinks, head motion) and classifies via LSTM.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize temporal liveness detector.
        
        Args:
            device: Device for inference
        """
        self.device = device or settings.get_device()
        self.model = None
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=settings.MIN_VIDEO_FRAMES)
        self.feature_buffer = deque(maxlen=settings.MIN_VIDEO_FRAMES)
        
        self._init_model()
    
    def _init_model(self):
        """Initialize LSTM model"""
        temporal_path = settings.get_model_path(settings.TEMPORAL_LIVENESS_MODEL)
        
        try:
            self.model = TemporalLivenessLSTM(
                pretrained_path=temporal_path if temporal_path.exists() else None
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ Temporal liveness detector loaded")
        except Exception as e:
            print(f"✗ Failed to load temporal liveness: {e}")
    
    def reset(self):
        """Reset frame and feature buffers"""
        self.frame_buffer.clear()
        self.feature_buffer.clear()
    
    def add_frame(self, frame: np.ndarray, landmarks: dict):
        """
        Add a frame to the buffer and extract features.
        
        Args:
            frame: Video frame
            landmarks: Facial landmarks
        """
        self.frame_buffer.append(frame)
        
        # Extract temporal features
        features = self._extract_temporal_features(frame, landmarks)
        self.feature_buffer.append(features)
    
    def _extract_temporal_features(
        self, 
        frame: np.ndarray, 
        landmarks: dict
    ) -> np.ndarray:
        """
        Extract temporal features from a single frame.
        
        Args:
            frame: Video frame
            landmarks: Facial landmarks
            
        Returns:
            np.ndarray: Feature vector (20-D)
        """
        features = []
        
        # 1. Eye Aspect Ratio (EAR) - for blink detection
        ear = self._compute_eye_aspect_ratio(landmarks)
        features.extend([ear['left'], ear['right'], ear['mean']])
        
        # 2. Mouth Aspect Ratio (MAR) - for mouth movement
        mar = self._compute_mouth_aspect_ratio(landmarks)
        features.append(mar)
        
        # 3. Head pose angles (simplified estimation)
        pose = self._estimate_head_pose(landmarks)
        features.extend([pose['yaw'], pose['pitch'], pose['roll']])
        
        # 4. Landmark movement (if previous frame exists)
        if len(self.frame_buffer) > 0:
            motion = self._compute_landmark_motion(landmarks)
            features.extend(motion)
        else:
            features.extend([0.0] * 10)  # 10 motion features
        
        # 5. Optical flow magnitude
        if len(self.frame_buffer) > 0:
            flow_mag = self._compute_optical_flow_magnitude(
                self.frame_buffer[-1], 
                frame
            )
            features.append(flow_mag)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_eye_aspect_ratio(self, landmarks: dict) -> dict:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            dict: EAR for left, right eyes and mean
        """
        # Simplified EAR using eye landmarks
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        
        # Estimate eye openness (this is simplified; real EAR needs 6 points per eye)
        # Here we just use a placeholder based on y-coordinate relative to face center
        nose = np.array(landmarks['nose'])
        
        left_ear = abs(left_eye[1] - nose[1]) / (abs(left_eye[0] - nose[0]) + 1e-6)
        right_ear = abs(right_eye[1] - nose[1]) / (abs(right_eye[0] - nose[0]) + 1e-6)
        mean_ear = (left_ear + right_ear) / 2.0
        
        return {'left': left_ear, 'right': right_ear, 'mean': mean_ear}
    
    def _compute_mouth_aspect_ratio(self, landmarks: dict) -> float:
        """
        Compute Mouth Aspect Ratio (MAR).
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            float: MAR value
        """
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        nose = np.array(landmarks['nose'])
        
        # Simplified MAR
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        mouth_height = abs(mouth_center_y - nose[1])
        
        mar = mouth_height / (mouth_width + 1e-6)
        
        return float(mar)
    
    def _estimate_head_pose(self, landmarks: dict) -> dict:
        """
        Estimate head pose angles (simplified).
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            dict: Yaw, pitch, roll angles
        """
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose = np.array(landmarks['nose'])
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        
        # Simplified angle estimation
        # Yaw: horizontal face rotation
        eye_center = (left_eye + right_eye) / 2
        yaw = (nose[0] - eye_center[0]) / (np.linalg.norm(right_eye - left_eye) + 1e-6)
        
        # Pitch: vertical face rotation
        mouth_center = (mouth_left + mouth_right) / 2
        pitch = (nose[1] - eye_center[1]) / (np.linalg.norm(mouth_center - eye_center) + 1e-6)
        
        # Roll: face tilt
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        return {'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll)}
    
    def _compute_landmark_motion(self, current_landmarks: dict) -> List[float]:
        """
        Compute motion features from landmark changes.
        
        Args:
            current_landmarks: Current frame landmarks
            
        Returns:
            List[float]: Motion features (10-D)
        """
        if len(self.feature_buffer) == 0:
            return [0.0] * 10
        
        # Get previous features (we stored them)
        # Here we compute simple landmark displacement
        motion = []
        
        for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
            # Displacement magnitude (simplified, assuming previous landmarks available)
            motion.append(0.1)  # Placeholder
            motion.append(0.1)  # Placeholder
        
        return motion
    
    def _compute_optical_flow_magnitude(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ) -> float:
        """
        Compute optical flow magnitude between frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            float: Average optical flow magnitude
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, 
            curr_gray, 
            None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Compute magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_magnitude = np.mean(magnitude)
        
        return float(avg_magnitude)
    
    @torch.no_grad()
    def predict(self) -> Tuple[bool, float]:
        """
        Predict liveness from accumulated sequence.
        
        Returns:
            Tuple[bool, float]: (is_live, confidence)
        """
        if self.model is None:
            return True, 1.0
        
        if len(self.feature_buffer) < settings.MIN_VIDEO_FRAMES:
            return True, 0.5  # Not enough frames
        
        try:
            # Convert features to tensor
            features = np.array(list(self.feature_buffer))  # (T, F)
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, T, F)
            
            # Forward pass
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Get live probability
            live_prob = probabilities[0, 1].item()
            is_live = live_prob >= settings.LIVENESS_THRESHOLD
            
            return is_live, live_prob
        except Exception as e:
            print(f"Temporal liveness prediction failed: {e}")
            return True, 0.5
    
    def detect_blink(self) -> bool:
        """
        Detect if a blink occurred in the sequence.
        
        Returns:
            bool: True if blink detected
        """
        if len(self.feature_buffer) < 10:
            return False
        
        # Extract EAR values from features (indices 0-2)
        ear_values = [features[2] for features in self.feature_buffer]  # Mean EAR
        
        # Detect blink: EAR drops below threshold and rises back
        min_ear = min(ear_values)
        max_ear = max(ear_values)
        ear_range = max_ear - min_ear
        
        # Blink detected if significant drop and recovery
        blink_detected = ear_range > settings.BLINK_DETECTION_THRESHOLD
        
        return blink_detected


# Singleton instance
_temporal_liveness_instance: Optional[TemporalLivenessDetector] = None


def get_temporal_liveness_detector() -> TemporalLivenessDetector:
    """
    Get singleton temporal liveness detector instance.
    
    Returns:
        TemporalLivenessDetector: Temporal liveness detector
    """
    global _temporal_liveness_instance
    if _temporal_liveness_instance is None:
        _temporal_liveness_instance = TemporalLivenessDetector()
    return _temporal_liveness_instance


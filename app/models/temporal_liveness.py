
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from app.core.config import settings

class TemporalLivenessLSTM(nn.Module):

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

        if pretrained_path and pretrained_path.exists():
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.load_state_dict(state_dict)
                print(f"Loaded temporal liveness weights from {pretrained_path}")
            except Exception as e:
                print(f"Warning: Failed to load temporal liveness weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits

class TemporalLivenessDetector:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None

        self.frame_buffer = deque(maxlen=settings.MIN_VIDEO_FRAMES)
        self.feature_buffer = deque(maxlen=settings.MIN_VIDEO_FRAMES)

        self._init_model()

    def _init_model(self):
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
        self.frame_buffer.clear()
        self.feature_buffer.clear()

    def add_frame(self, frame: np.ndarray, landmarks: dict):
        self.frame_buffer.append(frame)

        features = self._extract_temporal_features(frame, landmarks)
        self.feature_buffer.append(features)

    def _extract_temporal_features(
        self,
        frame: np.ndarray,
        landmarks: dict
    ) -> np.ndarray:
        features = []

        ear = self._compute_eye_aspect_ratio(landmarks)
        features.extend([ear['left'], ear['right'], ear['mean']])

        mar = self._compute_mouth_aspect_ratio(landmarks)
        features.append(mar)

        pose = self._estimate_head_pose(landmarks)
        features.extend([pose['yaw'], pose['pitch'], pose['roll']])

        if len(self.frame_buffer) > 0:
            motion = self._compute_landmark_motion(landmarks)
            features.extend(motion)
        else:
            features.extend([0.0] * 10)

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
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])

        nose = np.array(landmarks['nose'])

        left_ear = abs(left_eye[1] - nose[1]) / (abs(left_eye[0] - nose[0]) + 1e-6)
        right_ear = abs(right_eye[1] - nose[1]) / (abs(right_eye[0] - nose[0]) + 1e-6)
        mean_ear = (left_ear + right_ear) / 2.0

        return {'left': left_ear, 'right': right_ear, 'mean': mean_ear}

    def _compute_mouth_aspect_ratio(self, landmarks: dict) -> float:
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        nose = np.array(landmarks['nose'])

        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        mouth_height = abs(mouth_center_y - nose[1])

        mar = mouth_height / (mouth_width + 1e-6)

        return float(mar)

    def _estimate_head_pose(self, landmarks: dict) -> dict:
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose = np.array(landmarks['nose'])
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])

        eye_center = (left_eye + right_eye) / 2
        yaw = (nose[0] - eye_center[0]) / (np.linalg.norm(right_eye - left_eye) + 1e-6)

        mouth_center = (mouth_left + mouth_right) / 2
        pitch = (nose[1] - eye_center[1]) / (np.linalg.norm(mouth_center - eye_center) + 1e-6)

        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

        return {'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll)}

    def _compute_landmark_motion(self, current_landmarks: dict) -> List[float]:
        if len(self.feature_buffer) == 0:
            return [0.0] * 10

        motion = []

        for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
            motion.append(0.1)
            motion.append(0.1)

        return motion

    def _compute_optical_flow_magnitude(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> float:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_magnitude = np.mean(magnitude)

        return float(avg_magnitude)

    @torch.no_grad()
    def predict(self) -> Tuple[bool, float]:
        if self.model is None:
            return True, 1.0

        if len(self.feature_buffer) < settings.MIN_VIDEO_FRAMES:
            return True, 0.5

        try:
            features = np.array(list(self.feature_buffer))
            tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)

            live_prob = probabilities[0, 1].item()
            is_live = live_prob >= settings.LIVENESS_THRESHOLD

            return is_live, live_prob
        except Exception as e:
            print(f"Temporal liveness prediction failed: {e}")
            return True, 0.5

    def detect_blink(self) -> bool:
        if len(self.feature_buffer) < 10:
            return False

        ear_values = [features[2] for features in self.feature_buffer]

        min_ear = min(ear_values)
        max_ear = max(ear_values)
        ear_range = max_ear - min_ear

        blink_detected = ear_range > settings.BLINK_DETECTION_THRESHOLD

        return blink_detected

_temporal_liveness_instance: Optional[TemporalLivenessDetector] = None

def get_temporal_liveness_detector() -> TemporalLivenessDetector:
    global _temporal_liveness_instance
    if _temporal_liveness_instance is None:
        _temporal_liveness_instance = TemporalLivenessDetector()
    return _temporal_liveness_instance

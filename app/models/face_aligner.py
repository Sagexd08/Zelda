
from typing import Tuple, Dict, Optional
import numpy as np
import cv2
from sklearn.preprocessing import normalize

from app.models.face_detector import FaceDetection

class FaceAligner:

    STANDARD_LANDMARKS_160 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    STANDARD_LANDMARKS_224 = np.array([
        [53.6, 72.0],
        [102.4, 72.0],
        [78.0, 100.0],
        [58.0, 129.0],
        [98.0, 129.0]
    ], dtype=np.float32)

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def align(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        output_size: int = 160
    ) -> np.ndarray:
        if output_size == 224:
            standard_landmarks = self.STANDARD_LANDMARKS_224
        else:
            standard_landmarks = self.STANDARD_LANDMARKS_160

        source_landmarks = self._extract_landmarks_array(detection.landmarks)

        transform_matrix = self._estimate_transform(source_landmarks, standard_landmarks)

        aligned_face = cv2.warpAffine(
            image,
            transform_matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR
        )

        aligned_face = self._normalize_lighting(aligned_face)

        return aligned_face

    def _extract_landmarks_array(
        self,
        landmarks: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        return np.array([
            landmarks['left_eye'],
            landmarks['right_eye'],
            landmarks['nose'],
            landmarks['mouth_left'],
            landmarks['mouth_right']
        ], dtype=np.float32)

    def _estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> np.ndarray:
        transform_matrix = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.LMEDS
        )[0]

        return transform_matrix

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        l = self.clahe.apply(l)

        lab = cv2.merge([l, a, b])

        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        normalized = self._gray_world_normalization(normalized)

        return normalized

    def _gray_world_normalization(self, image: np.ndarray) -> np.ndarray:
        mean_b = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_r = np.mean(image[:, :, 2])

        mean_overall = (mean_b + mean_g + mean_r) / 3

        if mean_b > 0 and mean_g > 0 and mean_r > 0:
            scale_b = mean_overall / mean_b
            scale_g = mean_overall / mean_g
            scale_r = mean_overall / mean_r

            normalized = image.astype(np.float32)
            normalized[:, :, 0] *= scale_b
            normalized[:, :, 1] *= scale_g
            normalized[:, :, 2] *= scale_r

            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

            return normalized

        return image

    def align_crop(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        output_size: int = 160,
        margin: float = 0.2
    ) -> np.ndarray:
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)

        face_crop = image[y1:y2, x1:x2]

        face_resized = cv2.resize(face_crop, (output_size, output_size))

        face_normalized = self._normalize_lighting(face_resized)

        return face_normalized

_aligner_instance: Optional[FaceAligner] = None

def get_face_aligner() -> FaceAligner:
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = FaceAligner()
    return _aligner_instance

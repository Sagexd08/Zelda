
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch
from dataclasses import dataclass

try:
    from retinaface import RetinaFace as RetinaFaceModel
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("Warning: RetinaFace not available, using MTCNN only")

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not available")

from app.core.config import settings

@dataclass
class FaceDetection:
    bbox: np.ndarray
    confidence: float
    landmarks: Dict[str, Tuple[int, int]]
    quality_score: float = 1.0

class HybridFaceDetector:

    def __init__(
        self,
        confidence_threshold: float = None,
        nms_threshold: float = None,
        device: str = None
    ):
        self.confidence_threshold = confidence_threshold or settings.DETECTION_CONFIDENCE_THRESHOLD
        self.nms_threshold = nms_threshold or settings.DETECTION_NMS_THRESHOLD
        self.device = device or str(settings.get_device())

        self.retinaface = None
        self.mtcnn = None

        self._init_detectors()

    def _init_detectors(self):
        if RETINAFACE_AVAILABLE:
            try:
                self.retinaface_available = True
                print("RetinaFace detector initialized")
            except Exception as e:
                print(f"Failed to initialize RetinaFace: {e}")
                self.retinaface_available = False
        else:
            self.retinaface_available = False

        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    min_face_size=settings.MIN_FACE_SIZE,
                    device=self.device,
                    keep_all=True,
                    post_process=False
                )
                print("MTCNN detector initialized")
            except Exception as e:
                print(f"Failed to initialize MTCNN: {e}")
                self.mtcnn = None

        if not self.retinaface_available and self.mtcnn is None:
            raise RuntimeError("No face detector available. Install retinaface-pytorch or facenet-pytorch.")

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = None
    ) -> List[FaceDetection]:
        max_faces = max_faces or settings.MAX_FACES

        if self.retinaface_available:
            try:
                detections = self._detect_retinaface(image, max_faces)
                if detections:
                    return detections
            except Exception as e:
                print(f"RetinaFace detection failed: {e}, falling back to MTCNN")

        if self.mtcnn is not None:
            try:
                detections = self._detect_mtcnn(image, max_faces)
                return detections
            except Exception as e:
                print(f"MTCNN detection failed: {e}")

        return []

    def _detect_retinaface(
        self,
        image: np.ndarray,
        max_faces: int
    ) -> List[FaceDetection]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = RetinaFaceModel.detect_faces(rgb_image)

        if not faces or faces == {}:
            return []

        detections = []
        for key, face_data in faces.items():
            if len(detections) >= max_faces:
                break

            facial_area = face_data['facial_area']
            bbox = np.array([
                facial_area[0],
                facial_area[1],
                facial_area[2],
                facial_area[3]
            ])

            confidence = face_data['score']

            if confidence < self.confidence_threshold:
                continue

            landmarks = {
                'left_eye': tuple(face_data['landmarks']['left_eye']),
                'right_eye': tuple(face_data['landmarks']['right_eye']),
                'nose': tuple(face_data['landmarks']['nose']),
                'mouth_left': tuple(face_data['landmarks']['mouth_left']),
                'mouth_right': tuple(face_data['landmarks']['mouth_right'])
            }

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            area = bbox_width * bbox_height
            img_area = image.shape[0] * image.shape[1]
            quality_score = min(1.0, (area / img_area) * 10)

            detection = FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                quality_score=quality_score
            )
            detections.append(detection)

        detections.sort(key=lambda x: x.confidence, reverse=True)

        return detections[:max_faces]

    def _detect_mtcnn(
        self,
        image: np.ndarray,
        max_faces: int
    ) -> List[FaceDetection]:
        from PIL import Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
            if len(detections) >= max_faces:
                break

            if prob < self.confidence_threshold:
                continue

            bbox = np.array([
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3])
            ])

            landmarks_dict = {
                'left_eye': (int(landmark[0][0]), int(landmark[0][1])),
                'right_eye': (int(landmark[1][0]), int(landmark[1][1])),
                'nose': (int(landmark[2][0]), int(landmark[2][1])),
                'mouth_left': (int(landmark[3][0]), int(landmark[3][1])),
                'mouth_right': (int(landmark[4][0]), int(landmark[4][1]))
            }

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            area = bbox_width * bbox_height
            img_area = image.shape[0] * image.shape[1]
            quality_score = min(1.0, (area / img_area) * 10)

            detection = FaceDetection(
                bbox=bbox,
                confidence=float(prob),
                landmarks=landmarks_dict,
                quality_score=quality_score
            )
            detections.append(detection)

        detections.sort(key=lambda x: x.confidence, reverse=True)

        return detections[:max_faces]

    def detect_largest(self, image: np.ndarray) -> Optional[FaceDetection]:
        detections = self.detect(image, max_faces=settings.MAX_FACES)

        if not detections:
            return None

        largest = max(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        )

        return largest

    def visualize_detection(
        self,
        image: np.ndarray,
        detections: List[FaceDetection]
    ) -> np.ndarray:
        vis_image = image.copy()

        for detection in detections:
            bbox = detection.bbox.astype(int)
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )

            for name, (x, y) in detection.landmarks.items():
                cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)

            text = f"{detection.confidence:.2f}"
            cv2.putText(
                vis_image,
                text,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return vis_image

_detector_instance: Optional[HybridFaceDetector] = None

def get_face_detector() -> HybridFaceDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HybridFaceDetector()
    return _detector_instance

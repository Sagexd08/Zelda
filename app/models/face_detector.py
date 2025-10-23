"""
Face Detection Module
Hybrid approach with RetinaFace (primary) and MTCNN (fallback)
"""

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
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: MTCNN not available")

from app.core.config import settings


@dataclass
class FaceDetection:
    """
    Face detection result containing all relevant information.
    
    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Detection confidence score
        landmarks: 5-point facial landmarks {left_eye, right_eye, nose, mouth_left, mouth_right}
        quality_score: Face quality score (0-1)
    """
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    landmarks: Dict[str, Tuple[int, int]]
    quality_score: float = 1.0


class HybridFaceDetector:
    """
    Hybrid face detector using RetinaFace as primary and MTCNN as fallback.
    Provides robust face detection with facial landmarks for alignment.
    """
    
    def __init__(
        self,
        confidence_threshold: float = None,
        nms_threshold: float = None,
        device: str = None
    ):
        """
        Initialize hybrid face detector.
        
        Args:
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            device: Device to run detection on ('cuda' or 'cpu')
        """
        self.confidence_threshold = confidence_threshold or settings.DETECTION_CONFIDENCE_THRESHOLD
        self.nms_threshold = nms_threshold or settings.DETECTION_NMS_THRESHOLD
        self.device = device or str(settings.get_device())
        
        self.retinaface = None
        self.mtcnn = None
        
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize detection models"""
        # Initialize RetinaFace (primary)
        if RETINAFACE_AVAILABLE:
            try:
                # RetinaFace will be loaded on first use
                self.retinaface_available = True
                print("RetinaFace detector initialized")
            except Exception as e:
                print(f"Failed to initialize RetinaFace: {e}")
                self.retinaface_available = False
        else:
            self.retinaface_available = False
        
        # Initialize MTCNN (fallback)
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    min_face_size=settings.MIN_FACE_SIZE,
                    device=self.device
                )
                print("MTCNN detector initialized")
            except Exception as e:
                print(f"Failed to initialize MTCNN: {e}")
                self.mtcnn = None
        
        if not self.retinaface_available and self.mtcnn is None:
            raise RuntimeError("No face detector available. Install retinaface-pytorch or mtcnn.")
    
    def detect(
        self, 
        image: np.ndarray, 
        max_faces: int = None
    ) -> List[FaceDetection]:
        """
        Detect faces in an image using hybrid approach.
        
        Args:
            image: Input image (BGR format)
            max_faces: Maximum number of faces to detect
            
        Returns:
            List[FaceDetection]: List of detected faces
        """
        max_faces = max_faces or settings.MAX_FACES
        
        # Try RetinaFace first
        if self.retinaface_available:
            try:
                detections = self._detect_retinaface(image, max_faces)
                if detections:
                    return detections
            except Exception as e:
                print(f"RetinaFace detection failed: {e}, falling back to MTCNN")
        
        # Fallback to MTCNN
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
        """
        Detect faces using RetinaFace.
        
        Args:
            image: Input image (BGR format)
            max_faces: Maximum number of faces
            
        Returns:
            List[FaceDetection]: Detected faces
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces using RetinaFace
        faces = RetinaFaceModel.detect_faces(rgb_image)
        
        if not faces or faces == {}:
            return []
        
        detections = []
        for key, face_data in faces.items():
            if len(detections) >= max_faces:
                break
            
            # Extract bounding box
            facial_area = face_data['facial_area']
            bbox = np.array([
                facial_area[0],  # x1
                facial_area[1],  # y1
                facial_area[2],  # x2
                facial_area[3]   # y2
            ])
            
            # Extract confidence
            confidence = face_data['score']
            
            if confidence < self.confidence_threshold:
                continue
            
            # Extract landmarks
            landmarks = {
                'left_eye': tuple(face_data['landmarks']['left_eye']),
                'right_eye': tuple(face_data['landmarks']['right_eye']),
                'nose': tuple(face_data['landmarks']['nose']),
                'mouth_left': tuple(face_data['landmarks']['mouth_left']),
                'mouth_right': tuple(face_data['landmarks']['mouth_right'])
            }
            
            # Compute quality score (simple heuristic based on bbox size)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            area = bbox_width * bbox_height
            img_area = image.shape[0] * image.shape[1]
            quality_score = min(1.0, (area / img_area) * 10)  # Normalize
            
            detection = FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
                quality_score=quality_score
            )
            detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections[:max_faces]
    
    def _detect_mtcnn(
        self, 
        image: np.ndarray, 
        max_faces: int
    ) -> List[FaceDetection]:
        """
        Detect faces using MTCNN.
        
        Args:
            image: Input image (BGR format)
            max_faces: Maximum number of faces
            
        Returns:
            List[FaceDetection]: Detected faces
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.mtcnn.detect_faces(rgb_image)
        
        if not faces:
            return []
        
        detections = []
        for face_data in faces:
            if len(detections) >= max_faces:
                break
            
            # Extract bounding box
            bbox_list = face_data['box']
            bbox = np.array([
                bbox_list[0],  # x1
                bbox_list[1],  # y1
                bbox_list[0] + bbox_list[2],  # x2
                bbox_list[1] + bbox_list[3]   # y2
            ])
            
            # Extract confidence
            confidence = face_data['confidence']
            
            if confidence < self.confidence_threshold:
                continue
            
            # Extract landmarks (MTCNN provides: left_eye, right_eye, nose, mouth_left, mouth_right)
            keypoints = face_data['keypoints']
            landmarks = {
                'left_eye': tuple(keypoints['left_eye']),
                'right_eye': tuple(keypoints['right_eye']),
                'nose': tuple(keypoints['nose']),
                'mouth_left': tuple(keypoints['mouth_left']),
                'mouth_right': tuple(keypoints['mouth_right'])
            }
            
            # Compute quality score
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
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections[:max_faces]
    
    def detect_largest(self, image: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect the largest face in an image (most common use case).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Optional[FaceDetection]: Largest detected face or None
        """
        detections = self.detect(image, max_faces=settings.MAX_FACES)
        
        if not detections:
            return None
        
        # Return the face with largest bbox area
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
        """
        Visualize face detections on image.
        
        Args:
            image: Input image
            detections: List of face detections
            
        Returns:
            np.ndarray: Image with visualized detections
        """
        vis_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            bbox = detection.bbox.astype(int)
            cv2.rectangle(
                vis_image, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (0, 255, 0), 
                2
            )
            
            # Draw landmarks
            for name, (x, y) in detection.landmarks.items():
                cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Draw confidence
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


# Singleton instance for reuse
_detector_instance: Optional[HybridFaceDetector] = None


def get_face_detector() -> HybridFaceDetector:
    """
    Get singleton face detector instance.
    
    Returns:
        HybridFaceDetector: Face detector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HybridFaceDetector()
    return _detector_instance


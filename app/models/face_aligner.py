"""
Face Alignment Module
5-point landmark alignment with lighting normalization
"""

from typing import Tuple, Dict, Optional
import numpy as np
import cv2
from sklearn.preprocessing import normalize

from app.models.face_detector import FaceDetection


class FaceAligner:
    """
    Face alignment using 5-point landmarks.
    Includes lighting normalization (CLAHE + color constancy).
    """
    
    # Standard facial landmarks for alignment (eyes and nose)
    # Based on standard face template
    STANDARD_LANDMARKS_160 = np.array([
        [38.2946, 51.6963],  # left_eye
        [73.5318, 51.5014],  # right_eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # mouth_left
        [70.7299, 92.2041]   # mouth_right
    ], dtype=np.float32)
    
    STANDARD_LANDMARKS_224 = np.array([
        [53.6, 72.0],     # left_eye
        [102.4, 72.0],    # right_eye
        [78.0, 100.0],    # nose
        [58.0, 129.0],    # mouth_left
        [98.0, 129.0]     # mouth_right
    ], dtype=np.float32)
    
    def __init__(self):
        """Initialize face aligner"""
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def align(
        self, 
        image: np.ndarray, 
        detection: FaceDetection, 
        output_size: int = 160
    ) -> np.ndarray:
        """
        Align face using 5-point landmarks.
        
        Args:
            image: Input image (BGR format)
            detection: Face detection with landmarks
            output_size: Output image size (160 or 224)
            
        Returns:
            np.ndarray: Aligned face image
        """
        # Get standard landmarks for target size
        if output_size == 224:
            standard_landmarks = self.STANDARD_LANDMARKS_224
        else:
            standard_landmarks = self.STANDARD_LANDMARKS_160
        
        # Extract source landmarks
        source_landmarks = self._extract_landmarks_array(detection.landmarks)
        
        # Compute similarity transform
        transform_matrix = self._estimate_transform(source_landmarks, standard_landmarks)
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image, 
            transform_matrix, 
            (output_size, output_size),
            flags=cv2.INTER_LINEAR
        )
        
        # Apply lighting normalization
        aligned_face = self._normalize_lighting(aligned_face)
        
        return aligned_face
    
    def _extract_landmarks_array(
        self, 
        landmarks: Dict[str, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Convert landmarks dict to ordered numpy array.
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            np.ndarray: Landmarks array (5, 2)
        """
        # Order: left_eye, right_eye, nose, mouth_left, mouth_right
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
        """
        Estimate similarity transform between source and destination points.
        
        Args:
            src_points: Source landmark points
            dst_points: Destination landmark points
            
        Returns:
            np.ndarray: 2x3 transformation matrix
        """
        # Estimate similarity transform (rotation + scale + translation)
        transform_matrix = cv2.estimateAffinePartial2D(
            src_points, 
            dst_points,
            method=cv2.LMEDS
        )[0]
        
        return transform_matrix
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting using CLAHE and color constancy.
        
        Args:
            image: Input face image (BGR format)
            
        Returns:
            np.ndarray: Normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply gray-world color constancy
        normalized = self._gray_world_normalization(normalized)
        
        return normalized
    
    def _gray_world_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gray-world color constancy assumption.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            np.ndarray: Color-normalized image
        """
        # Compute mean for each channel
        mean_b = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_r = np.mean(image[:, :, 2])
        
        # Compute overall mean
        mean_overall = (mean_b + mean_g + mean_r) / 3
        
        # Compute scaling factors
        if mean_b > 0 and mean_g > 0 and mean_r > 0:
            scale_b = mean_overall / mean_b
            scale_g = mean_overall / mean_g
            scale_r = mean_overall / mean_r
            
            # Apply scaling
            normalized = image.astype(np.float32)
            normalized[:, :, 0] *= scale_b
            normalized[:, :, 1] *= scale_g
            normalized[:, :, 2] *= scale_r
            
            # Clip values
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
        """
        Alternative alignment: crop face with margin and resize.
        Useful when landmarks are not reliable.
        
        Args:
            image: Input image (BGR format)
            detection: Face detection
            output_size: Output size
            margin: Margin around face (fraction of bbox size)
            
        Returns:
            np.ndarray: Cropped and resized face
        """
        bbox = detection.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Add margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Expand bbox with margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to output size
        face_resized = cv2.resize(face_crop, (output_size, output_size))
        
        # Apply lighting normalization
        face_normalized = self._normalize_lighting(face_resized)
        
        return face_normalized


# Singleton instance
_aligner_instance: Optional[FaceAligner] = None


def get_face_aligner() -> FaceAligner:
    """
    Get singleton face aligner instance.
    
    Returns:
        FaceAligner: Face aligner instance
    """
    global _aligner_instance
    if _aligner_instance is None:
        _aligner_instance = FaceAligner()
    return _aligner_instance


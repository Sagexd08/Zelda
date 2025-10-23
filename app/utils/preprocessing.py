
from typing import Tuple, Dict
import numpy as np
import cv2

from app.core.config import settings

def assess_image_quality(image: np.ndarray) -> Dict[str, float]:
    metrics = {}

    metrics['sharpness'] = compute_sharpness(image)

    metrics['blur_score'] = detect_blur(image)

    metrics['brightness'] = compute_brightness(image)

    metrics['contrast'] = compute_contrast(image)

    metrics['overall_quality'] = compute_overall_quality(metrics)

    return metrics

def compute_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)

def detect_blur(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    blur_score = np.mean(magnitude)

    return float(blur_score)

def compute_brightness(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])

    return float(brightness)

def compute_contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    return float(contrast)

def compute_overall_quality(metrics: Dict[str, float]) -> float:
    sharpness_norm = min(1.0, metrics['sharpness'] / 500.0)
    blur_norm = min(1.0, metrics['blur_score'] / 50.0)
    brightness_norm = 1.0 - abs(metrics['brightness'] - 128) / 128.0
    contrast_norm = min(1.0, metrics['contrast'] / 100.0)

    overall = (
        0.35 * sharpness_norm +
        0.35 * blur_norm +
        0.15 * brightness_norm +
        0.15 * contrast_norm
    )

    return float(overall)

def is_image_acceptable(metrics: Dict[str, float]) -> Tuple[bool, str]:
    if metrics['sharpness'] < settings.MIN_SHARPNESS_SCORE:
        return False, "Image too blurry"

    if metrics['blur_score'] > settings.MAX_BLUR_VARIANCE:
        return False, "Excessive motion blur"

    if metrics['brightness'] < 50:
        return False, "Image too dark"

    if metrics['brightness'] > 230:
        return False, "Image overexposed"

    if metrics['contrast'] < 20:
        return False, "Low contrast"

    return True, "Acceptable quality"

def estimate_head_pose_simple(landmarks: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])
    nose = np.array(landmarks['nose'])
    mouth_left = np.array(landmarks['mouth_left'])
    mouth_right = np.array(landmarks['mouth_right'])

    eye_center = (left_eye + right_eye) / 2

    mouth_center = (mouth_left + mouth_right) / 2

    eye_dist = np.linalg.norm(right_eye - left_eye)
    yaw_offset = nose[0] - eye_center[0]
    yaw = np.degrees(np.arctan2(yaw_offset, eye_dist))

    face_height = np.linalg.norm(mouth_center - eye_center)
    nose_offset = nose[1] - (eye_center[1] + face_height * 0.4)
    pitch = np.degrees(np.arctan2(nose_offset, face_height))

    eye_diff = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_diff[1], eye_diff[0]))

    return {
        'yaw': float(yaw),
        'pitch': float(pitch),
        'roll': float(roll)
    }

def is_pose_acceptable(pose: Dict[str, float]) -> Tuple[bool, str]:
    if abs(pose['yaw']) > settings.MAX_HEAD_POSE_YAW:
        return False, f"Head turned too far (yaw: {pose['yaw']:.1f}°)"

    if abs(pose['pitch']) > settings.MAX_HEAD_POSE_PITCH:
        return False, f"Head tilted too much (pitch: {pose['pitch']:.1f}°)"

    return True, "Pose acceptable"

def normalize_face_image(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)

    mean = np.mean(image_float, axis=(0, 1), keepdims=True)
    std = np.std(image_float, axis=(0, 1), keepdims=True)
    normalized = (image_float - mean) / (std + 1e-8)

    normalized = ((normalized - normalized.min()) /
                  (normalized.max() - normalized.min() + 1e-8) * 255)

    return normalized.astype(np.uint8)

def augment_face_batch(images: np.ndarray) -> np.ndarray:
    augmented = []

    for image in images:
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)

        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        contrast_factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        if np.random.rand() > 0.8:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        augmented.append(image)

    return np.array(augmented)

def detect_occlusion(landmarks: Dict[str, Tuple[int, int]],
                      image: np.ndarray) -> Dict[str, bool]:
    occlusion = {}

    for name, (x, y) in landmarks.items():
        x, y = int(x), int(y)

        region_size = 10
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(image.shape[1], x + region_size)
        y2 = min(image.shape[0], y + region_size)

        region = image[y1:y2, x1:x2]

        mean_intensity = np.mean(region)
        is_occluded = mean_intensity < 20 or mean_intensity > 240

        occlusion[name] = is_occluded

    return occlusion

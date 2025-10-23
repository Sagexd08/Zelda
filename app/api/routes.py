"""
API Routes
REST endpoints for facial authentication system
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np
import cv2
from io import BytesIO

from app.core.database import get_db, Session
from app.core.config import settings
from app.services.registration_service import get_registration_service
from app.services.authentication_service import get_authentication_service
from app.services.identification_service import get_identification_service
from app.services.adaptive_learning_service import get_adaptive_learning_service
from app.utils.challenge import get_challenge_validator


router = APIRouter()


# ============================================
# Request/Response Models
# ============================================

class RegisterRequest(BaseModel):
    """Registration request model"""
    user_id: str = Field(..., description="Unique user identifier")


class RegisterResponse(BaseModel):
    """Registration response model"""
    success: bool
    user_id: Optional[str] = None
    samples_processed: Optional[int] = None
    valid_samples: Optional[int] = None
    avg_quality_score: Optional[float] = None
    avg_liveness_score: Optional[float] = None
    error: Optional[str] = None


class AuthenticateRequest(BaseModel):
    """Authentication request model"""
    user_id: str = Field(..., description="User identifier to authenticate")


class AuthenticateResponse(BaseModel):
    """Authentication response model"""
    authenticated: bool
    confidence: float
    threshold: Optional[float] = None
    liveness_score: Optional[float] = None
    reason: str
    similarities: Optional[dict] = None


class IdentifyResponse(BaseModel):
    """Identification response model"""
    found: bool
    liveness_score: Optional[float] = None
    matches: List[dict]
    total_users_checked: Optional[int] = None
    reason: Optional[str] = None


class DeleteUserRequest(BaseModel):
    """Delete user request model"""
    user_id: str = Field(..., description="User identifier to delete")


class ChallengeResponse(BaseModel):
    """Challenge response model"""
    challenge_id: str
    challenge_type: str
    instructions: str
    expires_at: str


# ============================================
# Helper Functions
# ============================================

def decode_image(file: UploadFile) -> np.ndarray:
    """
    Decode uploaded image file to numpy array.
    
    Args:
        file: Uploaded file
        
    Returns:
        np.ndarray: Image in BGR format
        
    Raises:
        HTTPException: If image cannot be decoded
    """
    try:
        # Read file bytes
        contents = file.file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    finally:
        file.file.close()


# ============================================
# Registration Endpoints
# ============================================

@router.post("/register", response_model=RegisterResponse)
async def register_user(
    user_id: str,
    images: List[UploadFile] = File(..., description="Multiple face images for registration"),
    db: Session = Depends(get_db)
):
    """
    Register a new user with multiple face samples.
    
    **Requirements:**
    - Minimum 5 face images
    - Maximum 10 face images
    - Each image must contain a clear, frontal face
    - Images must pass quality and liveness checks
    
    **Returns:**
    - Success status
    - Quality metrics
    - Validation details
    """
    registration_service = get_registration_service()
    
    # Decode images
    face_images = []
    for img_file in images:
        try:
            image = decode_image(img_file)
            face_images.append(image)
        except HTTPException:
            continue  # Skip invalid images
    
    if not face_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images provided"
        )
    
    # Register user
    success, result = registration_service.register_user(db, user_id, face_images)
    
    if success:
        return RegisterResponse(
            success=True,
            user_id=result['user_id'],
            samples_processed=result['samples_processed'],
            valid_samples=result['valid_samples'],
            avg_quality_score=result['avg_quality_score'],
            avg_liveness_score=result['avg_liveness_score']
        )
    else:
        return RegisterResponse(
            success=False,
            error=result.get('error', 'Registration failed')
        )


# ============================================
# Authentication Endpoints
# ============================================

@router.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate_user(
    user_id: str,
    image: UploadFile = File(..., description="Face image for authentication"),
    db: Session = Depends(get_db)
):
    """
    Authenticate a registered user.
    
    **Process:**
    1. Face detection
    2. Liveness detection
    3. Embedding extraction
    4. Similarity matching
    5. Threshold verification
    
    **Returns:**
    - Authentication result
    - Confidence score
    - Liveness score
    - Detailed reason
    """
    authentication_service = get_authentication_service()
    
    # Decode image
    face_image = decode_image(image)
    
    # Authenticate
    authenticated, result = authentication_service.authenticate(db, user_id, face_image)
    
    return AuthenticateResponse(
        authenticated=result['authenticated'],
        confidence=result['confidence'],
        threshold=result.get('threshold'),
        liveness_score=result.get('liveness_score'),
        reason=result['reason'],
        similarities=result.get('similarities')
    )


# ============================================
# Identification Endpoints
# ============================================

@router.post("/identify", response_model=IdentifyResponse)
async def identify_user(
    image: UploadFile = File(..., description="Unknown face image"),
    top_k: int = 3,
    db: Session = Depends(get_db)
):
    """
    Identify person from unknown face (1:N matching).
    
    **Process:**
    1. Face detection
    2. Liveness verification
    3. Embedding extraction
    4. Comparison against all enrolled users
    5. Return top-K matches
    
    **Parameters:**
    - top_k: Number of top matches to return (default: 3)
    
    **Returns:**
    - List of top matches with confidence scores
    - Liveness verification result
    """
    identification_service = get_identification_service()
    
    # Decode image
    face_image = decode_image(image)
    
    # Identify
    found, result = identification_service.identify(db, face_image, top_k=top_k)
    
    return IdentifyResponse(
        found=result['found'],
        liveness_score=result.get('liveness_score'),
        matches=result['matches'],
        total_users_checked=result.get('total_users_checked'),
        reason=result.get('reason')
    )


# ============================================
# User Management Endpoints
# ============================================

@router.post("/delete_user")
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete user and all associated data (GDPR compliant).
    
    **Deletes:**
    - User profile
    - Face embeddings
    - Liveness signatures
    - Audit logs
    - All personal data
    
    **Returns:**
    - Success status
    - Confirmation message
    """
    registration_service = get_registration_service()
    
    success, message = registration_service.delete_user(db, user_id)
    
    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )


# ============================================
# Challenge-Response Endpoints
# ============================================

@router.post("/challenge/create", response_model=ChallengeResponse)
async def create_challenge(user_id: Optional[str] = None):
    """
    Create a random challenge for liveness verification.
    
    **Challenge Types:**
    - Blink
    - Smile
    - Look left/right
    - Look up/down
    - Nod
    - Shake head
    
    **Returns:**
    - Challenge ID
    - Challenge type
    - Instructions
    - Expiration time
    """
    if not settings.ENABLE_CHALLENGE_RESPONSE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Challenge-response is disabled"
        )
    
    challenge_validator = get_challenge_validator()
    
    challenge = challenge_validator.create_challenge(user_id)
    
    return ChallengeResponse(
        challenge_id=challenge['challenge_id'],
        challenge_type=challenge['challenge_type'],
        instructions=challenge['instructions'],
        expires_at=challenge['expires_at'].isoformat()
    )


@router.post("/challenge/validate")
async def validate_challenge(
    challenge_id: str,
    sequence_data: dict
):
    """
    Validate challenge response.
    
    **Parameters:**
    - challenge_id: Challenge identifier
    - sequence_data: Dictionary containing sequences (EAR, MAR, yaw, pitch, etc.)
    
    **Returns:**
    - Validation result
    - Confidence score
    - Detailed message
    """
    challenge_validator = get_challenge_validator()
    
    is_valid, confidence, message = challenge_validator.validate_challenge(
        challenge_id, sequence_data
    )
    
    return {
        "valid": is_valid,
        "confidence": confidence,
        "message": message
    }


# ============================================
# Adaptive Learning Endpoints
# ============================================

@router.post("/adaptive/calibrate_threshold")
async def calibrate_threshold(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Calibrate verification threshold for user using Bayesian approach.
    
    **Requirements:**
    - User must have at least 10 authentication attempts
    
    **Returns:**
    - Calibrated threshold
    - Confidence score
    """
    if not settings.ENABLE_ADAPTIVE_THRESHOLD:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Adaptive thresholding is disabled"
        )
    
    adaptive_service = get_adaptive_learning_service()
    
    threshold = adaptive_service.calibrate_threshold(db, user_id)
    
    return {
        "user_id": user_id,
        "calibrated_threshold": threshold
    }


# ============================================
# System Information Endpoints
# ============================================

@router.get("/system/info")
async def system_info():
    """
    Get system configuration and status.
    
    **Returns:**
    - Configuration parameters
    - Enabled features
    - Model information
    """
    return {
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "features": {
            "liveness_detection": True,
            "depth_estimation": settings.ENABLE_DEPTH_ESTIMATION,
            "temporal_liveness": settings.ENABLE_TEMPORAL_LIVENESS,
            "voice_authentication": settings.ENABLE_VOICE_AUTH,
            "challenge_response": settings.ENABLE_CHALLENGE_RESPONSE,
            "adaptive_learning": settings.ENABLE_ONLINE_LEARNING,
            "adaptive_threshold": settings.ENABLE_ADAPTIVE_THRESHOLD,
            "bias_monitoring": settings.ENABLE_BIAS_MONITORING
        },
        "models": {
            "face_size": settings.FACE_SIZE,
            "verification_threshold": settings.VERIFICATION_THRESHOLD,
            "liveness_threshold": settings.LIVENESS_THRESHOLD,
            "min_registration_samples": settings.MIN_REGISTRATION_SAMPLES
        }
    }


@router.get("/system/models")
async def list_models():
    """
    List available ML models and their status.
    
    **Returns:**
    - Model names
    - Model status (loaded/not loaded)
    - Model versions
    """
    return {
        "embedding_models": [
            "ArcFace (ResNet100)",
            "FaceNet (Inception-ResNet-v1)",
            "MobileFaceNet"
        ],
        "liveness_models": [
            "CNN Liveness (ResNet18)",
            "Temporal LSTM",
            "MiDaS Depth Estimator"
        ],
        "fusion": "Learned MLP Fusion",
        "detection": "Hybrid (RetinaFace + MTCNN)"
    }


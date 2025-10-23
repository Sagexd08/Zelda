
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

class RegisterRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")

class RegisterResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    samples_processed: Optional[int] = None
    valid_samples: Optional[int] = None
    avg_quality_score: Optional[float] = None
    avg_liveness_score: Optional[float] = None
    error: Optional[str] = None

class AuthenticateRequest(BaseModel):
    user_id: str = Field(..., description="User identifier to authenticate")

class AuthenticateResponse(BaseModel):
    authenticated: bool
    confidence: float
    threshold: Optional[float] = None
    liveness_score: Optional[float] = None
    reason: str
    similarities: Optional[dict] = None

class IdentifyResponse(BaseModel):
    found: bool
    liveness_score: Optional[float] = None
    matches: List[dict]
    total_users_checked: Optional[int] = None
    reason: Optional[str] = None

class DeleteUserRequest(BaseModel):
    user_id: str = Field(..., description="User identifier to delete")

class ChallengeResponse(BaseModel):
    challenge_id: str
    challenge_type: str
    instructions: str
    expires_at: str

def decode_image(file: UploadFile) -> np.ndarray:
    try:
        contents = file.file.read()

        nparr = np.frombuffer(contents, np.uint8)

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

@router.post("/register", response_model=RegisterResponse)
async def register_user(
    user_id: str,
    images: List[UploadFile] = File(..., description="Multiple face images for registration"),
    db: Session = Depends(get_db)
):
    registration_service = get_registration_service()

    face_images = []
    for img_file in images:
        try:
            image = decode_image(img_file)
            face_images.append(image)
        except HTTPException:
            continue

    if not face_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images provided"
        )

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

@router.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate_user(
    user_id: str,
    image: UploadFile = File(..., description="Face image for authentication"),
    db: Session = Depends(get_db)
):
    authentication_service = get_authentication_service()

    face_image = decode_image(image)

    authenticated, result = authentication_service.authenticate(db, user_id, face_image)

    return AuthenticateResponse(
        authenticated=result['authenticated'],
        confidence=result['confidence'],
        threshold=result.get('threshold'),
        liveness_score=result.get('liveness_score'),
        reason=result['reason'],
        similarities=result.get('similarities')
    )

@router.post("/identify", response_model=IdentifyResponse)
async def identify_user(
    image: UploadFile = File(..., description="Unknown face image"),
    top_k: int = 3,
    db: Session = Depends(get_db)
):
    identification_service = get_identification_service()

    face_image = decode_image(image)

    found, result = identification_service.identify(db, face_image, top_k=top_k)

    return IdentifyResponse(
        found=result['found'],
        liveness_score=result.get('liveness_score'),
        matches=result['matches'],
        total_users_checked=result.get('total_users_checked'),
        reason=result.get('reason')
    )

@router.post("/delete_user")
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db)
):
    registration_service = get_registration_service()

    success, message = registration_service.delete_user(db, user_id)

    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )

@router.post("/challenge/create", response_model=ChallengeResponse)
async def create_challenge(user_id: Optional[str] = None):
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
    challenge_validator = get_challenge_validator()

    is_valid, confidence, message = challenge_validator.validate_challenge(
        challenge_id, sequence_data
    )

    return {
        "valid": is_valid,
        "confidence": confidence,
        "message": message
    }

@router.post("/adaptive/calibrate_threshold")
async def calibrate_threshold(
    user_id: str,
    db: Session = Depends(get_db)
):
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

@router.get("/system/info")
async def system_info():
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

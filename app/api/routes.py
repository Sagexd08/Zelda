
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Form, Query
from pydantic import BaseModel, Field
import numpy as np
import cv2
from io import BytesIO

from app.core.database import get_db, Session
from app.core.config import settings
from typing import Optional
from fastapi import Depends
from app.services.registration_service import get_registration_service
from app.services.authentication_service import get_authentication_service
from app.services.identification_service import get_identification_service
from app.services.adaptive_learning_service import get_adaptive_learning_service
from app.utils.challenge import get_challenge_validator
from app.utils.validation import validate_user_id

router = APIRouter()

# Conditional database dependency
def get_db_or_none():
    """Returns db session if not using Supabase, None otherwise"""
    if settings.USE_SUPABASE:
        return None
    else:
        return next(get_db())

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
    contents = None
    try:
        # Read file contents
        contents = file.file.read()
        
        # Reset file pointer for potential re-read
        file.file.seek(0)

        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file received"
            )

        if len(contents) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded image exceeds limit of {settings.MAX_UPLOAD_BYTES // (1024 * 1024)}MB"
            )

        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image. File may not be a valid image format.")

        if image.size == 0:
            raise ValueError("Decoded image is empty")

        return image

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Re-raise value errors with proper HTTP exception
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image file: {str(e)}"
        )
    finally:
        # Ensure file is closed
        if file.file:
            try:
                file.file.close()
            except:
                pass

@router.post("/register", response_model=RegisterResponse)
async def register_user(
    user_id: str = Form(...),
    images: List[UploadFile] = File(..., description="Multiple face images for registration"),
    db: Optional[Session] = Depends(get_db_or_none)
):
    try:
        user_id = validate_user_id(user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid user_id: {str(e)}"
        )
    
    if not images or len(images) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image is required for registration"
        )
    
    registration_service = get_registration_service()

    face_images = []
    decode_errors = []
    for idx, img_file in enumerate(images):
        try:
            image = decode_image(img_file)
            if image is not None and image.size > 0:
                face_images.append(image)
        except HTTPException as exc:
            decode_errors.append(f"Image {idx + 1}: {exc.detail}")
            continue
        except Exception as e:
            decode_errors.append(f"Image {idx + 1}: Failed to decode - {str(e)}")
            continue

    if not face_images:
        error_msg = "No valid images provided"
        if decode_errors:
            error_msg += f". Errors: {', '.join(decode_errors[:3])}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    try:
        success, result = registration_service.register_user(db, user_id, face_images)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration service error: {str(e)}"
        )

    if success:
        return RegisterResponse(
            success=True,
            user_id=result['user_id'],
            samples_processed=result['samples_processed'],
            valid_samples=result['valid_samples'],
            avg_quality_score=result['avg_quality_score'],
            avg_liveness_score=result['avg_liveness_score']
        )

    # Format error message from result dict
    error_message = result.get('error', 'Registration failed')
    
    # Build detailed error message
    if isinstance(result, dict):
        error_details = [error_message]
        if 'valid_count' in result and 'required' in result:
            error_details.append(f"Valid samples: {result['valid_count']}/{result['required']} required")
        if 'rejected_reasons' in result and result['rejected_reasons']:
            error_details.append(f"Rejected: {', '.join(result['rejected_reasons'][:3])}")
        error_message = ". ".join(error_details)
    
    status_code = status.HTTP_400_BAD_REQUEST
    if error_message == 'User already exists' or 'already exists' in error_message.lower():
        status_code = status.HTTP_409_CONFLICT

    raise HTTPException(status_code=status_code, detail=error_message)

@router.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate_user(
    user_id: str = Form(...),
    image: UploadFile = File(..., description="Face image for authentication"),
    db: Optional[Session] = Depends(get_db_or_none)
):
    user_id = validate_user_id(user_id)
    authentication_service = get_authentication_service()

    face_image = decode_image(image)

    try:
        authenticated, result = authentication_service.authenticate(db, user_id, face_image)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if result['reason'] == 'User not found':
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if result['reason'] == 'User account inactive':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account inactive")

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
    top_k: int = Query(3, ge=1, le=10),
    db: Optional[Session] = Depends(get_db_or_none)
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
    user_id: str = Form(...),
    db: Optional[Session] = Depends(get_db_or_none)
):
    user_id = validate_user_id(user_id)
    registration_service = get_registration_service()

    success, message = registration_service.delete_user(db, user_id)

    if success:
        return {"success": True, "message": message}
    else:
        status_code = status.HTTP_404_NOT_FOUND if message == "User not found" else status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status_code,
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

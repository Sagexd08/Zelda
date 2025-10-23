"""
Authentication Service
Face verification with liveness detection and adaptive thresholding
"""

from typing import Tuple, Dict, Optional
import numpy as np
from datetime import datetime

from app.core.database import Session, User, Embedding, AuditLog
from app.core.config import settings
from app.core.security import security_manager
from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
from app.models.embedding_extractor import get_embedding_extractor, cosine_similarity
from app.models.fusion_model import get_fusion_model
from app.models.liveness_detector import get_liveness_detector
from app.models.depth_estimator import get_depth_estimator
from app.utils.preprocessing import assess_image_quality, is_image_acceptable


class AuthenticationService:
    """
    Service for user authentication and verification.
    """
    
    def __init__(self):
        """Initialize authentication service"""
        self.face_detector = get_face_detector()
        self.face_aligner = get_face_aligner()
        self.embedding_extractor = get_embedding_extractor()
        self.fusion_model = get_fusion_model()
        self.liveness_detector = get_liveness_detector()
        self.depth_estimator = get_depth_estimator()
    
    def authenticate(
        self,
        db: Session,
        user_id: str,
        image: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Authenticate a user with face image.
        
        Args:
            db: Database session
            user_id: User identifier
            image: Face image (BGR format)
            
        Returns:
            Tuple[bool, Dict]: (authenticated, result_details)
        """
        start_time = datetime.utcnow()
        
        # Get user from database
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            self._log_authentication(db, None, False, 0.0, 0.0, "user_not_found", start_time)
            return False, {
                'authenticated': False,
                'reason': 'User not found',
                'confidence': 0.0
            }
        
        if not user.is_active:
            self._log_authentication(db, user.id, False, 0.0, 0.0, "user_inactive", start_time)
            return False, {
                'authenticated': False,
                'reason': 'User account inactive',
                'confidence': 0.0
            }
        
        # Detect face
        detection = self.face_detector.detect_largest(image)
        
        if detection is None:
            self._log_authentication(db, user.id, False, 0.0, 0.0, "no_face_detected", start_time)
            return False, {
                'authenticated': False,
                'reason': 'No face detected',
                'confidence': 0.0
            }
        
        # Check image quality
        quality_metrics = assess_image_quality(image)
        is_acceptable, quality_reason = is_image_acceptable(quality_metrics)
        
        if not is_acceptable:
            self._log_authentication(db, user.id, False, 0.0, 0.0, f"quality:{quality_reason}", start_time)
            return False, {
                'authenticated': False,
                'reason': f'Image quality: {quality_reason}',
                'confidence': 0.0
            }
        
        # Liveness detection
        face_region = image[
            int(detection.bbox[1]):int(detection.bbox[3]),
            int(detection.bbox[0]):int(detection.bbox[2])
        ]
        is_live, liveness_score = self.liveness_detector.predict(face_region)
        
        if not is_live:
            self._log_authentication(db, user.id, False, 0.0, liveness_score, "liveness_failed", start_time)
            return False, {
                'authenticated': False,
                'reason': 'Liveness check failed',
                'confidence': 0.0,
                'liveness_score': liveness_score
            }
        
        # Optional: Depth analysis
        if settings.ENABLE_DEPTH_ESTIMATION:
            depth_live, depth_conf, depth_analysis = self.depth_estimator.analyze_depth_for_liveness(
                image, detection.bbox
            )
            if not depth_live and depth_conf > 0.7:
                self._log_authentication(db, user.id, False, 0.0, liveness_score, "depth_failed", start_time)
                return False, {
                    'authenticated': False,
                    'reason': 'Depth analysis failed (possible 2D spoof)',
                    'confidence': 0.0,
                    'liveness_score': liveness_score
                }
        
        # Align face
        face_160 = self.face_aligner.align(image, detection, output_size=160)
        face_224 = self.face_aligner.align(image, detection, output_size=224)
        
        # Extract embeddings
        test_embeddings = self.embedding_extractor.extract_all_embeddings(face_160, face_224)
        
        if not test_embeddings:
            self._log_authentication(db, user.id, False, 0.0, liveness_score, "embedding_failed", start_time)
            return False, {
                'authenticated': False,
                'reason': 'Failed to extract face embeddings',
                'confidence': 0.0
            }
        
        # Compute fusion
        test_fusion = self.fusion_model.fuse_embeddings(test_embeddings)
        test_embeddings['fusion'] = test_fusion
        
        # Get stored embeddings
        stored_embedding = db.query(Embedding).filter(
            Embedding.user_id == user.id,
            Embedding.is_primary == True
        ).first()
        
        if not stored_embedding:
            self._log_authentication(db, user.id, False, 0.0, liveness_score, "no_stored_embedding", start_time)
            return False, {
                'authenticated': False,
                'reason': 'No enrolled embeddings found',
                'confidence': 0.0
            }
        
        # Decrypt stored embeddings
        stored_embeddings = self._decrypt_embeddings(stored_embedding)
        
        # Compute similarities
        similarities = {}
        for model_name in test_embeddings.keys():
            if model_name in stored_embeddings:
                sim = cosine_similarity(test_embeddings[model_name], stored_embeddings[model_name])
                similarities[model_name] = sim
        
        # Use fusion embedding for final decision
        if 'fusion' in similarities:
            final_similarity = similarities['fusion']
        else:
            # Fallback to average
            final_similarity = np.mean(list(similarities.values()))
        
        # Get threshold (adaptive or default)
        threshold = self._get_threshold(user)
        
        # Authentication decision
        authenticated = final_similarity >= threshold
        
        # Log authentication
        self._log_authentication(
            db, user.id, authenticated, final_similarity, liveness_score,
            "success" if authenticated else "similarity_below_threshold", start_time
        )
        
        # Update user statistics
        user.total_authentications += 1
        if authenticated:
            user.successful_authentications += 1
        else:
            user.failed_attempts += 1
        user.last_authentication = datetime.utcnow()
        db.commit()
        
        return authenticated, {
            'authenticated': authenticated,
            'confidence': float(final_similarity),
            'threshold': float(threshold),
            'liveness_score': float(liveness_score),
            'reason': 'live_match' if authenticated else 'similarity_below_threshold',
            'similarities': {k: float(v) for k, v in similarities.items()}
        }
    
    def _decrypt_embeddings(self, embedding_record: Embedding) -> Dict[str, np.ndarray]:
        """
        Decrypt stored embeddings.
        
        Args:
            embedding_record: Database embedding record
            
        Returns:
            Dict[str, np.ndarray]: Decrypted embeddings
        """
        embeddings = {}
        
        if embedding_record.arcface_embedding:
            embeddings['arcface'] = security_manager.decrypt_embedding(
                embedding_record.arcface_embedding
            )
        
        if embedding_record.facenet_embedding:
            embeddings['facenet'] = security_manager.decrypt_embedding(
                embedding_record.facenet_embedding
            )
        
        if embedding_record.mobilefacenet_embedding:
            embeddings['mobilefacenet'] = security_manager.decrypt_embedding(
                embedding_record.mobilefacenet_embedding
            )
        
        if embedding_record.fusion_embedding:
            embeddings['fusion'] = security_manager.decrypt_embedding(
                embedding_record.fusion_embedding
            )
        
        return embeddings
    
    def _get_threshold(self, user: User) -> float:
        """
        Get verification threshold for user (adaptive or default).
        
        Args:
            user: User object
            
        Returns:
            float: Threshold value
        """
        if settings.ENABLE_ADAPTIVE_THRESHOLD and user.custom_threshold is not None:
            # Use adaptive threshold with bounds
            threshold = np.clip(
                user.custom_threshold,
                settings.ADAPTIVE_THRESHOLD_MIN,
                settings.ADAPTIVE_THRESHOLD_MAX
            )
        else:
            threshold = settings.VERIFICATION_THRESHOLD
        
        return float(threshold)
    
    def _log_authentication(
        self,
        db: Session,
        user_id: Optional[int],
        success: bool,
        confidence: float,
        liveness_score: float,
        reason: str,
        start_time: datetime
    ):
        """
        Log authentication attempt.
        
        Args:
            db: Database session
            user_id: User ID
            success: Authentication result
            confidence: Confidence score
            liveness_score: Liveness score
            reason: Result reason
            start_time: Request start time
        """
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        audit_log = AuditLog(
            user_id=user_id,
            event_type='authenticate',
            event_result='success' if success else 'failure',
            confidence_score=confidence,
            liveness_score=liveness_score,
            failure_reason=reason if not success else None,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
        db.add(audit_log)
        db.commit()


# Singleton instance
_authentication_service_instance: Optional[AuthenticationService] = None


def get_authentication_service() -> AuthenticationService:
    """
    Get singleton authentication service instance.
    
    Returns:
        AuthenticationService: Authentication service
    """
    global _authentication_service_instance
    if _authentication_service_instance is None:
        _authentication_service_instance = AuthenticationService()
    return _authentication_service_instance


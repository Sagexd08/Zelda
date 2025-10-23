"""
Registration Service
Handle user enrollment with multi-sample face registration
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime

from app.core.database import Session, User, Embedding, LivenessSignature
from app.core.config import settings
from app.core.security import security_manager
from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
from app.models.embedding_extractor import get_embedding_extractor
from app.models.fusion_model import get_fusion_model
from app.models.liveness_detector import get_liveness_detector
from app.utils.preprocessing import assess_image_quality, is_image_acceptable, estimate_head_pose_simple, is_pose_acceptable


class RegistrationService:
    """
    Service for user registration and enrollment.
    """
    
    def __init__(self):
        """Initialize registration service"""
        self.face_detector = get_face_detector()
        self.face_aligner = get_face_aligner()
        self.embedding_extractor = get_embedding_extractor()
        self.fusion_model = get_fusion_model()
        self.liveness_detector = get_liveness_detector()
    
    def register_user(
        self,
        db: Session,
        user_id: str,
        face_images: List[np.ndarray]
    ) -> Tuple[bool, Dict]:
        """
        Register a new user with multiple face samples.
        
        Args:
            db: Database session
            user_id: Unique user identifier
            face_images: List of face images (BGR format)
            
        Returns:
            Tuple[bool, Dict]: (success, result_details)
        """
        # Validate number of samples
        if len(face_images) < settings.MIN_REGISTRATION_SAMPLES:
            return False, {
                'error': f'Minimum {settings.MIN_REGISTRATION_SAMPLES} images required',
                'provided': len(face_images)
            }
        
        if len(face_images) > settings.MAX_REGISTRATION_SAMPLES:
            face_images = face_images[:settings.MAX_REGISTRATION_SAMPLES]
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.user_id == user_id).first()
        if existing_user:
            return False, {'error': 'User already exists'}
        
        # Process each image
        valid_embeddings = []
        quality_scores = []
        liveness_scores = []
        rejected_reasons = []
        
        for idx, image in enumerate(face_images):
            result = self._process_registration_image(image, idx)
            
            if result['success']:
                valid_embeddings.append(result['embeddings'])
                quality_scores.append(result['quality_score'])
                liveness_scores.append(result['liveness_score'])
            else:
                rejected_reasons.append(f"Image {idx}: {result['reason']}")
        
        # Check if enough valid samples
        if len(valid_embeddings) < settings.MIN_REGISTRATION_SAMPLES:
            return False, {
                'error': 'Not enough valid face samples',
                'valid_count': len(valid_embeddings),
                'required': settings.MIN_REGISTRATION_SAMPLES,
                'rejected_reasons': rejected_reasons
            }
        
        # Compute mean embeddings
        mean_embeddings = self._compute_mean_embeddings(valid_embeddings)
        
        # Store in database
        try:
            # Create user
            new_user = User(
                user_id=user_id,
                created_at=datetime.utcnow(),
                is_active=True
            )
            db.add(new_user)
            db.flush()  # Get user.id
            
            # Store embeddings
            embedding_record = Embedding(
                user_id=new_user.id,
                arcface_embedding=security_manager.encrypt_embedding(mean_embeddings['arcface']) if 'arcface' in mean_embeddings else None,
                facenet_embedding=security_manager.encrypt_embedding(mean_embeddings['facenet']) if 'facenet' in mean_embeddings else None,
                mobilefacenet_embedding=security_manager.encrypt_embedding(mean_embeddings['mobilefacenet']) if 'mobilefacenet' in mean_embeddings else None,
                fusion_embedding=security_manager.encrypt_embedding(mean_embeddings['fusion']) if 'fusion' in mean_embeddings else None,
                quality_score=float(np.mean(quality_scores)),
                is_primary=True,
                created_at=datetime.utcnow()
            )
            db.add(embedding_record)
            
            # Store liveness signature (placeholder)
            liveness_sig = LivenessSignature(
                user_id=new_user.id,
                sample_count=len(valid_embeddings),
                confidence_score=float(np.mean(liveness_scores)),
                created_at=datetime.utcnow()
            )
            db.add(liveness_sig)
            
            db.commit()
            
            return True, {
                'user_id': user_id,
                'samples_processed': len(face_images),
                'valid_samples': len(valid_embeddings),
                'avg_quality_score': float(np.mean(quality_scores)),
                'avg_liveness_score': float(np.mean(liveness_scores)),
                'rejected_count': len(rejected_reasons),
                'rejected_reasons': rejected_reasons
            }
        
        except Exception as e:
            db.rollback()
            return False, {'error': f'Database error: {str(e)}'}
    
    def _process_registration_image(
        self,
        image: np.ndarray,
        image_idx: int
    ) -> Dict:
        """
        Process a single registration image.
        
        Args:
            image: Face image
            image_idx: Image index
            
        Returns:
            Dict: Processing result
        """
        # Detect face
        detection = self.face_detector.detect_largest(image)
        
        if detection is None:
            return {
                'success': False,
                'reason': 'No face detected'
            }
        
        # Check confidence
        if detection.confidence < settings.DETECTION_CONFIDENCE_THRESHOLD:
            return {
                'success': False,
                'reason': f'Low detection confidence: {detection.confidence:.2f}'
            }
        
        # Assess image quality
        quality_metrics = assess_image_quality(image)
        is_acceptable, quality_reason = is_image_acceptable(quality_metrics)
        
        if not is_acceptable:
            return {
                'success': False,
                'reason': quality_reason
            }
        
        # Check pose
        pose = estimate_head_pose_simple(detection.landmarks)
        pose_acceptable, pose_reason = is_pose_acceptable(pose)
        
        if not pose_acceptable:
            return {
                'success': False,
                'reason': pose_reason
            }
        
        # Check liveness
        face_region = image[
            int(detection.bbox[1]):int(detection.bbox[3]),
            int(detection.bbox[0]):int(detection.bbox[2])
        ]
        is_live, liveness_score = self.liveness_detector.predict(face_region)
        
        if not is_live:
            return {
                'success': False,
                'reason': f'Liveness check failed: {liveness_score:.2f}'
            }
        
        # Align face
        face_160 = self.face_aligner.align(image, detection, output_size=160)
        face_224 = self.face_aligner.align(image, detection, output_size=224)
        
        # Extract embeddings
        embeddings = self.embedding_extractor.extract_all_embeddings(face_160, face_224)
        
        if not embeddings:
            return {
                'success': False,
                'reason': 'Failed to extract embeddings'
            }
        
        # Compute fusion
        fusion_embedding = self.fusion_model.fuse_embeddings(embeddings)
        embeddings['fusion'] = fusion_embedding
        
        return {
            'success': True,
            'embeddings': embeddings,
            'quality_score': quality_metrics['overall_quality'],
            'liveness_score': liveness_score
        }
    
    def _compute_mean_embeddings(
        self,
        embeddings_list: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute mean embeddings from multiple samples with outlier removal.
        
        Args:
            embeddings_list: List of embedding dictionaries
            
        Returns:
            Dict[str, np.ndarray]: Mean embeddings
        """
        # Group embeddings by model
        grouped = {}
        for emb_dict in embeddings_list:
            for model_name, embedding in emb_dict.items():
                if model_name not in grouped:
                    grouped[model_name] = []
                grouped[model_name].append(embedding)
        
        # Compute mean with outlier removal
        mean_embeddings = {}
        for model_name, embeddings in grouped.items():
            embeddings_array = np.array(embeddings)
            
            # Remove outliers using z-score
            if len(embeddings_array) > 3:
                mean = np.mean(embeddings_array, axis=0)
                std = np.std(embeddings_array, axis=0)
                z_scores = np.abs((embeddings_array - mean) / (std + 1e-8))
                
                # Keep samples with z-score < 2
                mask = np.all(z_scores < 2, axis=1)
                filtered_embeddings = embeddings_array[mask]
                
                if len(filtered_embeddings) > 0:
                    mean_embedding = np.mean(filtered_embeddings, axis=0)
                else:
                    mean_embedding = mean
            else:
                mean_embedding = np.mean(embeddings_array, axis=0)
            
            # L2 normalize
            mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)
            
            mean_embeddings[model_name] = mean_embedding
        
        return mean_embeddings
    
    def delete_user(
        self,
        db: Session,
        user_id: str
    ) -> Tuple[bool, str]:
        """
        Delete user and all associated data (GDPR compliance).
        
        Args:
            db: Database session
            user_id: User identifier
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return False, "User not found"
            
            # Delete user (cascades to embeddings, liveness, audit logs)
            db.delete(user)
            db.commit()
            
            return True, f"User {user_id} deleted successfully"
        
        except Exception as e:
            db.rollback()
            return False, f"Deletion failed: {str(e)}"


# Singleton instance
_registration_service_instance: Optional[RegistrationService] = None


def get_registration_service() -> RegistrationService:
    """
    Get singleton registration service instance.
    
    Returns:
        RegistrationService: Registration service
    """
    global _registration_service_instance
    if _registration_service_instance is None:
        _registration_service_instance = RegistrationService()
    return _registration_service_instance


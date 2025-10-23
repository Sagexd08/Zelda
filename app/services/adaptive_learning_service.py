"""
Adaptive Learning Service
Online learning and threshold calibration
"""

from typing import Dict, Optional
import numpy as np
from datetime import datetime

from app.core.database import Session, User, Embedding
from app.core.config import settings
from app.core.security import security_manager


class AdaptiveLearningService:
    """
    Service for adaptive learning and threshold optimization.
    """
    
    def __init__(self):
        """Initialize adaptive learning service"""
        pass
    
    def update_embedding(
        self,
        db: Session,
        user_id: str,
        new_embeddings: Dict[str, np.ndarray],
        similarity_score: float
    ) -> bool:
        """
        Update user embedding with online learning (exponential moving average).
        
        Args:
            db: Database session
            user_id: User identifier
            new_embeddings: New embeddings from successful authentication
            similarity_score: Similarity score of authentication
            
        Returns:
            bool: Success status
        """
        if not settings.ENABLE_ONLINE_LEARNING:
            return False
        
        # Get user
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return False
        
        # Get primary embedding
        embedding_record = db.query(Embedding).filter(
            Embedding.user_id == user.id,
            Embedding.is_primary == True
        ).first()
        
        if not embedding_record:
            return False
        
        # Check update limit
        if embedding_record.update_count >= settings.MAX_EMBEDDING_UPDATES:
            return False
        
        # Only update if similarity is high (confident authentication)
        if similarity_score < 0.85:
            return False
        
        # Decrypt stored embeddings
        stored_embeddings = self._decrypt_embeddings(embedding_record)
        
        # Update each embedding with EMA
        alpha = settings.EMBEDDING_UPDATE_ALPHA
        
        for model_name in new_embeddings.keys():
            if model_name in stored_embeddings:
                # Exponential moving average: new = alpha * current + (1-alpha) * stored
                updated_embedding = (
                    alpha * new_embeddings[model_name] + 
                    (1 - alpha) * stored_embeddings[model_name]
                )
                
                # L2 normalize
                updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-8)
                
                # Encrypt and store
                if model_name == 'arcface':
                    embedding_record.arcface_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'facenet':
                    embedding_record.facenet_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'mobilefacenet':
                    embedding_record.mobilefacenet_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'fusion':
                    embedding_record.fusion_embedding = security_manager.encrypt_embedding(updated_embedding)
        
        # Update metadata
        embedding_record.update_count += 1
        embedding_record.last_updated = datetime.utcnow()
        
        db.commit()
        
        return True
    
    def calibrate_threshold(
        self,
        db: Session,
        user_id: str,
        recent_scores: Optional[list] = None
    ) -> float:
        """
        Calibrate verification threshold for user using Bayesian approach.
        
        Args:
            db: Database session
            user_id: User identifier
            recent_scores: Optional list of recent similarity scores
            
        Returns:
            float: Calibrated threshold
        """
        if not settings.ENABLE_ADAPTIVE_THRESHOLD:
            return settings.VERIFICATION_THRESHOLD
        
        # Get user
        user = db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            return settings.VERIFICATION_THRESHOLD
        
        # If not enough authentication history, use default
        if user.total_authentications < 10:
            return settings.VERIFICATION_THRESHOLD
        
        # Compute success rate
        success_rate = user.successful_authentications / user.total_authentications
        
        # Adjust threshold based on success rate
        # If success rate is high, can be slightly more strict
        # If success rate is low, be more lenient (but within bounds)
        
        base_threshold = settings.VERIFICATION_THRESHOLD
        
        if success_rate > 0.9:
            # High success rate - can be slightly stricter
            adjusted_threshold = base_threshold + 0.05
        elif success_rate < 0.7:
            # Low success rate - be more lenient
            adjusted_threshold = base_threshold - 0.05
        else:
            adjusted_threshold = base_threshold
        
        # Clip to bounds
        calibrated_threshold = np.clip(
            adjusted_threshold,
            settings.ADAPTIVE_THRESHOLD_MIN,
            settings.ADAPTIVE_THRESHOLD_MAX
        )
        
        # Update user threshold with confidence tracking
        confidence = min(1.0, user.total_authentications / 50.0)  # Confidence grows with data
        user.custom_threshold = calibrated_threshold
        user.threshold_confidence = confidence
        
        db.commit()
        
        return float(calibrated_threshold)
    
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


# Singleton instance
_adaptive_learning_service_instance: Optional[AdaptiveLearningService] = None


def get_adaptive_learning_service() -> AdaptiveLearningService:
    """
    Get singleton adaptive learning service instance.
    
    Returns:
        AdaptiveLearningService: Adaptive learning service
    """
    global _adaptive_learning_service_instance
    if _adaptive_learning_service_instance is None:
        _adaptive_learning_service_instance = AdaptiveLearningService()
    return _adaptive_learning_service_instance


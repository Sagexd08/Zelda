
from typing import Dict, Optional
import numpy as np
from datetime import datetime

from app.core.database import Session, User, Embedding
from app.core.config import settings
from app.core.security import security_manager

class AdaptiveLearningService:

    def __init__(self):
        pass

    def update_embedding(
        self,
        db: Session,
        user_id: str,
        new_embeddings: Dict[str, np.ndarray],
        similarity_score: float
    ) -> bool:
        if not settings.ENABLE_ONLINE_LEARNING:
            return False

        user = db.query(User).filter(User.user_id == user_id).first()

        if not user:
            return False

        embedding_record = db.query(Embedding).filter(
            Embedding.user_id == user.id,
            Embedding.is_primary == True
        ).first()

        if not embedding_record:
            return False

        if embedding_record.update_count >= settings.MAX_EMBEDDING_UPDATES:
            return False

        if similarity_score < 0.85:
            return False

        stored_embeddings = self._decrypt_embeddings(embedding_record)

        alpha = settings.EMBEDDING_UPDATE_ALPHA

        for model_name in new_embeddings.keys():
            if model_name in stored_embeddings:
                updated_embedding = (
                    alpha * new_embeddings[model_name] +
                    (1 - alpha) * stored_embeddings[model_name]
                )

                updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-8)

                if model_name == 'arcface':
                    embedding_record.arcface_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'facenet':
                    embedding_record.facenet_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'mobilefacenet':
                    embedding_record.mobilefacenet_embedding = security_manager.encrypt_embedding(updated_embedding)
                elif model_name == 'fusion':
                    embedding_record.fusion_embedding = security_manager.encrypt_embedding(updated_embedding)

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
        if not settings.ENABLE_ADAPTIVE_THRESHOLD:
            return settings.VERIFICATION_THRESHOLD

        user = db.query(User).filter(User.user_id == user_id).first()

        if not user:
            return settings.VERIFICATION_THRESHOLD

        if user.total_authentications < 10:
            return settings.VERIFICATION_THRESHOLD

        success_rate = user.successful_authentications / user.total_authentications

        base_threshold = settings.VERIFICATION_THRESHOLD

        if success_rate > 0.9:
            adjusted_threshold = base_threshold + 0.05
        elif success_rate < 0.7:
            adjusted_threshold = base_threshold - 0.05
        else:
            adjusted_threshold = base_threshold

        calibrated_threshold = np.clip(
            adjusted_threshold,
            settings.ADAPTIVE_THRESHOLD_MIN,
            settings.ADAPTIVE_THRESHOLD_MAX
        )

        confidence = min(1.0, user.total_authentications / 50.0)
        user.custom_threshold = calibrated_threshold
        user.threshold_confidence = confidence

        db.commit()

        return float(calibrated_threshold)

    def _decrypt_embeddings(self, embedding_record: Embedding) -> Dict[str, np.ndarray]:
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

_adaptive_learning_service_instance: Optional[AdaptiveLearningService] = None

def get_adaptive_learning_service() -> AdaptiveLearningService:
    global _adaptive_learning_service_instance
    if _adaptive_learning_service_instance is None:
        _adaptive_learning_service_instance = AdaptiveLearningService()
    return _adaptive_learning_service_instance

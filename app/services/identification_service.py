
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime

from app.core.database import Session, User, Embedding
from app.core.config import settings
from app.core.security import security_manager
from app.models.face_detector import get_face_detector
from app.models.face_aligner import get_face_aligner
from app.models.embedding_extractor import get_embedding_extractor, cosine_similarity
from app.models.fusion_model import get_fusion_model
from app.models.liveness_detector import get_liveness_detector

class IdentificationService:

    def __init__(self):
        self.face_detector = get_face_detector()
        self.face_aligner = get_face_aligner()
        self.embedding_extractor = get_embedding_extractor()
        self.fusion_model = get_fusion_model()
        self.liveness_detector = get_liveness_detector()

    def identify(
        self,
        db: Session,
        image: np.ndarray,
        top_k: int = 3,
        require_liveness: bool = True
    ) -> Tuple[bool, Dict]:
        detection = self.face_detector.detect_largest(image)

        if detection is None:
            return False, {
                'found': False,
                'reason': 'No face detected',
                'matches': []
            }

        if require_liveness:
            face_region = image[
                int(detection.bbox[1]):int(detection.bbox[3]),
                int(detection.bbox[0]):int(detection.bbox[2])
            ]
            is_live, liveness_score = self.liveness_detector.predict(face_region)

            if not is_live:
                return False, {
                    'found': False,
                    'reason': 'Liveness check failed',
                    'liveness_score': liveness_score,
                    'matches': []
                }
        else:
            liveness_score = 1.0

        face_160 = self.face_aligner.align(image, detection, output_size=160)
        face_224 = self.face_aligner.align(image, detection, output_size=224)

        test_embeddings = self.embedding_extractor.extract_all_embeddings(face_160, face_224)

        if not test_embeddings:
            return False, {
                'found': False,
                'reason': 'Failed to extract embeddings',
                'matches': []
            }

        test_fusion = self.fusion_model.fuse_embeddings(test_embeddings)
        test_embeddings['fusion'] = test_fusion

        users = db.query(User).filter(User.is_active == True).all()

        if not users:
            return False, {
                'found': False,
                'reason': 'No enrolled users',
                'matches': []
            }

        matches = []

        for user in users:
            stored_embedding = db.query(Embedding).filter(
                Embedding.user_id == user.id,
                Embedding.is_primary == True
            ).first()

            if not stored_embedding:
                continue

            stored_embeddings = self._decrypt_embeddings(stored_embedding)

            if 'fusion' in stored_embeddings:
                similarity = cosine_similarity(
                    test_embeddings['fusion'],
                    stored_embeddings['fusion']
                )
            else:
                sims = []
                for model_name in test_embeddings.keys():
                    if model_name != 'fusion' and model_name in stored_embeddings:
                        sim = cosine_similarity(
                            test_embeddings[model_name],
                            stored_embeddings[model_name]
                        )
                        sims.append(sim)
                similarity = np.mean(sims) if sims else 0.0

            matches.append({
                'user_id': user.user_id,
                'confidence': float(similarity)
            })

        matches.sort(key=lambda x: x['confidence'], reverse=True)

        top_matches = matches[:top_k]

        found = False
        if top_matches and top_matches[0]['confidence'] >= settings.IDENTIFICATION_THRESHOLD:
            found = True

        return found, {
            'found': found,
            'liveness_score': liveness_score,
            'matches': top_matches,
            'total_users_checked': len(users)
        }

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

_identification_service_instance: Optional[IdentificationService] = None

def get_identification_service() -> IdentificationService:
    global _identification_service_instance
    if _identification_service_instance is None:
        _identification_service_instance = IdentificationService()
    return _identification_service_instance

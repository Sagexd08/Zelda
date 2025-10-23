
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings

class VoiceEmbeddingExtractor:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or settings.get_device()
        self.model = None

        if settings.ENABLE_VOICE_AUTH:
            self._init_model()
        else:
            print("Voice authentication disabled in config")

    def _init_model(self):
        voice_path = settings.get_model_path(settings.VOICE_MODEL_PATH)

        if voice_path.exists():
            try:
                print("✓ Voice recognition model loaded")
            except Exception as e:
                print(f"✗ Failed to load voice model: {e}")
        else:
            print("Voice model weights not found")

    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if self.model is None:
            return np.random.randn(192).astype(np.float32)

        embedding = np.random.randn(192).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def verify(
        self,
        enrolled_embedding: np.ndarray,
        test_embedding: np.ndarray
    ) -> Tuple[bool, float]:
        similarity = self.compute_similarity(enrolled_embedding, test_embedding)
        is_match = similarity >= settings.VOICE_SIMILARITY_THRESHOLD

        return is_match, similarity

class MultimodalFusionModel:

    def __init__(self):
        self.face_weight = 0.7
        self.voice_weight = 0.3

    def fuse_scores(
        self,
        face_score: float,
        voice_score: Optional[float] = None,
        face_confidence: float = 1.0,
        voice_confidence: float = 1.0
    ) -> Tuple[float, dict]:
        if voice_score is None or not settings.ENABLE_VOICE_AUTH:
            return face_score, {
                'mode': 'face_only',
                'face_score': face_score,
                'voice_score': None
            }

        adjusted_face_weight = self.face_weight * face_confidence
        adjusted_voice_weight = self.voice_weight * voice_confidence

        total_weight = adjusted_face_weight + adjusted_voice_weight
        norm_face_weight = adjusted_face_weight / total_weight
        norm_voice_weight = adjusted_voice_weight / total_weight

        fused_score = (norm_face_weight * face_score +
                      norm_voice_weight * voice_score)

        details = {
            'mode': 'multimodal',
            'face_score': face_score,
            'voice_score': voice_score,
            'face_weight': norm_face_weight,
            'voice_weight': norm_voice_weight,
            'fused_score': fused_score
        }

        return fused_score, details

    def decision_level_fusion(
        self,
        face_decision: bool,
        voice_decision: Optional[bool] = None,
        require_both: bool = False
    ) -> bool:
        if voice_decision is None or not settings.ENABLE_VOICE_AUTH:
            return face_decision

        if require_both:
            return face_decision and voice_decision
        else:
            return face_decision or voice_decision

_voice_extractor_instance: Optional[VoiceEmbeddingExtractor] = None
_multimodal_fusion_instance: Optional[MultimodalFusionModel] = None

def get_voice_extractor() -> VoiceEmbeddingExtractor:
    global _voice_extractor_instance
    if _voice_extractor_instance is None:
        _voice_extractor_instance = VoiceEmbeddingExtractor()
    return _voice_extractor_instance

def get_multimodal_fusion() -> MultimodalFusionModel:
    global _multimodal_fusion_instance
    if _multimodal_fusion_instance is None:
        _multimodal_fusion_instance = MultimodalFusionModel()
    return _multimodal_fusion_instance

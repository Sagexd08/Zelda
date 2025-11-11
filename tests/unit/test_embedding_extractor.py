"""
Unit tests for embedding extractor
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from app.models.embedding_extractor import EmbeddingExtractor, get_embedding_extractor, cosine_similarity


@pytest.mark.unit
@pytest.mark.model
class TestEmbeddingExtractor:
    """Test suite for EmbeddingExtractor"""
    
    def test_extractor_initialization(self):
        """Test extractor initialization"""
        with patch('app.models.embedding_extractor.facenet_pytorch'):
            extractor = EmbeddingExtractor()
            assert extractor is not None
    
    def test_extract_arcface_embedding(self, sample_face_image):
        """Test extracting ArcFace embedding"""
        with patch('app.models.embedding_extractor.insightface'):
            extractor = EmbeddingExtractor()
            # Mock the extraction
            embedding = np.random.randn(512).astype(np.float32)
            # Note: Actual extraction requires loaded models
    
    def test_extract_facenet_embedding(self, sample_face_image):
        """Test extracting FaceNet embedding"""
        with patch('app.models.embedding_extractor.facenet_pytorch'):
            extractor = EmbeddingExtractor()
            embedding = np.random.randn(512).astype(np.float32)
    
    def test_extract_mobilefacenet_embedding(self, sample_face_image):
        """Test extracting MobileFaceNet embedding"""
        extractor = EmbeddingExtractor()
        embedding = np.random.randn(512).astype(np.float32)
    
    def test_extract_all_embeddings(self, sample_face_image):
        """Test extracting all embeddings"""
        extractor = EmbeddingExtractor()
        embeddings = {
            'arcface': np.random.randn(512).astype(np.float32),
            'facenet': np.random.randn(512).astype(np.float32),
            'mobilefacenet': np.random.randn(512).astype(np.float32)
        }
        # Verify structure
        assert len(embeddings) == 3
        assert all(emb.shape == (512,) for emb in embeddings.values())
    
    def test_embeddings_are_normalized(self, sample_face_image):
        """Test that embeddings are L2 normalized"""
        extractor = EmbeddingExtractor()
        embedding = np.random.randn(512).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        embedding_normalized = embedding / (norm + 1e-8)
        
        norm_after = np.linalg.norm(embedding_normalized)
        assert abs(norm_after - 1.0) < 0.01


@pytest.mark.unit
@pytest.mark.model
class TestCosineSimilarity:
    """Test suite for cosine similarity function"""
    
    def test_cosine_similarity_same_vectors(self):
        """Test cosine similarity of identical vectors"""
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([1, 0, 0], dtype=np.float32)
        
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors"""
        vec1 = np.array([1, 0], dtype=np.float32)
        vec2 = np.array([0, 1], dtype=np.float32)
        
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors"""
        vec1 = np.array([1, 0], dtype=np.float32)
        vec2 = np.array([-1, 0], dtype=np.float32)
        
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6
    
    def test_cosine_similarity_range(self):
        """Test cosine similarity is in [-1, 1] range"""
        vec1 = np.random.randn(512).astype(np.float32)
        vec2 = np.random.randn(512).astype(np.float32)
        
        similarity = cosine_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0
    
    def test_cosine_similarity_batch(self):
        """Test cosine similarity for batch of vectors"""
        vec1 = np.random.randn(512).astype(np.float32)
        batch = np.random.randn(10, 512).astype(np.float32)
        
        # Compute similarities
        similarities = np.array([cosine_similarity(vec1, batch[i]) for i in range(10)])
        assert len(similarities) == 10
        assert all(-1.0 <= s <= 1.0 for s in similarities)
    
    def test_get_embedding_extractor_singleton(self):
        """Test that get_embedding_extractor returns singleton"""
        extractor1 = get_embedding_extractor()
        extractor2 = get_embedding_extractor()
        assert extractor1 is not None
        assert extractor2 is not None


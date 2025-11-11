"""
Pytest fixtures and configuration
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Mock environment variables before imports
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("ENCRYPTION_KEY", "test-encryption-key-for-testing-only")

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings

# Test database
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine"""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create a test database session"""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture
def db_session(test_db):
    """Override get_db dependency with test session"""
    yield test_db

@pytest.fixture
def client(db_session):
    """Create a test client"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_face_image():
    """Generate a sample face image for testing"""
    # Create a simple synthetic face image
    image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    # Add some structure to make it more face-like
    cv2.rectangle(image, (60, 60), (100, 100), (200, 150, 100), -1)  # Face
    cv2.circle(image, (75, 75), 5, (0, 0, 0), -1)  # Eye
    cv2.circle(image, (85, 75), 5, (0, 0, 0), -1)  # Eye
    cv2.ellipse(image, (80, 90), (15, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    return image

@pytest.fixture
def sample_face_embedding():
    """Generate a sample face embedding"""
    return np.random.randn(512).astype(np.float32)

@pytest.fixture
def mock_device():
    """Mock device (CPU for testing)"""
    return torch.device("cpu")

@pytest.fixture
def mock_face_detector():
    """Mock face detector"""
    detector = Mock()
    detection = Mock()
    detection.box = [50, 50, 150, 150]
    detection.landmarks = np.array([[70, 70], [130, 70], [100, 100], [80, 130], [120, 130]])
    detection.confidence = 0.95
    detector.detect_largest.return_value = detection
    detector.detect_all.return_value = [detection]
    return detector

@pytest.fixture
def mock_face_aligner():
    """Mock face aligner"""
    aligner = Mock()
    aligned_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    aligner.align.return_value = aligned_image
    return aligner

@pytest.fixture
def mock_embedding_extractor():
    """Mock embedding extractor"""
    extractor = Mock()
    embedding = np.random.randn(512).astype(np.float32)
    extractor.extract_embedding.return_value = embedding
    extractor.extract_all_embeddings.return_value = {
        'arcface': embedding,
        'facenet': embedding,
        'mobilefacenet': embedding
    }
    return extractor

@pytest.fixture
def mock_liveness_detector():
    """Mock liveness detector"""
    detector = Mock()
    detector.predict.return_value = (True, 0.95)
    return detector

@pytest.fixture
def mock_fusion_model():
    """Mock fusion model"""
    model = Mock()
    fused_embedding = np.random.randn(512).astype(np.float32)
    model.fuse_embeddings.return_value = fused_embedding
    return model

@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_user():
    """Mock user data"""
    return {
        "user_id": "test_user_123",
        "username": "testuser",
        "email": "test@example.com"
    }

@pytest.fixture
def mock_registration_data(sample_face_image):
    """Mock registration data"""
    return {
        "user_id": "test_user_123",
        "images": [sample_face_image for _ in range(5)]
    }

@pytest.fixture
def sample_jwt_token():
    """Generate a sample JWT token"""
    from datetime import datetime, timedelta
    import jwt
    
    payload = {
        "user_id": "test_user_123",
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    yield
    # Any cleanup can go here

@pytest.fixture
def sample_weights_directory(temp_directory):
    """Create a mock weights directory with dummy model files"""
    weights_dir = Path(temp_directory) / "weights"
    weights_dir.mkdir()
    
    # Create dummy model files
    model_files = [
        "fusion_mlp.pth",
        "liveness_resnet18.pth",
    ]
    
    for model_file in model_files:
        dummy_content = b"dummy model weights"
        (weights_dir / model_file).write_bytes(dummy_content)
    
    return weights_dir

@pytest.fixture
def mock_torch_model():
    """Create a simple mock PyTorch model"""
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 128)
        
        def forward(self, x):
            return self.fc(x)
    
    return SimpleModel()

@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector"""
    collector = Mock()
    collector.increment = Mock()
    collector.observe = Mock()
    collector.gauge = Mock()
    return collector

@pytest.fixture
def sample_video_frames():
    """Generate sample video frames for temporal testing"""
    frames = []
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add slight variations
        frame[100 + i//2, 100 + i//2] = [255, 255, 255]
        frames.append(frame)
    return frames

@pytest.fixture
def mock_challenge_validator():
    """Mock challenge validator"""
    validator = Mock()
    validator.create_challenge.return_value = {
        "challenge_id": "test_challenge_123",
        "challenge_type": "blink",
        "instructions": "Please blink once",
        "expires_at": Mock()
    }
    validator.validate_challenge.return_value = (True, 0.95, "Challenge passed")
    return validator


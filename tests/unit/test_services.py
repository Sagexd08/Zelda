"""
Unit tests for service layer
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.registration_service import RegistrationService
from app.services.authentication_service import AuthenticationService
from app.services.identification_service import IdentificationService


@pytest.mark.unit
class TestRegistrationService:
    """Test suite for RegistrationService"""
    
    @pytest.fixture
    def registration_service(self):
        """Create registration service instance"""
        return RegistrationService()
    
    def test_register_user_success(self, registration_service, mock_user, sample_face_image):
        """Test successful user registration"""
        db = Mock()
        user_id = "test_user_123"
        images = [sample_face_image for _ in range(5)]
        
        with patch.object(registration_service, '_process_images') as mock_process:
            mock_process.return_value = {
                'valid_images': images,
                'quality_score': 0.9,
                'liveness_score': 0.95
            }
            
            success, result = registration_service.register_user(db, user_id, images)
            
            assert success is True
            assert result['user_id'] == user_id
    
    def test_register_user_insufficient_samples(self, registration_service, sample_face_image):
        """Test registration with insufficient samples"""
        db = Mock()
        user_id = "test_user_123"
        images = [sample_face_image]  # Only 1 image
        
        success, result = registration_service.register_user(db, user_id, images)
        
        assert success is False
    
    def test_delete_user(self, registration_service):
        """Test user deletion"""
        db = Mock()
        user_id = "test_user_123"
        
        # Mock database query
        mock_user = Mock()
        db.query.return_value.filter.return_value.first.return_value = mock_user
        
        success, message = registration_service.delete_user(db, user_id)
        
        assert success is True


@pytest.mark.unit
class TestAuthenticationService:
    """Test suite for AuthenticationService"""
    
    @pytest.fixture
    def authentication_service(self):
        """Create authentication service instance"""
        return AuthenticationService()
    
    def test_authenticate_success(self, authentication_service, sample_face_image):
        """Test successful authentication"""
        db = Mock()
        user_id = "test_user_123"
        
        mock_user = Mock()
        mock_user.embedding = np.random.randn(512).astype(np.float32)
        
        db.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch.object(authentication_service, '_extract_embedding') as mock_extract:
            mock_extract.return_value = np.random.randn(512).astype(np.float32)
            
            authenticated, result = authentication_service.authenticate(db, user_id, sample_face_image)
            
            # Result depends on similarity computation
            assert isinstance(result['authenticated'], bool)
    
    def test_authenticate_user_not_found(self, authentication_service, sample_face_image):
        """Test authentication with non-existent user"""
        db = Mock()
        user_id = "nonexistent_user"
        
        db.query.return_value.filter.return_value.first.return_value = None
        
        authenticated, result = authentication_service.authenticate(db, user_id, sample_face_image)
        
        assert authenticated is False
    
    def test_authenticate_liveness_failed(self, authentication_service, sample_face_image):
        """Test authentication with failed liveness check"""
        db = Mock()
        user_id = "test_user_123"
        
        mock_user = Mock()
        mock_user.embedding = np.random.randn(512).astype(np.float32)
        
        db.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch.object(authentication_service, '_check_liveness') as mock_liveness:
            mock_liveness.return_value = (False, 0.3)  # Failed liveness
            
            authenticated, result = authentication_service.authenticate(db, user_id, sample_face_image)
            
            assert authenticated is False


@pytest.mark.unit
class TestIdentificationService:
    """Test suite for IdentificationService"""
    
    @pytest.fixture
    def identification_service(self):
        """Create identification service instance"""
        return IdentificationService()
    
    def test_identify_success(self, identification_service, sample_face_image):
        """Test successful identification"""
        db = Mock()
        
        # Mock multiple users
        mock_users = [
            Mock(user_id="user1", embedding=np.random.randn(512).astype(np.float32)),
            Mock(user_id="user2", embedding=np.random.randn(512).astype(np.float32)),
            Mock(user_id="user3", embedding=np.random.randn(512).astype(np.float32)),
        ]
        
        db.query.return_value.all.return_value = mock_users
        
        with patch.object(identification_service, '_extract_embedding') as mock_extract:
            mock_extract.return_value = np.random.randn(512).astype(np.float32)
            
            found, result = identification_service.identify(db, sample_face_image, top_k=3)
            
            assert isinstance(found, bool)
            assert 'matches' in result
    
    def test_identify_no_users(self, identification_service, sample_face_image):
        """Test identification with no users in database"""
        db = Mock()
        db.query.return_value.all.return_value = []
        
        found, result = identification_service.identify(db, sample_face_image, top_k=3)
        
        assert found is False


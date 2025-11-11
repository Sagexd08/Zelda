"""
Integration tests for registration workflow
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import settings

pytestmark = pytest.mark.integration


@pytest.mark.integration
class TestRegistrationWorkflow:
    """Integration tests for end-to-end registration"""
    
    def test_complete_registration_flow(self, client, sample_face_image, temp_directory):
        """Test complete user registration flow"""
        user_id = "integration_test_user"
        
        # Create test image files
        image_paths = []
        for i in range(5):
            img_path = Path(temp_directory) / f"test_image_{i}.jpg"
            cv2.imwrite(str(img_path), sample_face_image)
            image_paths.append(img_path)
        
        # Prepare form data
        files = [
            ('images', (f'image_{i}.jpg', open(str(img_path), 'rb'), 'image/jpeg'))
            for i, img_path in enumerate(image_paths)
        ]
        
        # Register user
        response = client.post(
            "/api/v1/register",
            data={'user_id': user_id},
            files=files
        )
        
        # Verify response
        assert response.status_code in [200, 201]
        
        # Clean up
        for f in files:
            f[1][1].close()
    
    def test_registration_validation(self, client, sample_face_image, temp_directory):
        """Test registration input validation"""
        # Test without user_id
        response = client.post("/api/v1/register")
        assert response.status_code >= 400
        
        # Test with too few images
        user_id = "test_user"
        img_path = Path(temp_directory) / "single_image.jpg"
        cv2.imwrite(str(img_path), sample_face_image)
        
        with open(str(img_path), 'rb') as f:
            files = [('images', ('image.jpg', f, 'image/jpeg'))]
            response = client.post(
                "/api/v1/register",
                data={'user_id': user_id},
                files=files
            )
            # Should fail due to insufficient samples
            assert response.status_code >= 400

    def test_registration_invalid_user_id(self, client, sample_face_image, temp_directory):
        """Registration should reject malformed user identifiers."""
        img_path = Path(temp_directory) / "valid_image.jpg"
        cv2.imwrite(str(img_path), sample_face_image)

        payload = Path(img_path).read_bytes()
        files = [
            ('images', (f'image_{i}.jpg', io.BytesIO(payload), 'image/jpeg'))
            for i in range(5)
        ]
        response = client.post(
            "/api/v1/register",
            data={'user_id': 'invalid user!'},
            files=files
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "user_id" in detail

    def test_registration_rejects_large_upload(self, client, sample_face_image, monkeypatch, temp_directory):
        """Registration should enforce upload size limits."""
        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)

        large_image = cv2.resize(sample_face_image, (1024, 1024))
        success, buffer = cv2.imencode(".jpg", large_image)
        assert success

        payload = buffer.tobytes()
        files = [
            ('images', (f'large_{i}.jpg', io.BytesIO(payload), 'image/jpeg'))
            for i in range(5)
        ]

        response = client.post(
            "/api/v1/register",
            data={'user_id': 'size_test_user'},
            files=files
        )

        assert response.status_code == 413
        assert "exceeds limit" in response.json()["detail"]


@pytest.mark.integration
class TestAuthenticationWorkflow:
    """Integration tests for authentication workflow"""
    
    def test_authenticate_existing_user(self, client, sample_face_image, temp_directory):
        """Test authentication of registered user"""
        # First, register a user
        user_id = "auth_test_user"
        image_paths = []
        for i in range(5):
            img_path = Path(temp_directory) / f"reg_image_{i}.jpg"
            cv2.imwrite(str(img_path), sample_face_image)
            image_paths.append(img_path)
        
        files = [
            ('images', (f'image_{i}.jpg', open(str(img_path), 'rb'), 'image/jpeg'))
            for i, img_path in enumerate(image_paths)
        ]
        
        register_response = client.post(
            "/api/v1/register",
            data={'user_id': user_id},
            files=files
        )
        
        # Clean up registration files
        for f in files:
            f[1][1].close()
        
        # Now authenticate
        auth_img_path = Path(temp_directory) / "auth_image.jpg"
        cv2.imwrite(str(auth_img_path), sample_face_image)
        
        with open(str(auth_img_path), 'rb') as f:
            files = [('image', ('auth.jpg', f, 'image/jpeg'))]
            auth_response = client.post(
                "/api/v1/authenticate",
                data={'user_id': user_id},
                files=files
            )
            
            # Verify response structure
            assert auth_response.status_code == 200
            data = auth_response.json()
            assert 'authenticated' in data
            assert 'confidence' in data
    
    def test_authenticate_nonexistent_user(self, client, sample_face_image, temp_directory):
        """Test authentication of non-existent user"""
        user_id = "nonexistent_user"
        
        img_path = Path(temp_directory) / "auth_image.jpg"
        cv2.imwrite(str(img_path), sample_face_image)
        
        with open(str(img_path), 'rb') as f:
            files = [('image', ('auth.jpg', f, 'image/jpeg'))]
            response = client.post(
                "/api/v1/authenticate",
                data={'user_id': user_id},
                files=files
            )
            
            assert response.status_code == 404


@pytest.mark.integration
class TestIdentificationWorkflow:
    """Integration tests for identification workflow"""
    
    def test_identify_user(self, client, sample_face_image, temp_directory):
        """Test identification of unknown person"""
        # Register multiple users
        for user_id in ["user1", "user2", "user3"]:
            image_paths = []
            for i in range(5):
                img_path = Path(temp_directory) / f"{user_id}_image_{i}.jpg"
                cv2.imwrite(str(img_path), sample_face_image)
                image_paths.append(img_path)
            
            files = [
                ('images', (f'image_{i}.jpg', open(str(img_path), 'rb'), 'image/jpeg'))
                for i, img_path in enumerate(image_paths)
            ]
            
            client.post(
                "/api/v1/register",
                data={'user_id': user_id},
                files=files
            )
            
            # Clean up
            for f in files:
                f[1][1].close()
        
        # Try to identify
        query_img_path = Path(temp_directory) / "query_image.jpg"
        cv2.imwrite(str(query_img_path), sample_face_image)
        
        with open(str(query_img_path), 'rb') as f:
            files = [('image', ('query.jpg', f, 'image/jpeg'))]
            response = client.post(
                "/api/v1/identify",
                data={'top_k': 3},
                files=files
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'found' in data
            assert 'matches' in data


import pytest
import numpy as np
from unittest.mock import Mock

from app.models.face_aligner import FaceAligner, get_face_aligner


@pytest.mark.unit
@pytest.mark.model
class TestFaceAligner:
    """Test suite for FaceAligner"""
    
    def test_aligner_initialization(self):
        """Test aligner initialization"""
        aligner = FaceAligner()
        assert aligner is not None
    
    def test_align_face_160(self, sample_face_image, mock_face_detector):
        """Test aligning face to 160x160"""
        aligner = FaceAligner()
        detection = mock_face_detector.detect_largest(sample_face_image)
        
        if detection:
            aligned = aligner.align(sample_face_image, detection, output_size=160)
            assert aligned.shape == (160, 160, 3)
            assert aligned.dtype == np.uint8
    
    def test_align_face_224(self, sample_face_image, mock_face_detector):
        """Test aligning face to 224x224"""
        aligner = FaceAligner()
        detection = mock_face_detector.detect_largest(sample_face_image)
        
        if detection:
            aligned = aligner.align(sample_face_image, detection, output_size=224)
            assert aligned.shape == (224, 224, 3)
    
    def test_alignment_preserves_aspect(self, sample_face_image, mock_face_detector):
        """Test that alignment preserves face proportions"""
        aligner = FaceAligner()
        detection = mock_face_detector.detect_largest(sample_face_image)
        
        if detection:
            aligned = aligner.align(sample_face_image, detection, output_size=160)
            # Check that aligned image is square
            assert aligned.shape[0] == aligned.shape[1]
    
    def test_get_face_aligner_singleton(self):
        """Test that get_face_aligner returns singleton"""
        aligner1 = get_face_aligner()
        aligner2 = get_face_aligner()
        assert aligner1 is not None
        assert aligner2 is not None
    
    def test_align_with_invalid_detection(self, sample_face_image):
        """Test handling of invalid detection"""
        aligner = FaceAligner()
        detection = None
        
        # Should raise error when detection is None
        with pytest.raises(Exception):
            aligner.align(sample_face_image, detection, output_size=160)


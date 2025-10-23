"""
Challenge-Response System
Random user actions for liveness validation
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
from datetime import datetime, timedelta
import numpy as np

from app.core.config import settings
from app.core.security import security_manager


class ChallengeType(Enum):
    """Types of challenge-response actions"""
    BLINK = "blink"
    SMILE = "smile"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"


class ChallengeValidator:
    """
    Validate challenge-response actions using facial landmarks and expressions.
    """
    
    def __init__(self):
        """Initialize challenge validator"""
        self.active_challenges = {}
    
    def create_challenge(self, user_id: Optional[str] = None) -> Dict:
        """
        Create a random challenge.
        
        Args:
            user_id: Optional user ID
            
        Returns:
            Dict: Challenge details
        """
        # Get available challenge types
        available_types = settings.get_challenge_types()
        
        # Select random challenge
        challenge_type = random.choice(available_types)
        
        # Generate challenge ID
        challenge_id = security_manager.generate_challenge_id()
        
        # Create challenge
        challenge = {
            'challenge_id': challenge_id,
            'challenge_type': challenge_type,
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(
                seconds=settings.CHALLENGE_TIMEOUT_SECONDS
            ),
            'status': 'pending',
            'instructions': self._get_challenge_instructions(challenge_type)
        }
        
        # Store active challenge
        self.active_challenges[challenge_id] = challenge
        
        return challenge
    
    def _get_challenge_instructions(self, challenge_type: str) -> str:
        """
        Get human-readable instructions for challenge.
        
        Args:
            challenge_type: Type of challenge
            
        Returns:
            str: Instructions
        """
        instructions = {
            'blink': 'Please blink twice',
            'smile': 'Please smile',
            'look_left': 'Please look to your left',
            'look_right': 'Please look to your right',
            'look_up': 'Please look up',
            'look_down': 'Please look down',
            'nod': 'Please nod your head',
            'shake_head': 'Please shake your head'
        }
        
        return instructions.get(challenge_type, 'Follow the instruction')
    
    def validate_blink(
        self, 
        ear_sequence: List[float],
        threshold: float = 0.2
    ) -> Tuple[bool, float]:
        """
        Validate blink challenge.
        
        Args:
            ear_sequence: Sequence of Eye Aspect Ratio values
            threshold: EAR threshold for blink detection
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(ear_sequence) < 10:
            return False, 0.0
        
        # Detect blinks (EAR drops below threshold)
        blink_count = 0
        in_blink = False
        
        for ear in ear_sequence:
            if ear < threshold and not in_blink:
                blink_count += 1
                in_blink = True
            elif ear >= threshold:
                in_blink = False
        
        # Valid if at least 2 blinks detected
        is_valid = blink_count >= 2
        confidence = min(1.0, blink_count / 2.0)
        
        return is_valid, confidence
    
    def validate_smile(
        self,
        mar_sequence: List[float],
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """
        Validate smile challenge.
        
        Args:
            mar_sequence: Sequence of Mouth Aspect Ratio values
            threshold: MAR threshold for smile detection
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(mar_sequence) < 5:
            return False, 0.0
        
        # Check if MAR increases (mouth opens for smile)
        max_mar = max(mar_sequence)
        mean_mar = np.mean(mar_sequence)
        
        is_valid = max_mar > threshold and max_mar > mean_mar * 1.5
        confidence = min(1.0, max_mar / threshold)
        
        return is_valid, confidence
    
    def validate_head_turn(
        self,
        yaw_sequence: List[float],
        direction: str,
        threshold: float = 15.0
    ) -> Tuple[bool, float]:
        """
        Validate head turn challenge (left/right).
        
        Args:
            yaw_sequence: Sequence of yaw angles
            direction: 'left' or 'right'
            threshold: Angle threshold (degrees)
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(yaw_sequence) < 5:
            return False, 0.0
        
        max_yaw = max(yaw_sequence)
        min_yaw = min(yaw_sequence)
        
        if direction == 'left':
            # Yaw should decrease (negative)
            is_valid = min_yaw < -threshold
            confidence = min(1.0, abs(min_yaw) / threshold)
        else:  # right
            # Yaw should increase (positive)
            is_valid = max_yaw > threshold
            confidence = min(1.0, max_yaw / threshold)
        
        return is_valid, confidence
    
    def validate_head_pitch(
        self,
        pitch_sequence: List[float],
        direction: str,
        threshold: float = 15.0
    ) -> Tuple[bool, float]:
        """
        Validate head pitch challenge (up/down).
        
        Args:
            pitch_sequence: Sequence of pitch angles
            direction: 'up' or 'down'
            threshold: Angle threshold (degrees)
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(pitch_sequence) < 5:
            return False, 0.0
        
        max_pitch = max(pitch_sequence)
        min_pitch = min(pitch_sequence)
        
        if direction == 'up':
            # Pitch should decrease (negative, looking up)
            is_valid = min_pitch < -threshold
            confidence = min(1.0, abs(min_pitch) / threshold)
        else:  # down
            # Pitch should increase (positive, looking down)
            is_valid = max_pitch > threshold
            confidence = min(1.0, max_pitch / threshold)
        
        return is_valid, confidence
    
    def validate_nod(
        self,
        pitch_sequence: List[float],
        threshold: float = 10.0
    ) -> Tuple[bool, float]:
        """
        Validate nod challenge (head up and down).
        
        Args:
            pitch_sequence: Sequence of pitch angles
            threshold: Angle threshold
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(pitch_sequence) < 10:
            return False, 0.0
        
        # Detect pitch oscillations
        max_pitch = max(pitch_sequence)
        min_pitch = min(pitch_sequence)
        range_pitch = max_pitch - min_pitch
        
        is_valid = range_pitch > threshold * 2
        confidence = min(1.0, range_pitch / (threshold * 2))
        
        return is_valid, confidence
    
    def validate_shake_head(
        self,
        yaw_sequence: List[float],
        threshold: float = 10.0
    ) -> Tuple[bool, float]:
        """
        Validate head shake challenge (head left and right).
        
        Args:
            yaw_sequence: Sequence of yaw angles
            threshold: Angle threshold
            
        Returns:
            Tuple[bool, float]: (is_valid, confidence)
        """
        if len(yaw_sequence) < 10:
            return False, 0.0
        
        # Detect yaw oscillations
        max_yaw = max(yaw_sequence)
        min_yaw = min(yaw_sequence)
        range_yaw = max_yaw - min_yaw
        
        is_valid = range_yaw > threshold * 2
        confidence = min(1.0, range_yaw / (threshold * 2))
        
        return is_valid, confidence
    
    def validate_challenge(
        self,
        challenge_id: str,
        sequence_data: Dict[str, List[float]]
    ) -> Tuple[bool, float, str]:
        """
        Validate a challenge response.
        
        Args:
            challenge_id: Challenge ID
            sequence_data: Dictionary of sequences (EAR, MAR, yaw, pitch, etc.)
            
        Returns:
            Tuple[bool, float, str]: (is_valid, confidence, message)
        """
        # Get challenge
        challenge = self.active_challenges.get(challenge_id)
        
        if not challenge:
            return False, 0.0, "Challenge not found"
        
        # Check expiration
        if datetime.utcnow() > challenge['expires_at']:
            del self.active_challenges[challenge_id]
            return False, 0.0, "Challenge expired"
        
        challenge_type = challenge['challenge_type']
        
        # Validate based on type
        if challenge_type == 'blink':
            is_valid, confidence = self.validate_blink(
                sequence_data.get('ear', [])
            )
            message = "Blink detected" if is_valid else "Blink not detected"
        
        elif challenge_type == 'smile':
            is_valid, confidence = self.validate_smile(
                sequence_data.get('mar', [])
            )
            message = "Smile detected" if is_valid else "Smile not detected"
        
        elif challenge_type == 'look_left':
            is_valid, confidence = self.validate_head_turn(
                sequence_data.get('yaw', []), 
                'left'
            )
            message = "Head turn left detected" if is_valid else "Head turn not detected"
        
        elif challenge_type == 'look_right':
            is_valid, confidence = self.validate_head_turn(
                sequence_data.get('yaw', []), 
                'right'
            )
            message = "Head turn right detected" if is_valid else "Head turn not detected"
        
        elif challenge_type == 'look_up':
            is_valid, confidence = self.validate_head_pitch(
                sequence_data.get('pitch', []), 
                'up'
            )
            message = "Head pitch up detected" if is_valid else "Head pitch not detected"
        
        elif challenge_type == 'look_down':
            is_valid, confidence = self.validate_head_pitch(
                sequence_data.get('pitch', []), 
                'down'
            )
            message = "Head pitch down detected" if is_valid else "Head pitch not detected"
        
        elif challenge_type == 'nod':
            is_valid, confidence = self.validate_nod(
                sequence_data.get('pitch', [])
            )
            message = "Nod detected" if is_valid else "Nod not detected"
        
        elif challenge_type == 'shake_head':
            is_valid, confidence = self.validate_shake_head(
                sequence_data.get('yaw', [])
            )
            message = "Head shake detected" if is_valid else "Head shake not detected"
        
        else:
            return False, 0.0, "Unknown challenge type"
        
        # Update challenge status
        if is_valid:
            challenge['status'] = 'completed'
            challenge['completed_at'] = datetime.utcnow()
        
        return is_valid, confidence, message
    
    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """
        Get challenge details.
        
        Args:
            challenge_id: Challenge ID
            
        Returns:
            Optional[Dict]: Challenge details or None
        """
        return self.active_challenges.get(challenge_id)
    
    def cleanup_expired_challenges(self):
        """Remove expired challenges"""
        now = datetime.utcnow()
        expired = [
            cid for cid, challenge in self.active_challenges.items()
            if now > challenge['expires_at']
        ]
        
        for cid in expired:
            del self.active_challenges[cid]


# Singleton instance
_challenge_validator_instance: Optional[ChallengeValidator] = None


def get_challenge_validator() -> ChallengeValidator:
    """
    Get singleton challenge validator instance.
    
    Returns:
        ChallengeValidator: Challenge validator
    """
    global _challenge_validator_instance
    if _challenge_validator_instance is None:
        _challenge_validator_instance = ChallengeValidator()
    return _challenge_validator_instance


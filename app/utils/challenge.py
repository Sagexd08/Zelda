
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random
from datetime import datetime, timedelta
import numpy as np

from app.core.config import settings
from app.core.security import security_manager

class ChallengeType(Enum):
    BLINK = "blink"
    SMILE = "smile"
    LOOK_LEFT = "look_left"
    LOOK_RIGHT = "look_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    NOD = "nod"
    SHAKE_HEAD = "shake_head"

class ChallengeValidator:

    def __init__(self):
        self.active_challenges = {}

    def create_challenge(self, user_id: Optional[str] = None) -> Dict:
        available_types = settings.get_challenge_types()

        challenge_type = random.choice(available_types)

        challenge_id = security_manager.generate_challenge_id()

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

        self.active_challenges[challenge_id] = challenge

        return challenge

    def _get_challenge_instructions(self, challenge_type: str) -> str:
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
        if len(ear_sequence) < 10:
            return False, 0.0

        blink_count = 0
        in_blink = False

        for ear in ear_sequence:
            if ear < threshold and not in_blink:
                blink_count += 1
                in_blink = True
            elif ear >= threshold:
                in_blink = False

        is_valid = blink_count >= 2
        confidence = min(1.0, blink_count / 2.0)

        return is_valid, confidence

    def validate_smile(
        self,
        mar_sequence: List[float],
        threshold: float = 0.3
    ) -> Tuple[bool, float]:
        if len(mar_sequence) < 5:
            return False, 0.0

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
        if len(yaw_sequence) < 5:
            return False, 0.0

        max_yaw = max(yaw_sequence)
        min_yaw = min(yaw_sequence)

        if direction == 'left':
            is_valid = min_yaw < -threshold
            confidence = min(1.0, abs(min_yaw) / threshold)
        else:
            is_valid = max_yaw > threshold
            confidence = min(1.0, max_yaw / threshold)

        return is_valid, confidence

    def validate_head_pitch(
        self,
        pitch_sequence: List[float],
        direction: str,
        threshold: float = 15.0
    ) -> Tuple[bool, float]:
        if len(pitch_sequence) < 5:
            return False, 0.0

        max_pitch = max(pitch_sequence)
        min_pitch = min(pitch_sequence)

        if direction == 'up':
            is_valid = min_pitch < -threshold
            confidence = min(1.0, abs(min_pitch) / threshold)
        else:
            is_valid = max_pitch > threshold
            confidence = min(1.0, max_pitch / threshold)

        return is_valid, confidence

    def validate_nod(
        self,
        pitch_sequence: List[float],
        threshold: float = 10.0
    ) -> Tuple[bool, float]:
        if len(pitch_sequence) < 10:
            return False, 0.0

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
        if len(yaw_sequence) < 10:
            return False, 0.0

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
        challenge = self.active_challenges.get(challenge_id)

        if not challenge:
            return False, 0.0, "Challenge not found"

        if datetime.utcnow() > challenge['expires_at']:
            del self.active_challenges[challenge_id]
            return False, 0.0, "Challenge expired"

        challenge_type = challenge['challenge_type']

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

        if is_valid:
            challenge['status'] = 'completed'
            challenge['completed_at'] = datetime.utcnow()

        return is_valid, confidence, message

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        return self.active_challenges.get(challenge_id)

    def cleanup_expired_challenges(self):
        now = datetime.utcnow()
        expired = [
            cid for cid, challenge in self.active_challenges.items()
            if now > challenge['expires_at']
        ]

        for cid in expired:
            del self.active_challenges[cid]

_challenge_validator_instance: Optional[ChallengeValidator] = None

def get_challenge_validator() -> ChallengeValidator:
    global _challenge_validator_instance
    if _challenge_validator_instance is None:
        _challenge_validator_instance = ChallengeValidator()
    return _challenge_validator_instance

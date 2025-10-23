
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets

from jose import JWTError, jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet
import numpy as np

from app.core.config import settings

class SecurityManager:

    def __init__(self):
        self.ph = PasswordHasher(
            time_cost=settings.ARGON2_TIME_COST,
            memory_cost=settings.ARGON2_MEMORY_COST,
            parallelism=settings.ARGON2_PARALLELISM
        )

        self._setup_encryption()

    def _setup_encryption(self):
        try:
            key = settings.ENCRYPTION_KEY.encode()
            if len(key) != 44:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                import base64

                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'facial_auth_salt',
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(settings.ENCRYPTION_KEY.encode()))

            self.cipher = Fernet(key)
        except Exception as e:
            print(f"Warning: Encryption setup failed: {e}. Using fallback.")
            self.cipher = Fernet(Fernet.generate_key())

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.JWT_EXPIRATION_MINUTES
            )

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )

        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except JWTError:
            return None

    def create_api_key(self) -> str:
        return secrets.token_urlsafe(32)

    def hash_password(self, password: str) -> str:
        return self.ph.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        try:
            self.ph.verify(hashed, password)

            if self.ph.check_needs_rehash(hashed):
                return True

            return True
        except VerifyMismatchError:
            return False

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        embedding_bytes = embedding.tobytes()

        encrypted = self.cipher.encrypt(embedding_bytes)

        return encrypted

    def decrypt_embedding(
        self,
        encrypted_data: bytes,
        shape: tuple = (512,),
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        decrypted_bytes = self.cipher.decrypt(encrypted_data)

        embedding = np.frombuffer(decrypted_bytes, dtype=dtype)
        embedding = embedding.reshape(shape)

        return embedding

    def encrypt_string(self, text: str) -> bytes:
        return self.cipher.encrypt(text.encode())

    def decrypt_string(self, encrypted_data: bytes) -> str:
        return self.cipher.decrypt(encrypted_data).decode()

    def generate_challenge_id(self) -> str:
        return secrets.token_hex(16)

    def generate_session_id(self) -> str:
        return secrets.token_urlsafe(24)

security_manager = SecurityManager()

def get_security_manager() -> SecurityManager:
    return security_manager

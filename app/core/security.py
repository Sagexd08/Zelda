"""
Security Utilities
Handles JWT tokens, password hashing, encryption, and API authentication
"""

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
    """
    Centralized security management for authentication and encryption.
    """
    
    def __init__(self):
        """Initialize security components"""
        # Argon2 password hasher
        self.ph = PasswordHasher(
            time_cost=settings.ARGON2_TIME_COST,
            memory_cost=settings.ARGON2_MEMORY_COST,
            parallelism=settings.ARGON2_PARALLELISM
        )
        
        # Fernet encryption for embeddings
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup Fernet encryption cipher"""
        try:
            # Use configured encryption key or generate new one
            key = settings.ENCRYPTION_KEY.encode()
            if len(key) != 44:  # Fernet key should be 44 bytes (base64 encoded 32 bytes)
                # Generate proper Fernet key from password
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                import base64
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'facial_auth_salt',  # In production, use random salt stored securely
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(settings.ENCRYPTION_KEY.encode()))
            
            self.cipher = Fernet(key)
        except Exception as e:
            print(f"Warning: Encryption setup failed: {e}. Using fallback.")
            # Fallback: generate temporary key
            self.cipher = Fernet(Fernet.generate_key())
    
    # ============================================
    # JWT Token Management
    # ============================================
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Payload data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            str: Encoded JWT token
        """
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
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Optional[Dict]: Decoded payload if valid, None otherwise
        """
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
        """
        Generate a secure API key.
        
        Returns:
            str: Random API key
        """
        return secrets.token_urlsafe(32)
    
    # ============================================
    # Password Hashing (Argon2)
    # ============================================
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2.
        
        Args:
            password: Plain text password
            
        Returns:
            str: Hashed password
        """
        return self.ph.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            bool: True if password matches
        """
        try:
            self.ph.verify(hashed, password)
            
            # Check if rehashing is needed (parameters changed)
            if self.ph.check_needs_rehash(hashed):
                return True  # Caller should rehash
            
            return True
        except VerifyMismatchError:
            return False
    
    # ============================================
    # Embedding Encryption (AES-256 via Fernet)
    # ============================================
    
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Encrypt a face embedding for secure storage.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            bytes: Encrypted embedding
        """
        # Convert numpy array to bytes
        embedding_bytes = embedding.tobytes()
        
        # Encrypt
        encrypted = self.cipher.encrypt(embedding_bytes)
        
        return encrypted
    
    def decrypt_embedding(
        self, 
        encrypted_data: bytes, 
        shape: tuple = (512,),
        dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Decrypt an encrypted face embedding.
        
        Args:
            encrypted_data: Encrypted embedding bytes
            shape: Original embedding shape
            dtype: Original embedding data type
            
        Returns:
            np.ndarray: Decrypted embedding
        """
        # Decrypt
        decrypted_bytes = self.cipher.decrypt(encrypted_data)
        
        # Convert back to numpy array
        embedding = np.frombuffer(decrypted_bytes, dtype=dtype)
        embedding = embedding.reshape(shape)
        
        return embedding
    
    def encrypt_string(self, text: str) -> bytes:
        """
        Encrypt a string.
        
        Args:
            text: Plain text string
            
        Returns:
            bytes: Encrypted data
        """
        return self.cipher.encrypt(text.encode())
    
    def decrypt_string(self, encrypted_data: bytes) -> str:
        """
        Decrypt encrypted string data.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            str: Decrypted string
        """
        return self.cipher.decrypt(encrypted_data).decode()
    
    # ============================================
    # Secure Random Generation
    # ============================================
    
    def generate_challenge_id(self) -> str:
        """
        Generate a unique challenge ID.
        
        Returns:
            str: Random challenge ID
        """
        return secrets.token_hex(16)
    
    def generate_session_id(self) -> str:
        """
        Generate a unique session ID.
        
        Returns:
            str: Random session ID
        """
        return secrets.token_urlsafe(24)


# Global security manager instance
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """
    Dependency injection for security manager.
    
    Returns:
        SecurityManager: Security manager instance
    """
    return security_manager


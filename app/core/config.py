"""
Configuration Management
Centralized settings using Pydantic for environment variable management
"""

from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
import torch
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type checking.
    """
    
    # ============================================
    # Application Settings
    # ============================================
    APP_NAME: str = "Facial Authentication System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # ============================================
    # API Configuration
    # ============================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    RELOAD: bool = False
    
    # ============================================
    # Security Settings
    # ============================================
    SECRET_KEY: str = "change-this-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 1440  # 24 hours
    API_KEY_HEADER: str = "X-API-Key"
    
    # Argon2 password hashing
    ARGON2_TIME_COST: int = 2
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 4
    
    # ============================================
    # Database Configuration
    # ============================================
    DATABASE_URL: str = "sqlite:///./facial_auth.db"
    ENCRYPTION_KEY: str = "change-this-encryption-key"
    
    # ============================================
    # ML Model Configuration
    # ============================================
    DEVICE: str = "auto"  # cuda, cpu, auto
    
    # Model paths
    MODEL_BASE_PATH: str = "./weights"
    RETINAFACE_MODEL: str = "retinaface_resnet50.pth"
    ARCFACE_MODEL: str = "arcface_resnet100.pth"
    FACENET_MODEL: str = "facenet_inception_resnet_v1.pth"
    MOBILEFACENET_MODEL: str = "mobilefacenet.pth"
    LIVENESS_MODEL: str = "liveness_resnet18.pth"
    TEMPORAL_LIVENESS_MODEL: str = "temporal_lstm.pth"
    DEPTH_MODEL: str = "midas_v21_small.pt"
    FUSION_MODEL: str = "fusion_mlp.pth"
    
    # Model input sizes
    FACE_SIZE: int = 160
    FACE_SIZE_ARCFACE: int = 224
    
    # ============================================
    # Face Detection Settings
    # ============================================
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.85
    DETECTION_NMS_THRESHOLD: float = 0.4
    MIN_FACE_SIZE: int = 40
    MAX_FACES: int = 5
    
    # ============================================
    # Face Recognition Settings
    # ============================================
    VERIFICATION_THRESHOLD: float = 0.65
    IDENTIFICATION_THRESHOLD: float = 0.70
    
    # Adaptive thresholding
    ENABLE_ADAPTIVE_THRESHOLD: bool = True
    ADAPTIVE_THRESHOLD_MIN: float = 0.55
    ADAPTIVE_THRESHOLD_MAX: float = 0.75
    
    # ============================================
    # Liveness Detection Settings
    # ============================================
    LIVENESS_THRESHOLD: float = 0.90
    ENABLE_DEPTH_ESTIMATION: bool = True
    DEPTH_THRESHOLD: float = 0.15
    
    # Temporal liveness
    ENABLE_TEMPORAL_LIVENESS: bool = True
    MIN_VIDEO_FRAMES: int = 30
    BLINK_DETECTION_THRESHOLD: float = 0.2
    
    # ============================================
    # Registration Settings
    # ============================================
    MIN_REGISTRATION_SAMPLES: int = 5
    MAX_REGISTRATION_SAMPLES: int = 10
    REGISTRATION_QUALITY_THRESHOLD: float = 0.80
    
    # Image quality checks
    MIN_SHARPNESS_SCORE: float = 50.0
    MAX_BLUR_VARIANCE: float = 100.0
    MAX_HEAD_POSE_YAW: float = 30.0
    MAX_HEAD_POSE_PITCH: float = 30.0
    
    # ============================================
    # Online Learning Settings
    # ============================================
    ENABLE_ONLINE_LEARNING: bool = True
    EMBEDDING_UPDATE_ALPHA: float = 0.1
    MAX_EMBEDDING_UPDATES: int = 100
    
    # ============================================
    # Bias Monitoring
    # ============================================
    ENABLE_BIAS_MONITORING: bool = True
    BIAS_CHECK_INTERVAL_HOURS: int = 24
    BIAS_THRESHOLD: float = 0.05
    
    # ============================================
    # Challenge-Response Settings
    # ============================================
    ENABLE_CHALLENGE_RESPONSE: bool = True
    CHALLENGE_TIMEOUT_SECONDS: int = 10
    CHALLENGE_TYPES: str = "blink,smile,look_left,look_right,look_up,look_down"
    
    # ============================================
    # Performance Settings
    # ============================================
    MAX_BATCH_SIZE: int = 32
    INFERENCE_TIMEOUT_SECONDS: int = 5
    
    # Caching
    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 300
    
    # ============================================
    # Logging Configuration
    # ============================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/facial_auth.log"
    LOG_ROTATION: str = "10MB"
    LOG_RETENTION_DAYS: int = 30
    
    # ============================================
    # Rate Limiting
    # ============================================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    
    # ============================================
    # CORS Settings
    # ============================================
    CORS_ENABLED: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # ============================================
    # Monitoring & Metrics
    # ============================================
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_ENABLED: bool = True
    
    # ============================================
    # Voice Recognition (Optional)
    # ============================================
    ENABLE_VOICE_AUTH: bool = False
    VOICE_MODEL_PATH: str = "./weights/ecapa_tdnn.pth"
    VOICE_SIMILARITY_THRESHOLD: float = 0.75
    
    # ============================================
    # WebSocket Settings
    # ============================================
    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_MAX_CONNECTIONS: int = 100
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    
    # ============================================
    # Data Retention & GDPR
    # ============================================
    ENABLE_AUTO_DELETION: bool = False
    DATA_RETENTION_DAYS: int = 365
    AUDIT_LOG_RETENTION_DAYS: int = 730
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    def get_device(self) -> torch.device:
        """
        Determine the computation device (GPU/CPU).
        
        Returns:
            torch.device: The device to use for model inference
        """
        if self.DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.DEVICE)
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Get full path to a model file.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Path: Full path to the model
        """
        return Path(self.MODEL_BASE_PATH) / model_name
    
    def get_challenge_types(self) -> List[str]:
        """
        Get list of challenge types.
        
        Returns:
            List[str]: Available challenge types
        """
        return [c.strip() for c in self.CHALLENGE_TYPES.split(',')]
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.
    
    Returns:
        Settings: Application settings instance
    """
    return settings


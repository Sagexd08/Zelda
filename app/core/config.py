
from typing import List, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings
import torch
from pathlib import Path
from secrets import token_urlsafe

class Settings(BaseSettings):

    APP_NAME: str = "Facial Authentication System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    RELOAD: bool = False

    SECRET_KEY: str = "change-this-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 1440
    API_KEY_HEADER: str = "X-API-Key"

    ARGON2_TIME_COST: int = 2
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 4

    DATABASE_URL: str = "sqlite:///./facial_auth.db"
    ENCRYPTION_KEY: str = "change-this-encryption-key"
    
    # Supabase configuration
    SUPABASE_URL: str = "https://grtnutwjmlhpdekllbxl.supabase.co"
    SUPABASE_ANON_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdydG51dHdqbWxocGRla2xsYnhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2OTU0NDcsImV4cCI6MjA3ODI3MTQ0N30.7QYuaztLjajQD3Serppa6CNXeesN7N_Qu0Pz04nOpB4"
    SUPABASE_SERVICE_KEY: Optional[str] = None
    SUPABASE_JWT_SECRET: str = "FUyqyH6GPuACY0YeO4o2Pdm68IDC5P2S1JWFtjNAq0ZvMKWjBPU+h1k5C8hHM6GbNngHbnkBYuwG8Yx00dUGfg=="
    USE_SUPABASE: bool = True  # Set to True to use Supabase instead of SQLAlchemy

    DEVICE: str = "auto"

    MODEL_BASE_PATH: str = "./weights"
    RETINAFACE_MODEL: str = "retinaface_resnet50.pth"
    ARCFACE_MODEL: str = "arcface_resnet100.pth"
    FACENET_MODEL: str = "facenet_inception_resnet_v1.pth"
    MOBILEFACENET_MODEL: str = "mobilefacenet.pth"
    LIVENESS_MODEL: str = "liveness_resnet18.pth"
    TEMPORAL_LIVENESS_MODEL: str = "temporal_lstm.pth"
    DEPTH_MODEL: str = "midas_v21_small.pt"
    FUSION_MODEL: str = "fusion_mlp.pth"

    FACE_SIZE: int = 160
    FACE_SIZE_ARCFACE: int = 224

    DETECTION_CONFIDENCE_THRESHOLD: float = 0.85
    DETECTION_NMS_THRESHOLD: float = 0.4
    MIN_FACE_SIZE: int = 40
    MAX_FACES: int = 5

    VERIFICATION_THRESHOLD: float = 0.65
    IDENTIFICATION_THRESHOLD: float = 0.70

    ENABLE_ADAPTIVE_THRESHOLD: bool = True
    ADAPTIVE_THRESHOLD_MIN: float = 0.55
    ADAPTIVE_THRESHOLD_MAX: float = 0.75

    LIVENESS_THRESHOLD: float = 0.90
    ENABLE_DEPTH_ESTIMATION: bool = True
    DEPTH_THRESHOLD: float = 0.15

    ENABLE_TEMPORAL_LIVENESS: bool = True
    MIN_VIDEO_FRAMES: int = 30
    BLINK_DETECTION_THRESHOLD: float = 0.2

    MIN_REGISTRATION_SAMPLES: int = 1
    MAX_REGISTRATION_SAMPLES: int = 10
    REGISTRATION_QUALITY_THRESHOLD: float = 0.80

    MIN_SHARPNESS_SCORE: float = 50.0
    MAX_BLUR_VARIANCE: float = 100.0
    MAX_HEAD_POSE_YAW: float = 30.0
    MAX_HEAD_POSE_PITCH: float = 30.0

    ENABLE_ONLINE_LEARNING: bool = True
    EMBEDDING_UPDATE_ALPHA: float = 0.1
    MAX_EMBEDDING_UPDATES: int = 100

    ENABLE_BIAS_MONITORING: bool = True
    BIAS_CHECK_INTERVAL_HOURS: int = 24
    BIAS_THRESHOLD: float = 0.05

    ENABLE_CHALLENGE_RESPONSE: bool = True
    CHALLENGE_TIMEOUT_SECONDS: int = 10
    CHALLENGE_TYPES: str = "blink,smile,look_left,look_right,look_up,look_down"

    MAX_BATCH_SIZE: int = 32
    INFERENCE_TIMEOUT_SECONDS: int = 5

    MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024  # 5 MB
    USER_ID_MAX_LENGTH: int = 64
    USER_ID_PATTERN: str = r"^[A-Za-z0-9_\-]+$"

    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 300

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/facial_auth.log"
    LOG_ROTATION: str = "10MB"
    LOG_RETENTION_DAYS: int = 30

    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    CORS_ENABLED: bool = True
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "https://zelda-facial-auth.vercel.app"]
    CORS_ALLOW_CREDENTIALS: bool = True

    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    PROMETHEUS_ENABLED: bool = True

    ENABLE_VOICE_AUTH: bool = False
    VOICE_MODEL_PATH: str = "./weights/ecapa_tdnn.pth"
    VOICE_SIMILARITY_THRESHOLD: float = 0.75

    WEBSOCKET_ENABLED: bool = True
    WEBSOCKET_MAX_CONNECTIONS: int = 100
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30

    ENABLE_AUTO_DELETION: bool = False
    DATA_RETENTION_DAYS: int = 365
    AUDIT_LOG_RETENTION_DAYS: int = 730

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    @field_validator('DEBUG', mode='before')
    @classmethod
    def normalize_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            truthy = {'1', 'true', 'yes', 'on'}
            falsy = {'0', 'false', 'no', 'off', 'warn'}
            if lowered in truthy:
                return True
            if lowered in falsy:
                return False
        return bool(value)

    def get_device(self) -> torch.device:
        if self.DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.DEVICE)

    def get_model_path(self, model_name: str) -> Path:
        return Path(self.MODEL_BASE_PATH) / model_name

    def get_challenge_types(self) -> List[str]:
        return [c.strip() for c in self.CHALLENGE_TYPES.split(',')]

    @model_validator(mode="after")
    def enforce_security_and_limits(self):
        env = (self.ENVIRONMENT or "").lower()

        if self.MAX_UPLOAD_BYTES <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be a positive integer")

        if env == "production":
            if not self.SECRET_KEY or self.SECRET_KEY == "change-this-secret-key-in-production":
                raise ValueError("SECRET_KEY must be set for production environment")
            if not self.ENCRYPTION_KEY or self.ENCRYPTION_KEY == "change-this-encryption-key":
                raise ValueError("ENCRYPTION_KEY must be set for production environment")
        else:
            if not self.SECRET_KEY or self.SECRET_KEY == "change-this-secret-key-in-production":
                self.SECRET_KEY = token_urlsafe(32)
            if not self.ENCRYPTION_KEY or self.ENCRYPTION_KEY == "change-this-encryption-key":
                self.ENCRYPTION_KEY = token_urlsafe(32)

        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()

def get_settings() -> Settings:
    return settings

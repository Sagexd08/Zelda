"""
Database Layer
SQLAlchemy ORM with support for SQLite and PostgreSQL
Includes encryption for sensitive data
"""

from datetime import datetime
from typing import Optional, List
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, LargeBinary, Text, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings

# Create base class for models
Base = declarative_base()


# ============================================
# Database Models
# ============================================

class User(Base):
    """
    User model storing user information and authentication metadata.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Authentication settings
    is_active = Column(Boolean, default=True)
    api_key_hash = Column(String(255), nullable=True)
    
    # Adaptive thresholding
    custom_threshold = Column(Float, nullable=True)
    threshold_confidence = Column(Float, default=0.5)
    
    # Statistics
    total_authentications = Column(Integer, default=0)
    successful_authentications = Column(Integer, default=0)
    failed_attempts = Column(Integer, default=0)
    last_authentication = Column(DateTime, nullable=True)
    
    # Relationships
    embeddings = relationship("Embedding", back_populates="user", cascade="all, delete-orphan")
    liveness_signatures = relationship("LivenessSignature", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    voice_embeddings = relationship("VoiceEmbedding", back_populates="user", cascade="all, delete-orphan")


class Embedding(Base):
    """
    Face embedding storage with encryption.
    Stores multiple embeddings per user for robustness.
    """
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Encrypted embedding data
    arcface_embedding = Column(LargeBinary, nullable=True)
    facenet_embedding = Column(LargeBinary, nullable=True)
    mobilefacenet_embedding = Column(LargeBinary, nullable=True)
    fusion_embedding = Column(LargeBinary, nullable=True)
    
    # Embedding metadata
    embedding_version = Column(String(50), default="1.0")
    quality_score = Column(Float, nullable=True)
    is_primary = Column(Boolean, default=False)
    
    # Update tracking for online learning
    update_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="embeddings")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_user_primary', 'user_id', 'is_primary'),
    )


class LivenessSignature(Base):
    """
    Liveness detection signatures for each user.
    Stores biometric patterns for enhanced anti-spoofing.
    """
    __tablename__ = "liveness_signatures"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Temporal patterns
    blink_pattern = Column(Text, nullable=True)  # JSON encoded
    head_movement_pattern = Column(Text, nullable=True)
    micro_expression_signature = Column(LargeBinary, nullable=True)
    
    # Depth information
    depth_variance = Column(Float, nullable=True)
    depth_signature = Column(LargeBinary, nullable=True)
    
    # Metadata
    sample_count = Column(Integer, default=0)
    confidence_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="liveness_signatures")


class VoiceEmbedding(Base):
    """
    Voice embeddings for multimodal authentication (optional).
    """
    __tablename__ = "voice_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Encrypted voice embedding
    voice_embedding = Column(LargeBinary, nullable=True)
    
    # Metadata
    quality_score = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="voice_embeddings")


class AuditLog(Base):
    """
    Audit log for authentication attempts and system events.
    GDPR-compliant: only stores user_id, no PII.
    """
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Event information
    event_type = Column(String(100), nullable=False)  # register, authenticate, identify, etc.
    event_result = Column(String(50), nullable=False)  # success, failure, error
    
    # Anonymized details
    confidence_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    failure_reason = Column(String(255), nullable=True)
    
    # System metadata
    ip_address_hash = Column(String(64), nullable=True)  # Hashed for privacy
    user_agent_hash = Column(String(64), nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes for querying
    __table_args__ = (
        Index('idx_event_timestamp', 'event_type', 'timestamp'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )


class BiasMetric(Base):
    """
    Bias monitoring metrics for fairness evaluation.
    Tracks performance across demographic groups.
    """
    __tablename__ = "bias_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Demographic category (race, gender, age_group, etc.)
    category = Column(String(50), nullable=False)
    group_name = Column(String(100), nullable=False)
    
    # Performance metrics
    total_samples = Column(Integer, default=0)
    true_positives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)
    
    # Computed metrics
    accuracy = Column(Float, nullable=True)
    far = Column(Float, nullable=True)  # False Accept Rate
    frr = Column(Float, nullable=True)  # False Reject Rate
    
    # Fairness indicators
    demographic_parity_diff = Column(Float, nullable=True)
    equalized_odds_diff = Column(Float, nullable=True)
    
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_category_group', 'category', 'group_name'),
    )


class ChallengeSession(Base):
    """
    Challenge-response session storage for liveness validation.
    """
    __tablename__ = "challenge_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Challenge details
    challenge_type = Column(String(50), nullable=False)
    challenge_data = Column(Text, nullable=True)  # JSON encoded
    
    # Session state
    is_completed = Column(Boolean, default=False)
    is_successful = Column(Boolean, default=False)
    attempts = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)


# ============================================
# Database Connection Management
# ============================================

class DatabaseManager:
    """
    Database connection and session management.
    Supports both SQLite (dev) and PostgreSQL (production).
    """
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup database engine based on configuration"""
        if settings.DATABASE_URL.startswith("sqlite"):
            # SQLite configuration
            self.engine = create_engine(
                settings.DATABASE_URL,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.DEBUG
            )
        else:
            # PostgreSQL configuration
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=settings.DEBUG
            )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution)"""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """
        Get a database session.
        
        Returns:
            Session: SQLAlchemy session
        """
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Session:
    """
    Dependency injection for database session.
    Automatically handles session cleanup.
    
    Yields:
        Session: Database session
    """
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database with tables.
    Called on application startup.
    """
    db_manager.create_tables()
    print(f"Database initialized: {settings.DATABASE_URL}")


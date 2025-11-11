
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

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    is_active = Column(Boolean, default=True)
    api_key_hash = Column(String(255), nullable=True)

    custom_threshold = Column(Float, nullable=True)
    threshold_confidence = Column(Float, default=0.5)

    total_authentications = Column(Integer, default=0)
    successful_authentications = Column(Integer, default=0)
    failed_attempts = Column(Integer, default=0)
    last_authentication = Column(DateTime, nullable=True)

    embeddings = relationship("Embedding", back_populates="user", cascade="all, delete-orphan")
    liveness_signatures = relationship("LivenessSignature", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    voice_embeddings = relationship("VoiceEmbedding", back_populates="user", cascade="all, delete-orphan")

class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    arcface_embedding = Column(LargeBinary, nullable=True)
    facenet_embedding = Column(LargeBinary, nullable=True)
    mobilefacenet_embedding = Column(LargeBinary, nullable=True)
    fusion_embedding = Column(LargeBinary, nullable=True)

    embedding_version = Column(String(50), default="1.0")
    quality_score = Column(Float, nullable=True)
    is_primary = Column(Boolean, default=False)

    update_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="embeddings")

    __table_args__ = (
        Index('idx_user_primary', 'user_id', 'is_primary'),
    )

class LivenessSignature(Base):
    __tablename__ = "liveness_signatures"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    blink_pattern = Column(Text, nullable=True)
    head_movement_pattern = Column(Text, nullable=True)
    micro_expression_signature = Column(LargeBinary, nullable=True)

    depth_variance = Column(Float, nullable=True)
    depth_signature = Column(LargeBinary, nullable=True)

    sample_count = Column(Integer, default=0)
    confidence_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="liveness_signatures")

class VoiceEmbedding(Base):
    __tablename__ = "voice_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    voice_embedding = Column(LargeBinary, nullable=True)

    quality_score = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="voice_embeddings")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    event_type = Column(String(100), nullable=False)
    event_result = Column(String(50), nullable=False)

    confidence_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    failure_reason = Column(String(255), nullable=True)

    ip_address_hash = Column(String(64), nullable=True)
    user_agent_hash = Column(String(64), nullable=True)
    processing_time_ms = Column(Float, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index('idx_event_timestamp', 'event_type', 'timestamp'),
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )

class BiasMetric(Base):
    __tablename__ = "bias_metrics"

    id = Column(Integer, primary_key=True, index=True)

    category = Column(String(50), nullable=False)
    group_name = Column(String(100), nullable=False)

    total_samples = Column(Integer, default=0)
    true_positives = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    true_negatives = Column(Integer, default=0)
    false_negatives = Column(Integer, default=0)

    accuracy = Column(Float, nullable=True)
    far = Column(Float, nullable=True)
    frr = Column(Float, nullable=True)

    demographic_parity_diff = Column(Float, nullable=True)
    equalized_odds_diff = Column(Float, nullable=True)

    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_category_group', 'category', 'group_name'),
    )

class ChallengeSession(Base):
    __tablename__ = "challenge_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    challenge_type = Column(String(50), nullable=False)
    challenge_data = Column(Text, nullable=True)

    is_completed = Column(Boolean, default=False)
    is_successful = Column(Boolean, default=False)
    attempts = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

class DatabaseManager:

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()

    def _setup_engine(self):
        if settings.DATABASE_URL.startswith("sqlite"):
            self.engine = create_engine(
                settings.DATABASE_URL,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.DEBUG
            )
        else:
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
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def close(self):
        if self.engine:
            self.engine.dispose()

db_manager = DatabaseManager()

def get_db() -> Session:
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database based on configuration"""
    if settings.USE_SUPABASE:
        print("=" * 60)
        print("Using Supabase as database backend")
        print(f"Supabase URL: {settings.SUPABASE_URL}")
        print("=" * 60)
        # Test Supabase connection
        try:
            from app.database.supabase_client import get_supabase_client
            client = get_supabase_client()
            # Test query
            client.table("users").select("id").limit(1).execute()
            print("✅ Supabase connection successful")
        except Exception as e:
            print(f"⚠️  Supabase connection test failed: {e}")
            print("Please ensure:")
            print("  1. Supabase schema is set up (run supabase_schema.sql)")
            print("  2. Supabase credentials are correct")
            print("  3. Network connectivity to Supabase")
    else:
        db_manager.create_tables()
        print(f"Database initialized: {settings.DATABASE_URL}")
        print("Using SQLAlchemy (SQLite/PostgreSQL)")

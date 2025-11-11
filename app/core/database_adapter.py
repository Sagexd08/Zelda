"""
Database Adapter - Provides a unified interface for database operations
Supports both SQLAlchemy (SQLite/PostgreSQL) and Supabase backends
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from app.core.config import settings

if not settings.USE_SUPABASE:
    from sqlalchemy.orm import Session
    from app.core.database import User, Embedding, LivenessSignature, AuditLog, get_db as get_sqlalchemy_db
else:
    from app.database.supabase_client import (
        get_user_from_db as supabase_get_user,
        create_user_in_db as supabase_create_user,
        update_user_in_db as supabase_update_user,
        delete_user_from_db as supabase_delete_user,
        create_auth_log as supabase_create_auth_log,
        get_all_users_from_db as supabase_get_all_users,
        get_auth_logs_from_db as supabase_get_auth_logs,
        store_embedding as supabase_store_embedding,
        get_embeddings_for_user as supabase_get_embeddings,
    )


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id"""
        pass
    
    @abstractmethod
    def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new user"""
        pass
    
    @abstractmethod
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user data"""
        pass
    
    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete user and related data"""
        pass


class SQLAlchemyAdapter(DatabaseAdapter):
    """SQLAlchemy database adapter"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id from SQLAlchemy"""
        from app.core.database import User
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user:
            return {
                'id': user.id,
                'user_id': user.user_id,
                'created_at': user.created_at,
                'updated_at': user.updated_at,
                'is_active': user.is_active,
                'api_key_hash': user.api_key_hash,
                'custom_threshold': user.custom_threshold,
                'threshold_confidence': user.threshold_confidence,
                'total_authentications': user.total_authentications,
                'successful_authentications': user.successful_authentications,
                'failed_attempts': user.failed_attempts,
                'last_authentication': user.last_authentication,
            }
        return None
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create user in SQLAlchemy"""
        from app.core.database import User
        new_user = User(
            user_id=user_data['user_id'],
            created_at=user_data.get('created_at', datetime.utcnow()),
            is_active=user_data.get('is_active', True),
            custom_threshold=user_data.get('custom_threshold'),
            threshold_confidence=user_data.get('threshold_confidence', 0.5),
        )
        self.db.add(new_user)
        self.db.flush()
        return self.get_user(user_data['user_id'])
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user in SQLAlchemy"""
        from app.core.database import User
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user:
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            self.db.commit()
            self.db.refresh(user)
            return self.get_user(user_id)
        return None
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user from SQLAlchemy"""
        from app.core.database import User
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user:
            self.db.delete(user)
            self.db.commit()
            return True
        return False


class SupabaseAdapter(DatabaseAdapter):
    """Supabase database adapter"""
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id from Supabase"""
        return supabase_get_user(user_id)
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create user in Supabase"""
        return supabase_create_user(user_data)
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update user in Supabase"""
        return supabase_update_user(user_id, updates)
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user from Supabase"""
        return supabase_delete_user(user_id)


def get_database_adapter(db=None):
    """Get the appropriate database adapter based on configuration"""
    if settings.USE_SUPABASE:
        return SupabaseAdapter()
    else:
        if db is None:
            # For dependency injection, we need to get db from get_db
            from app.core.database import get_db
            db = next(get_db())
        return SQLAlchemyAdapter(db)


def get_db_session():
    """Get database session (SQLAlchemy) or None (Supabase)"""
    if settings.USE_SUPABASE:
        return None
    else:
        from app.core.database import get_db
        return next(get_db())


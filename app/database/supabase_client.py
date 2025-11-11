import os
from supabase import create_client, Client
from typing import Optional, List, Dict, Any
import json

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://grtnutwjmlhpdekllbxl.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdydG51dHdqbWxocGRla2xsYnhsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI2OTU0NDcsImV4cCI6MjA3ODI3MTQ0N30.7QYuaztLjajQD3Serppa6CNXeesN7N_Qu0Pz04nOpB4")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "FUyqyH6GPuACY0YeO4o2Pdm68IDC5P2S1JWFtjNAq0ZvMKWjBPU+h1k5C8hHM6GbNngHbnkBYuwG8Yx00dUGfg==")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    return supabase

# Database helper functions
def get_user_from_db(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user from Supabase database."""
    try:
        response = supabase.table("users").select("*").eq("user_id", user_id).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error fetching user from Supabase: {e}")
        return None

def create_user_in_db(user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create user in Supabase database."""
    try:
        response = supabase.table("users").insert(user_data).execute()
        return response.data[0] if response.data and len(response.data) > 0 else None
    except Exception as e:
        print(f"Error creating user in Supabase: {e}")
        raise

def update_user_in_db(user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update user in Supabase database."""
    try:
        response = supabase.table("users").update(updates).eq("user_id", user_id).execute()
        return response.data[0] if response.data and len(response.data) > 0 else None
    except Exception as e:
        print(f"Error updating user in Supabase: {e}")
        raise

def delete_user_from_db(user_id: str) -> bool:
    """Delete user from Supabase database."""
    try:
        # First check if user exists
        user = get_user_from_db(user_id)
        if not user:
            return False
        
        # Get user's internal id for foreign key relationships
        user_internal_id = user.get('id')
        
        # Delete embeddings first (by user_id string)
        try:
            supabase.table("embeddings").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Warning: Error deleting embeddings: {e}")
        
        # Delete auth logs (by user_id string)
        try:
            supabase.table("auth_logs").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Warning: Error deleting auth logs: {e}")
        
        # Delete liveness signatures (by user_id string)
        try:
            supabase.table("liveness_signatures").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Warning: Error deleting liveness signatures: {e}")
        
        # Delete voice embeddings (by user_id string)
        try:
            supabase.table("voice_embeddings").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Warning: Error deleting voice embeddings: {e}")
        
        # Delete challenge sessions (by user_id string)
        try:
            supabase.table("challenge_sessions").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"Warning: Error deleting challenge sessions: {e}")
        
        # Delete user (by user_id string)
        supabase.table("users").delete().eq("user_id", user_id).execute()
        return True
    except Exception as e:
        print(f"Error deleting user from Supabase: {e}")
        return False

def create_auth_log(log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create authentication log in Supabase."""
    try:
        response = supabase.table("auth_logs").insert(log_data).execute()
        return response.data[0] if response.data and len(response.data) > 0 else None
    except Exception as e:
        print(f"Error creating auth log in Supabase: {e}")
        return None

def get_all_users_from_db() -> List[Dict[str, Any]]:
    """Get all users from Supabase database."""
    try:
        response = supabase.table("users").select("*").order("created_at", desc=False).execute()
        # Reverse to get descending order
        data = response.data or []
        return list(reversed(data))
    except Exception as e:
        print(f"Error fetching users from Supabase: {e}")
        return []

def get_auth_logs_from_db(limit: int = 100) -> List[Dict[str, Any]]:
    """Get authentication logs from Supabase."""
    try:
        response = supabase.table("auth_logs").select("*").order("created_at", desc=False).limit(limit).execute()
        # Reverse to get descending order
        data = response.data or []
        return list(reversed(data))[:limit]
    except Exception as e:
        print(f"Error fetching auth logs from Supabase: {e}")
        return []

def store_embedding(user_id: str, embedding: Dict[str, List[float]], quality_score: float, liveness_score: float, is_primary: bool = False) -> Optional[Dict[str, Any]]:
    """Store embedding in Supabase database."""
    try:
        # Convert embeddings to JSONB format
        embedding_json = json.dumps(embedding)
        
        embedding_data = {
            "user_id": user_id,
            "embedding": embedding_json,
            "quality_score": quality_score,
            "liveness_score": liveness_score,
            "is_primary": is_primary,
        }
        response = supabase.table("embeddings").insert(embedding_data).execute()
        return response.data[0] if response.data and len(response.data) > 0 else None
    except Exception as e:
        print(f"Error storing embedding in Supabase: {e}")
        return None

def get_embeddings_for_user(user_id: str, primary_only: bool = False) -> List[Dict[str, Any]]:
    """Get all embeddings for a user from Supabase."""
    try:
        query = supabase.table("embeddings").select("*").eq("user_id", user_id)
        if primary_only:
            query = query.eq("is_primary", True)
        response = query.order("created_at", desc=False).execute()
        # Reverse to get descending order
        embeddings = list(reversed(response.data or []))
        # Parse JSONB embedding back to dict
        for emb in embeddings:
            if isinstance(emb.get('embedding'), str):
                emb['embedding'] = json.loads(emb['embedding'])
        return embeddings
    except Exception as e:
        print(f"Error fetching embeddings from Supabase: {e}")
        return []

def update_primary_embedding(user_id: str, embedding_id: str) -> bool:
    """Set an embedding as primary (and unset others)."""
    try:
        # Unset all primary embeddings for this user
        supabase.table("embeddings").update({"is_primary": False}).eq("user_id", user_id).execute()
        # Set this one as primary
        supabase.table("embeddings").update({"is_primary": True}).eq("id", embedding_id).execute()
        return True
    except Exception as e:
        print(f"Error updating primary embedding in Supabase: {e}")
        return False


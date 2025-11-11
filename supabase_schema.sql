-- Supabase Database Schema for Facial Authentication System
-- Run this SQL in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    api_key_hash VARCHAR(255),
    custom_threshold FLOAT,
    threshold_confidence FLOAT DEFAULT 0.5,
    total_authentications INTEGER DEFAULT 0,
    successful_authentications INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    last_authentication TIMESTAMP WITH TIME ZONE,
    embedding_count INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 0.0,
    success_rate FLOAT DEFAULT 0.0
);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    embedding JSONB NOT NULL,
    quality_score FLOAT NOT NULL,
    liveness_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_name VARCHAR(100),
    is_primary BOOLEAN DEFAULT FALSE
);

-- Liveness signatures table
CREATE TABLE IF NOT EXISTS liveness_signatures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    signature JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Auth logs table
CREATE TABLE IF NOT EXISTS auth_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    confidence FLOAT,
    liveness_score FLOAT,
    threshold FLOAT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45),
    user_agent TEXT
);

-- Voice embeddings table
CREATE TABLE IF NOT EXISTS voice_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    embedding JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Challenge sessions table
CREATE TABLE IF NOT EXISTS challenge_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    challenge_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_is_primary ON embeddings(is_primary);
CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON auth_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_logs_event_type ON auth_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_auth_logs_created_at ON auth_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_auth_logs_status ON auth_logs(status);
CREATE INDEX IF NOT EXISTS idx_liveness_signatures_user_id ON liveness_signatures(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_embeddings_user_id ON voice_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_challenge_sessions_user_id ON challenge_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_challenge_sessions_session_token ON challenge_sessions(session_token);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for users table
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE liveness_signatures ENABLE ROW LEVEL SECURITY;
ALTER TABLE auth_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE voice_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE challenge_sessions ENABLE ROW LEVEL SECURITY;

-- Create policies for service role (backend will use service role key)
-- For anon/public access, you might want to restrict these
CREATE POLICY "Service role can do everything" ON users FOR ALL USING (true);
CREATE POLICY "Service role can do everything" ON embeddings FOR ALL USING (true);
CREATE POLICY "Service role can do everything" ON liveness_signatures FOR ALL USING (true);
CREATE POLICY "Service role can do everything" ON auth_logs FOR ALL USING (true);
CREATE POLICY "Service role can do everything" ON voice_embeddings FOR ALL USING (true);
CREATE POLICY "Service role can do everything" ON challenge_sessions FOR ALL USING (true);

-- Create a function to update user statistics
CREATE OR REPLACE FUNCTION update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' AND TG_TABLE_NAME = 'embeddings' THEN
        UPDATE users SET 
            embedding_count = (SELECT COUNT(*) FROM embeddings WHERE user_id = NEW.user_id),
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    ELSIF TG_OP = 'DELETE' AND TG_TABLE_NAME = 'embeddings' THEN
        UPDATE users SET 
            embedding_count = (SELECT COUNT(*) FROM embeddings WHERE user_id = OLD.user_id),
            updated_at = NOW()
        WHERE user_id = OLD.user_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for embeddings
CREATE TRIGGER update_user_embedding_count 
    AFTER INSERT OR DELETE ON embeddings
    FOR EACH ROW EXECUTE FUNCTION update_user_stats();

-- Create a function to update authentication statistics
CREATE OR REPLACE FUNCTION update_auth_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_TABLE_NAME = 'auth_logs' AND NEW.event_type = 'authenticate' THEN
        UPDATE users SET 
            total_authentications = total_authentications + 1,
            last_authentication = NEW.created_at,
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
        
        IF NEW.status = 'success' THEN
            UPDATE users SET 
                successful_authentications = successful_authentications + 1,
                updated_at = NOW()
            WHERE user_id = NEW.user_id;
        ELSE
            UPDATE users SET 
                failed_attempts = failed_attempts + 1,
                updated_at = NOW()
            WHERE user_id = NEW.user_id;
        END IF;
        
        -- Update success rate
        UPDATE users SET 
            success_rate = CASE 
                WHEN total_authentications > 0 
                THEN (successful_authentications::FLOAT / total_authentications::FLOAT) * 100
                ELSE 0
            END,
            updated_at = NOW()
        WHERE user_id = NEW.user_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for auth logs
CREATE TRIGGER update_user_auth_stats 
    AFTER INSERT ON auth_logs
    FOR EACH ROW EXECUTE FUNCTION update_auth_stats();


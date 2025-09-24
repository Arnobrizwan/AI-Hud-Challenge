-- Personalization Service Database Schema

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_interactions INTEGER DEFAULT 0,
    last_interaction_at TIMESTAMP,
    topic_preferences JSONB DEFAULT '{}',
    source_preferences JSONB DEFAULT '{}',
    reading_patterns JSONB DEFAULT '{}',
    collaborative_weight FLOAT DEFAULT 0.5,
    content_weight FLOAT DEFAULT 0.5,
    diversity_preference FLOAT DEFAULT 0.3,
    serendipity_preference FLOAT DEFAULT 0.2,
    demographic_data JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{}'
);

-- User interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    item_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    rating FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context JSONB DEFAULT '{}',
    session_id VARCHAR(255),
    device_type VARCHAR(50),
    location VARCHAR(100),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

-- Content items table
CREATE TABLE IF NOT EXISTS content_items (
    item_id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    content TEXT,
    topics JSONB DEFAULT '[]',
    source VARCHAR(255),
    author VARCHAR(255),
    published_at TIMESTAMP,
    content_features JSONB DEFAULT '{}',
    embedding VECTOR(384), -- For content embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- A/B testing experiments table
CREATE TABLE IF NOT EXISTS ab_experiments (
    experiment_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    variants JSONB NOT NULL,
    traffic_allocation JSONB NOT NULL,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User experiment assignments
CREATE TABLE IF NOT EXISTS user_experiments (
    user_id VARCHAR(255) NOT NULL,
    experiment_id VARCHAR(255) NOT NULL,
    variant VARCHAR(255) NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, experiment_id),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_id) REFERENCES ab_experiments(experiment_id) ON DELETE CASCADE
);

-- Recommendation logs
CREATE TABLE IF NOT EXISTS recommendation_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    algorithm_used VARCHAR(100) NOT NULL,
    recommendations JSONB NOT NULL,
    context JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Cold start profiles
CREATE TABLE IF NOT EXISTS cold_start_profiles (
    profile_type VARCHAR(100) PRIMARY KEY,
    template_profile JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_content_items_topics ON content_items USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_content_items_source ON content_items(source);
CREATE INDEX IF NOT EXISTS idx_recommendation_logs_user_id ON recommendation_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_logs_timestamp ON recommendation_logs(timestamp);

-- Enable vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector index for content embeddings
CREATE INDEX IF NOT EXISTS idx_content_embeddings ON content_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

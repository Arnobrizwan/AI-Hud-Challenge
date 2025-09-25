-- scripts/init-db.sql
-- Initialize database for News Hub Pipeline

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS content;
CREATE SCHEMA IF NOT EXISTS users;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml;

-- Content tables
CREATE TABLE IF NOT EXISTS content.articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    embedding VECTOR(1536)
);

CREATE TABLE IF NOT EXISTS content.categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS content.article_categories (
    article_id UUID REFERENCES content.articles(id) ON DELETE CASCADE,
    category_id UUID REFERENCES content.categories(id) ON DELETE CASCADE,
    confidence FLOAT,
    PRIMARY KEY (article_id, category_id)
);

-- User tables
CREATE TABLE IF NOT EXISTS users.profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users.reading_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    article_id UUID REFERENCES content.articles(id) ON DELETE CASCADE,
    read_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    time_spent INTEGER, -- seconds
    rating INTEGER CHECK (rating >= 1 AND rating <= 5)
);

-- Analytics tables
CREATE TABLE IF NOT EXISTS analytics.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    user_id TEXT,
    article_id UUID REFERENCES content.articles(id) ON DELETE SET NULL,
    properties JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML tables
CREATE TABLE IF NOT EXISTS ml.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'training',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS ml.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ml.models(id) ON DELETE CASCADE,
    input_data JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON content.articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source ON content.articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON content.articles USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_reading_history_user_id ON users.reading_history(user_id);
CREATE INDEX IF NOT EXISTS idx_reading_history_article_id ON users.reading_history(article_id);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON analytics.events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON analytics.events(event_type);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON analytics.metrics(timestamp DESC);

-- Insert sample data
INSERT INTO content.categories (name, description) VALUES 
    ('Technology', 'Tech news and updates'),
    ('Business', 'Business and finance news'),
    ('Politics', 'Political news and analysis'),
    ('Sports', 'Sports news and updates'),
    ('Entertainment', 'Entertainment and celebrity news')
ON CONFLICT (name) DO NOTHING;

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON content.articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON users.profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON ml.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


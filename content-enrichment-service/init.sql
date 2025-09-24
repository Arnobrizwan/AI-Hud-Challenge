-- Database initialization script for Content Enrichment Service

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Create entities table for knowledge base
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    aliases JSONB DEFAULT '[]',
    categories JSONB DEFAULT '[]',
    properties JSONB DEFAULT '{}',
    embedding VECTOR(384), -- For sentence-transformers embeddings
    confidence_score FLOAT DEFAULT 0.8,
    entity_type VARCHAR(50) NOT NULL,
    wikidata_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, entity_type)
);

-- Create index for entity search
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_wikidata_id ON entities(wikidata_id);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops);

-- Create enriched content table for caching
CREATE TABLE IF NOT EXISTS enriched_content (
    id UUID PRIMARY KEY,
    original_content JSONB NOT NULL,
    entities JSONB DEFAULT '[]',
    topics JSONB DEFAULT '[]',
    sentiment JSONB,
    signals JSONB,
    trust_score JSONB,
    enrichment_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_versions JSONB DEFAULT '{}',
    processing_time_ms INTEGER,
    language_detected VARCHAR(10),
    processing_mode VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for enriched content
CREATE INDEX IF NOT EXISTS idx_enriched_content_timestamp ON enriched_content(enrichment_timestamp);
CREATE INDEX IF NOT EXISTS idx_enriched_content_language ON enriched_content(language_detected);
CREATE INDEX IF NOT EXISTS idx_enriched_content_mode ON enriched_content(processing_mode);

-- Create enrichment requests table for tracking
CREATE TABLE IF NOT EXISTS enrichment_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id VARCHAR(255) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    entities_count INTEGER DEFAULT 0,
    topics_count INTEGER DEFAULT 0,
    processing_mode VARCHAR(20),
    user_id VARCHAR(255),
    ip_address INET
);

-- Create index for enrichment requests
CREATE INDEX IF NOT EXISTS idx_enrichment_requests_timestamp ON enrichment_requests(request_timestamp);
CREATE INDEX IF NOT EXISTS idx_enrichment_requests_success ON enrichment_requests(success);
CREATE INDEX IF NOT EXISTS idx_enrichment_requests_user ON enrichment_requests(user_id);

-- Create model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    evaluation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_name VARCHAR(100),
    evaluation_split VARCHAR(20)
);

-- Create index for model performance
CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_model_performance_metric ON model_performance(metric_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(evaluation_timestamp);

-- Create A/B testing table
CREATE TABLE IF NOT EXISTS ab_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_name VARCHAR(100) NOT NULL,
    variant_name VARCHAR(50) NOT NULL,
    content_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    assignment_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metrics JSONB DEFAULT '{}',
    UNIQUE(test_name, content_id)
);

-- Create index for A/B tests
CREATE INDEX IF NOT EXISTS idx_ab_tests_name ON ab_tests(test_name);
CREATE INDEX IF NOT EXISTS idx_ab_tests_variant ON ab_tests(variant_name);
CREATE INDEX IF NOT EXISTS idx_ab_tests_content ON ab_tests(content_id);

-- Create content quality metrics table
CREATE TABLE IF NOT EXISTS content_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id VARCHAR(255) NOT NULL,
    readability_score FLOAT,
    factual_claims_count INTEGER DEFAULT 0,
    citations_count INTEGER DEFAULT 0,
    bias_score FLOAT,
    political_leaning VARCHAR(20),
    engagement_prediction FLOAT,
    virality_potential FLOAT,
    content_freshness FLOAT,
    authority_score FLOAT,
    trust_score FLOAT,
    processing_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for content quality metrics
CREATE INDEX IF NOT EXISTS idx_content_quality_content ON content_quality_metrics(content_id);
CREATE INDEX IF NOT EXISTS idx_content_quality_timestamp ON content_quality_metrics(processing_timestamp);
CREATE INDEX IF NOT EXISTS idx_content_quality_trust ON content_quality_metrics(trust_score);

-- Create MLflow tracking database
CREATE DATABASE IF NOT EXISTS mlflow;

-- Connect to MLflow database
\c mlflow;

-- MLflow tables (simplified)
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    name VARCHAR(256) NOT NULL UNIQUE,
    artifact_location VARCHAR(256),
    lifecycle_stage VARCHAR(32),
    creation_time BIGINT,
    last_update_time BIGINT
);

CREATE TABLE IF NOT EXISTS runs (
    run_uuid VARCHAR(32) PRIMARY KEY,
    name VARCHAR(250),
    source_type VARCHAR(20),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(50),
    user_id VARCHAR(64),
    status VARCHAR(9),
    start_time BIGINT,
    end_time BIGINT,
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(32),
    artifact_uri VARCHAR(200),
    experiment_id INTEGER REFERENCES experiments(experiment_id)
);

-- Switch back to main database
\c content_enrichment;

-- Create views for common queries
CREATE OR REPLACE VIEW enrichment_stats AS
SELECT 
    DATE_TRUNC('hour', request_timestamp) as hour,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE success = TRUE) as successful_requests,
    COUNT(*) FILTER (WHERE success = FALSE) as failed_requests,
    AVG(processing_time_ms) as avg_processing_time_ms,
    AVG(entities_count) as avg_entities_count,
    AVG(topics_count) as avg_topics_count
FROM enrichment_requests
GROUP BY DATE_TRUNC('hour', request_timestamp)
ORDER BY hour DESC;

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model_name,
    model_version,
    metric_name,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    COUNT(*) as evaluation_count,
    MAX(evaluation_timestamp) as last_evaluation
FROM model_performance
GROUP BY model_name, model_version, metric_name
ORDER BY model_name, model_version, metric_name;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_entity_embedding(
    entity_id UUID,
    new_embedding VECTOR(384)
) RETURNS VOID AS $$
BEGIN
    UPDATE entities 
    SET embedding = new_embedding, updated_at = NOW()
    WHERE id = entity_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_similar_entities(
    query_embedding VECTOR(384),
    similarity_threshold FLOAT DEFAULT 0.7,
    limit_count INTEGER DEFAULT 10
) RETURNS TABLE(
    id UUID,
    name VARCHAR(255),
    description TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id,
        e.name,
        e.description,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM entities e
    WHERE 1 - (e.embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data
INSERT INTO entities (name, description, entity_type, wikidata_id) VALUES
('Apple Inc.', 'American multinational technology company', 'ORG', 'Q95'),
('Tim Cook', 'CEO of Apple Inc.', 'PERSON', 'Q312'),
('iPhone', 'Smartphone product line by Apple', 'PRODUCT', 'Q78'),
('Machine Learning', 'Field of artificial intelligence', 'CUSTOM', 'Q2539'),
('Artificial Intelligence', 'Intelligence demonstrated by machines', 'CUSTOM', 'Q11660')
ON CONFLICT (name, entity_type) DO NOTHING;

-- Create triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO content_enrichment_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO content_enrichment_user;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO content_enrichment_user;

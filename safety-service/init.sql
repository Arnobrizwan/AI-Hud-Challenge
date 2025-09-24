-- Safety Service Database Initialization
-- This script creates the necessary tables and indexes for the safety service

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS safety_db;

-- Use the safety database
\c safety_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create audit events table
CREATE TABLE IF NOT EXISTS audit_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    request_id VARCHAR(255),
    source_ip INET,
    user_agent TEXT,
    resource VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    details JSONB,
    metadata JSONB,
    tags TEXT[],
    correlation_id VARCHAR(255),
    parent_event_id UUID,
    duration_ms FLOAT,
    error_code VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for audit events
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_severity ON audit_events(severity);
CREATE INDEX IF NOT EXISTS idx_audit_events_status ON audit_events(status);
CREATE INDEX IF NOT EXISTS idx_audit_events_correlation_id ON audit_events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_tags ON audit_events USING GIN(tags);

-- Create safety checks table
CREATE TABLE IF NOT EXISTS safety_checks (
    check_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    request_id VARCHAR(255),
    check_type VARCHAR(50) NOT NULL,
    overall_score FLOAT NOT NULL,
    requires_intervention BOOLEAN NOT NULL DEFAULT FALSE,
    drift_score FLOAT,
    abuse_score FLOAT,
    content_score FLOAT,
    anomaly_score FLOAT,
    rate_limit_score FLOAT,
    details JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for safety checks
CREATE INDEX IF NOT EXISTS idx_safety_checks_user_id ON safety_checks(user_id);
CREATE INDEX IF NOT EXISTS idx_safety_checks_check_type ON safety_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_safety_checks_created_at ON safety_checks(created_at);
CREATE INDEX IF NOT EXISTS idx_safety_checks_overall_score ON safety_checks(overall_score);

-- Create drift detection results table
CREATE TABLE IF NOT EXISTS drift_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_id UUID REFERENCES safety_checks(check_id),
    drift_type VARCHAR(50) NOT NULL,
    overall_severity FLOAT NOT NULL,
    requires_action BOOLEAN NOT NULL DEFAULT FALSE,
    drifted_features TEXT[],
    feature_results JSONB,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for drift results
CREATE INDEX IF NOT EXISTS idx_drift_results_check_id ON drift_results(check_id);
CREATE INDEX IF NOT EXISTS idx_drift_results_drift_type ON drift_results(drift_type);
CREATE INDEX IF NOT EXISTS idx_drift_results_created_at ON drift_results(created_at);

-- Create abuse detection results table
CREATE TABLE IF NOT EXISTS abuse_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_id UUID REFERENCES safety_checks(check_id),
    user_id VARCHAR(255) NOT NULL,
    abuse_score FLOAT NOT NULL,
    threat_level VARCHAR(20) NOT NULL,
    rule_violations TEXT[],
    behavioral_signals JSONB,
    graph_signals JSONB,
    ml_prediction JSONB,
    reputation_score FLOAT,
    response_actions JSONB,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for abuse results
CREATE INDEX IF NOT EXISTS idx_abuse_results_check_id ON abuse_results(check_id);
CREATE INDEX IF NOT EXISTS idx_abuse_results_user_id ON abuse_results(user_id);
CREATE INDEX IF NOT EXISTS idx_abuse_results_threat_level ON abuse_results(threat_level);
CREATE INDEX IF NOT EXISTS idx_abuse_results_created_at ON abuse_results(created_at);

-- Create content moderation results table
CREATE TABLE IF NOT EXISTS content_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_id UUID REFERENCES safety_checks(check_id),
    content_id VARCHAR(255) NOT NULL,
    overall_safety_score FLOAT NOT NULL,
    toxicity_score FLOAT,
    hate_speech_score FLOAT,
    spam_score FLOAT,
    misinformation_score FLOAT,
    adult_content_score FLOAT,
    violence_score FLOAT,
    violations TEXT[],
    recommended_action VARCHAR(50),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for content results
CREATE INDEX IF NOT EXISTS idx_content_results_check_id ON content_results(check_id);
CREATE INDEX IF NOT EXISTS idx_content_results_content_id ON content_results(content_id);
CREATE INDEX IF NOT EXISTS idx_content_results_created_at ON content_results(created_at);

-- Create rate limiting results table
CREATE TABLE IF NOT EXISTS rate_limit_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    check_id UUID REFERENCES safety_checks(check_id),
    user_id VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    is_rate_limited BOOLEAN NOT NULL DEFAULT FALSE,
    triggered_limits TEXT[],
    remaining_capacity INTEGER,
    retry_after INTEGER,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for rate limit results
CREATE INDEX IF NOT EXISTS idx_rate_limit_results_check_id ON rate_limit_results(check_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_results_user_id ON rate_limit_results(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_results_endpoint ON rate_limit_results(endpoint);
CREATE INDEX IF NOT EXISTS idx_rate_limit_results_created_at ON rate_limit_results(created_at);

-- Create incidents table
CREATE TABLE IF NOT EXISTS incidents (
    incident_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    description TEXT NOT NULL,
    affected_systems TEXT[],
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for incidents
CREATE INDEX IF NOT EXISTS idx_incidents_incident_type ON incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON incidents(detected_at);

-- Create user reputation table
CREATE TABLE IF NOT EXISTS user_reputation (
    user_id VARCHAR(255) PRIMARY KEY,
    reputation_score FLOAT NOT NULL DEFAULT 0.5,
    abuse_count INTEGER NOT NULL DEFAULT 0,
    good_behavior_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for user reputation
CREATE INDEX IF NOT EXISTS idx_user_reputation_score ON user_reputation(reputation_score);
CREATE INDEX IF NOT EXISTS idx_user_reputation_last_updated ON user_reputation(last_updated);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for system metrics
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type);

-- Create configuration table
CREATE TABLE IF NOT EXISTS configuration (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for configuration
CREATE INDEX IF NOT EXISTS idx_configuration_key ON configuration(config_key);
CREATE INDEX IF NOT EXISTS idx_configuration_active ON configuration(is_active);

-- Insert default configuration
INSERT INTO configuration (config_key, config_value, description) VALUES
('drift_threshold', '0.1', 'Default drift detection threshold'),
('abuse_threshold', '0.5', 'Default abuse detection threshold'),
('content_safety_threshold', '0.7', 'Default content safety threshold'),
('rate_limit_default', '100', 'Default rate limit per minute'),
('retention_days', '90', 'Default data retention period')
ON CONFLICT (config_key) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW safety_check_summary AS
SELECT 
    DATE(created_at) as check_date,
    COUNT(*) as total_checks,
    AVG(overall_score) as avg_score,
    COUNT(CASE WHEN requires_intervention THEN 1 END) as intervention_count,
    COUNT(CASE WHEN overall_score < 0.5 THEN 1 END) as low_score_count
FROM safety_checks
GROUP BY DATE(created_at)
ORDER BY check_date DESC;

CREATE OR REPLACE VIEW incident_summary AS
SELECT 
    incident_type,
    severity,
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))/3600) as avg_resolution_hours
FROM incidents
GROUP BY incident_type, severity, status
ORDER BY count DESC;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_user_reputation(
    p_user_id VARCHAR(255),
    p_score_change FLOAT,
    p_abuse_incident BOOLEAN DEFAULT FALSE
) RETURNS VOID AS $$
BEGIN
    INSERT INTO user_reputation (user_id, reputation_score, abuse_count, good_behavior_count)
    VALUES (p_user_id, 0.5 + p_score_change, 
            CASE WHEN p_abuse_incident THEN 1 ELSE 0 END,
            CASE WHEN p_abuse_incident THEN 0 ELSE 1 END)
    ON CONFLICT (user_id) DO UPDATE SET
        reputation_score = GREATEST(0.0, LEAST(1.0, reputation_score + p_score_change)),
        abuse_count = abuse_count + CASE WHEN p_abuse_incident THEN 1 ELSE 0 END,
        good_behavior_count = good_behavior_count + CASE WHEN p_abuse_incident THEN 0 ELSE 1 END,
        last_updated = NOW();
END;
$$ LANGUAGE plpgsql;

-- Create function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data(p_retention_days INTEGER DEFAULT 90) RETURNS VOID AS $$
BEGIN
    DELETE FROM audit_events WHERE created_at < NOW() - INTERVAL '1 day' * p_retention_days;
    DELETE FROM safety_checks WHERE created_at < NOW() - INTERVAL '1 day' * p_retention_days;
    DELETE FROM system_metrics WHERE timestamp < NOW() - INTERVAL '1 day' * p_retention_days;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO safety_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO safety_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO safety_user;

-- Create a user for the application
CREATE USER IF NOT EXISTS safety_user WITH PASSWORD 'safety_pass';
GRANT ALL PRIVILEGES ON DATABASE safety_db TO safety_user;

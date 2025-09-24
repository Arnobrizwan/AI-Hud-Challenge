-- MLOps Orchestration Service Database Schema
-- PostgreSQL initialization script

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS mlops;

-- Use the database
\c mlops;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS pipelines;
CREATE SCHEMA IF NOT EXISTS training;
CREATE SCHEMA IF NOT EXISTS deployment;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS retraining;

-- Pipelines table
CREATE TABLE IF NOT EXISTS pipelines.pipelines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    pipeline_type VARCHAR(50) NOT NULL,
    orchestrator VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    config JSONB NOT NULL,
    airflow_dag_id VARCHAR(255),
    vertex_pipeline_id VARCHAR(255),
    kubeflow_pipeline_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    tags JSONB DEFAULT '{}'::jsonb
);

-- Pipeline executions table
CREATE TABLE IF NOT EXISTS pipelines.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pipeline_id UUID NOT NULL REFERENCES pipelines.pipelines(id) ON DELETE CASCADE,
    execution_params JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    external_run_id VARCHAR(255),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    triggered_by VARCHAR(255) DEFAULT 'manual',
    error_message TEXT,
    outputs JSONB DEFAULT '{}'::jsonb,
    metrics JSONB DEFAULT '{}'::jsonb
);

-- Training jobs table
CREATE TABLE IF NOT EXISTS training.training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    experiment_id VARCHAR(255),
    best_hyperparameters JSONB,
    model_result JSONB,
    evaluation_result JSONB,
    registered_model_version JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_by VARCHAR(255)
);

-- Deployments table
CREATE TABLE IF NOT EXISTS deployment.deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(255) NOT NULL,
    deployment_strategy VARCHAR(50) NOT NULL,
    target_environment VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    config JSONB NOT NULL,
    endpoint_url VARCHAR(500),
    endpoint_name VARCHAR(255),
    deployment_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT,
    created_by VARCHAR(255),
    tags JSONB DEFAULT '{}'::jsonb
);

-- Feature sets table
CREATE TABLE IF NOT EXISTS features.feature_sets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    features JSONB NOT NULL,
    source_config JSONB NOT NULL,
    update_schedule VARCHAR(100),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    vertex_feature_set_name VARCHAR(255),
    ingestion_pipeline_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    tags JSONB DEFAULT '{}'::jsonb
);

-- Feature serving requests table
CREATE TABLE IF NOT EXISTS features.serving_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_store_name VARCHAR(255) NOT NULL,
    feature_set_names JSONB NOT NULL,
    entity_ids JSONB NOT NULL,
    transformations JSONB,
    cache_ttl INTEGER DEFAULT 3600,
    request_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time TIMESTAMP WITH TIME ZONE,
    cache_hit BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- Monitoring configurations table
CREATE TABLE IF NOT EXISTS monitoring.monitoring_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Model metrics table
CREATE TABLE IF NOT EXISTS monitoring.model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    metrics JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alerts table
CREATE TABLE IF NOT EXISTS monitoring.alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolved_by VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Retraining triggers table
CREATE TABLE IF NOT EXISTS retraining.triggers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    config JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_fired_at TIMESTAMP WITH TIME ZONE,
    fire_count INTEGER DEFAULT 0
);

-- Retraining results table
CREATE TABLE IF NOT EXISTS retraining.results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    trigger_id UUID REFERENCES retraining.triggers(id) ON DELETE SET NULL,
    trigger_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    config JSONB NOT NULL,
    training_result JSONB,
    ab_test_id VARCHAR(255),
    performance_comparison JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_by VARCHAR(255)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pipelines_name ON pipelines.pipelines(name);
CREATE INDEX IF NOT EXISTS idx_pipelines_status ON pipelines.pipelines(status);
CREATE INDEX IF NOT EXISTS idx_pipelines_created_at ON pipelines.pipelines(created_at);

CREATE INDEX IF NOT EXISTS idx_executions_pipeline_id ON pipelines.executions(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON pipelines.executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_started_at ON pipelines.executions(started_at);

CREATE INDEX IF NOT EXISTS idx_training_jobs_model_name ON training.training_jobs(model_name);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training.training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_started_at ON training.training_jobs(started_at);

CREATE INDEX IF NOT EXISTS idx_deployments_model_name ON deployment.deployments(model_name);
CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployment.deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_created_at ON deployment.deployments(created_at);

CREATE INDEX IF NOT EXISTS idx_feature_sets_name ON features.feature_sets(name);
CREATE INDEX IF NOT EXISTS idx_feature_sets_status ON features.feature_sets(status);

CREATE INDEX IF NOT EXISTS idx_serving_requests_model_name ON features.serving_requests(feature_store_name);
CREATE INDEX IF NOT EXISTS idx_serving_requests_request_time ON features.serving_requests(request_time);

CREATE INDEX IF NOT EXISTS idx_monitoring_configs_model_name ON monitoring.monitoring_configs(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON monitoring.model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp ON monitoring.model_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_alerts_model_name ON monitoring.alerts(model_name);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON monitoring.alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON monitoring.alerts(triggered_at);

CREATE INDEX IF NOT EXISTS idx_triggers_model_name ON retraining.triggers(model_name);
CREATE INDEX IF NOT EXISTS idx_triggers_status ON retraining.triggers(status);

CREATE INDEX IF NOT EXISTS idx_retraining_results_model_name ON retraining.results(model_name);
CREATE INDEX IF NOT EXISTS idx_retraining_results_status ON retraining.results(status);
CREATE INDEX IF NOT EXISTS idx_retraining_results_started_at ON retraining.results(started_at);

-- Create functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updating timestamps
CREATE TRIGGER update_pipelines_updated_at BEFORE UPDATE ON pipelines.pipelines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_deployments_updated_at BEFORE UPDATE ON deployment.deployments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feature_sets_updated_at BEFORE UPDATE ON features.feature_sets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_monitoring_configs_updated_at BEFORE UPDATE ON monitoring.monitoring_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW pipelines.pipeline_summary AS
SELECT 
    p.id,
    p.name,
    p.pipeline_type,
    p.orchestrator,
    p.status,
    p.created_at,
    p.updated_at,
    COUNT(e.id) as execution_count,
    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_executions
FROM pipelines.pipelines p
LEFT JOIN pipelines.executions e ON p.id = e.pipeline_id
GROUP BY p.id, p.name, p.pipeline_type, p.orchestrator, p.status, p.created_at, p.updated_at;

CREATE OR REPLACE VIEW monitoring.model_health_summary AS
SELECT 
    model_name,
    COUNT(*) as total_metrics,
    AVG((metrics->>'accuracy')::float) as avg_accuracy,
    AVG((metrics->>'latency_ms')::float) as avg_latency_ms,
    AVG((metrics->>'error_rate')::float) as avg_error_rate,
    MAX(timestamp) as last_metric_time
FROM monitoring.model_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY model_name;

-- Insert sample data
INSERT INTO pipelines.pipelines (name, description, pipeline_type, orchestrator, config, created_by) VALUES
('customer_churn_prediction', 'Predict customer churn using ML', 'training', 'airflow', 
 '{"model_class": "sklearn.ensemble.RandomForestClassifier", "enable_hyperparameter_tuning": true}', 'admin'),
('fraud_detection', 'Detect fraudulent transactions', 'training', 'vertex_ai',
 '{"model_class": "sklearn.ensemble.IsolationForest", "enable_hyperparameter_tuning": false}', 'admin');

INSERT INTO features.feature_sets (name, description, features, source_config, created_by) VALUES
('customer_features', 'Customer demographic and behavioral features',
 '[{"name": "age", "type": "numerical"}, {"name": "income", "type": "numerical"}]',
 '{"type": "bigquery", "query": "SELECT * FROM customer_data"}', 'admin'),
('transaction_features', 'Transaction history features',
 '[{"name": "transaction_amount", "type": "numerical"}, {"name": "transaction_frequency", "type": "numerical"}]',
 '{"type": "bigquery", "query": "SELECT * FROM transactions"}', 'admin');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pipelines TO mlops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA training TO mlops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA deployment TO mlops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO mlops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA features TO mlops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA retraining TO mlops;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pipelines TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA training TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA deployment TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA features TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA retraining TO mlops;

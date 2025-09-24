-- Feedback Service Database Schema
-- This file initializes the PostgreSQL database with all required tables

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE feedback_type AS ENUM ('explicit', 'implicit', 'crowdsourced', 'editorial');
CREATE TYPE signal_type AS ENUM ('click', 'dwell_time', 'share', 'like', 'dislike', 'rating', 'comment', 'complaint', 'report');
CREATE TYPE task_status AS ENUM ('pending', 'assigned', 'in_progress', 'completed', 'cancelled', 'overdue');
CREATE TYPE task_priority AS ENUM ('low', 'normal', 'high', 'urgent');
CREATE TYPE review_decision AS ENUM ('approve', 'reject', 'request_changes', 'escalate');
CREATE TYPE annotation_type AS ENUM ('sentiment', 'topic', 'quality', 'bias', 'factual', 'completeness');
CREATE TYPE campaign_status AS ENUM ('draft', 'active', 'paused', 'completed', 'cancelled');
CREATE TYPE quality_level AS ENUM ('low', 'medium', 'high', 'expert');

-- Users and roles
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL DEFAULT 'annotator',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preferences JSONB DEFAULT '{}',
    notification_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content items
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255),
    title TEXT,
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    category VARCHAR(100),
    source VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback collection
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    content_id UUID REFERENCES content_items(id),
    feedback_type feedback_type NOT NULL,
    signal_type signal_type,
    rating DECIMAL(3,2) CHECK (rating >= 0 AND rating <= 5),
    comment TEXT,
    metadata JSONB DEFAULT '{}',
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE feedback_processing_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id UUID REFERENCES feedback(id) ON DELETE CASCADE,
    quality_score DECIMAL(3,2),
    sentiment_score DECIMAL(3,2),
    confidence_score DECIMAL(3,2),
    actions_taken JSONB DEFAULT '[]',
    processing_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Editorial workflow
CREATE TABLE review_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES content_items(id),
    task_type VARCHAR(100) NOT NULL,
    priority task_priority DEFAULT 'normal',
    assigned_to UUID REFERENCES users(id),
    created_by UUID REFERENCES users(id),
    status task_status DEFAULT 'pending',
    due_date TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE review_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES review_tasks(id) ON DELETE CASCADE,
    reviewer_id UUID REFERENCES users(id),
    decision review_decision NOT NULL,
    comments TEXT,
    changes_requested TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Annotation system
CREATE TABLE annotation_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES content_items(id),
    annotation_type annotation_type NOT NULL,
    guidelines TEXT,
    deadline TIMESTAMP WITH TIME ZONE,
    status task_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE annotation_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES annotation_tasks(id) ON DELETE CASCADE,
    annotator_id UUID REFERENCES users(id),
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES annotation_tasks(id) ON DELETE CASCADE,
    annotator_id UUID REFERENCES users(id),
    annotation_data JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    time_spent_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quality assurance
CREATE TABLE quality_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES content_items(id),
    overall_quality_score DECIMAL(3,2),
    factual_accuracy DECIMAL(3,2),
    bias_score DECIMAL(3,2),
    readability_score DECIMAL(3,2),
    completeness_score DECIMAL(3,2),
    spam_likelihood DECIMAL(3,2),
    needs_human_review BOOLEAN DEFAULT FALSE,
    assessment_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE moderation_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES content_items(id),
    action VARCHAR(50) NOT NULL,
    reason TEXT,
    severity VARCHAR(20),
    moderator_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Crowdsourcing
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    task_type VARCHAR(100) NOT NULL,
    target_annotations INTEGER,
    reward_per_task DECIMAL(10,2),
    quality_threshold DECIMAL(3,2),
    status campaign_status DEFAULT 'draft',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE campaign_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id),
    task_data JSONB NOT NULL,
    status task_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE campaign_submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID REFERENCES campaign_tasks(id) ON DELETE CASCADE,
    worker_id VARCHAR(255),
    submission_data JSONB NOT NULL,
    quality_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model training and updates
CREATE TABLE training_batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    batch_data JSONB NOT NULL,
    example_count INTEGER,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE model_updates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    training_batch_id UUID REFERENCES training_batches(id),
    performance_metrics JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analytics and insights
CREATE TABLE feedback_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insight_type VARCHAR(100) NOT NULL,
    insight_data JSONB NOT NULL,
    time_window_start TIMESTAMP WITH TIME ZONE,
    time_window_end TIMESTAMP WITH TIME ZONE,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    model_name VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Audit trail
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_feedback_user_id ON feedback(user_id);
CREATE INDEX idx_feedback_content_id ON feedback(content_id);
CREATE INDEX idx_feedback_created_at ON feedback(created_at);
CREATE INDEX idx_feedback_type ON feedback(feedback_type);

CREATE INDEX idx_review_tasks_assigned_to ON review_tasks(assigned_to);
CREATE INDEX idx_review_tasks_status ON review_tasks(status);
CREATE INDEX idx_review_tasks_due_date ON review_tasks(due_date);

CREATE INDEX idx_annotations_task_id ON annotations(task_id);
CREATE INDEX idx_annotations_annotator_id ON annotations(annotator_id);

CREATE INDEX idx_quality_assessments_content_id ON quality_assessments(content_id);
CREATE INDEX idx_quality_assessments_created_at ON quality_assessments(created_at);

CREATE INDEX idx_campaign_tasks_campaign_id ON campaign_tasks(campaign_id);
CREATE INDEX idx_campaign_tasks_status ON campaign_tasks(status);

CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_performance_metrics_model_name ON performance_metrics(model_name);

-- Full-text search indexes
CREATE INDEX idx_content_items_content_fts ON content_items USING gin(to_tsvector('english', content));
CREATE INDEX idx_feedback_comment_fts ON feedback USING gin(to_tsvector('english', comment));

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_items_updated_at BEFORE UPDATE ON content_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_review_tasks_updated_at BEFORE UPDATE ON review_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotation_tasks_updated_at BEFORE UPDATE ON annotation_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user
INSERT INTO users (username, email, full_name, role) VALUES 
('admin', 'admin@feedback-service.com', 'System Administrator', 'admin'),
('editor', 'editor@feedback-service.com', 'Content Editor', 'editor'),
('annotator', 'annotator@feedback-service.com', 'Content Annotator', 'annotator');

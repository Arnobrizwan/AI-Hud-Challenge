-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create articles table
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    quality_score FLOAT DEFAULT 0.0,
    content_hash TEXT,
    title_hash TEXT,
    embedding VECTOR(384), -- For sentence-BERT embeddings
    entities JSONB DEFAULT '[]',
    topics JSONB DEFAULT '[]',
    locations JSONB DEFAULT '[]',
    language TEXT DEFAULT 'en',
    word_count INTEGER DEFAULT 0,
    reading_time INTEGER DEFAULT 0
);

-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    representative_article_id UUID REFERENCES articles(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    article_count INTEGER DEFAULT 0,
    quality_score FLOAT DEFAULT 0.0,
    topics JSONB DEFAULT '[]',
    entities JSONB DEFAULT '[]',
    locations JSONB DEFAULT '[]',
    time_span INTERVAL,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create article_clusters table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS article_clusters (
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    cluster_id UUID REFERENCES clusters(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (article_id, cluster_id)
);

-- Create duplicates table
CREATE TABLE IF NOT EXISTS duplicates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    duplicate_of_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    similarity_type TEXT NOT NULL, -- 'lsh', 'semantic', 'content'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, duplicate_of_id)
);

-- Create LSH index table
CREATE TABLE IF NOT EXISTS lsh_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    minhash_signature BYTEA NOT NULL,
    content_fingerprint TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    priority INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_articles_title_hash ON articles(title_hash);
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_articles_quality_score ON articles(quality_score);
CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language);

CREATE INDEX IF NOT EXISTS idx_clusters_created_at ON clusters(created_at);
CREATE INDEX IF NOT EXISTS idx_clusters_quality_score ON clusters(quality_score);
CREATE INDEX IF NOT EXISTS idx_clusters_is_active ON clusters(is_active);

CREATE INDEX IF NOT EXISTS idx_article_clusters_article_id ON article_clusters(article_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_cluster_id ON article_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_similarity ON article_clusters(similarity_score);

CREATE INDEX IF NOT EXISTS idx_duplicates_article_id ON duplicates(article_id);
CREATE INDEX IF NOT EXISTS idx_duplicates_duplicate_of_id ON duplicates(duplicate_of_id);
CREATE INDEX IF NOT EXISTS idx_duplicates_similarity ON duplicates(similarity_score);

CREATE INDEX IF NOT EXISTS idx_lsh_index_article_id ON lsh_index(article_id);
CREATE INDEX IF NOT EXISTS idx_lsh_index_content_fingerprint ON lsh_index(content_fingerprint);

CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status);
CREATE INDEX IF NOT EXISTS idx_processing_queue_priority ON processing_queue(priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_processing_queue_created_at ON processing_queue(created_at);

-- Create functions for similarity calculations
CREATE OR REPLACE FUNCTION cosine_similarity(a VECTOR, b VECTOR) 
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to update article count in clusters
CREATE OR REPLACE FUNCTION update_cluster_article_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE clusters 
        SET article_count = article_count + 1,
            updated_at = NOW()
        WHERE id = NEW.cluster_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE clusters 
        SET article_count = article_count - 1,
            updated_at = NOW()
        WHERE id = OLD.cluster_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_cluster_article_count
    AFTER INSERT OR DELETE ON article_clusters
    FOR EACH ROW EXECUTE FUNCTION update_cluster_article_count();

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER trigger_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_clusters_updated_at
    BEFORE UPDATE ON clusters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_processing_queue_updated_at
    BEFORE UPDATE ON processing_queue
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

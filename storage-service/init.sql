-- Initialize PostgreSQL with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create articles table
CREATE TABLE IF NOT EXISTS articles (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    author VARCHAR(255),
    source VARCHAR(255) NOT NULL,
    published_at TIMESTAMP,
    categories TEXT[],
    tags TEXT[],
    language VARCHAR(10) DEFAULT 'en',
    url VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector embeddings table
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    content_id VARCHAR(255) NOT NULL,
    embedding_type VARCHAR(100) NOT NULL,
    vector VECTOR(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(content_id, embedding_type)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_categories ON articles USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_articles_tags ON articles USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_articles_metadata ON articles USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_articles_fulltext ON articles USING GIN(
    to_tsvector('english', title || ' ' || COALESCE(content, '') || ' ' || COALESCE(summary, ''))
);

-- Create vector indexes
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_content_id ON vector_embeddings(content_id);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_type ON vector_embeddings(embedding_type);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_metadata ON vector_embeddings USING GIN(metadata);

-- Create HNSW index for approximate similarity search
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_hnsw ON vector_embeddings 
USING hnsw (vector vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vector_embeddings_updated_at BEFORE UPDATE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

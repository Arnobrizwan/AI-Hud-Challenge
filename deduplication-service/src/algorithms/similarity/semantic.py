"""Semantic similarity calculation using sentence embeddings."""

import asyncio
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..lsh.minhash import MinHashGenerator
from ...models.schemas import NormalizedArticle, Entity, Topic, Location


class SemanticSimilarityCalculator:
    """Semantic similarity calculator using sentence embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        batch_size: int = 32
    ):
        """Initialize semantic similarity calculator.
        
        Args:
            model_name: Name of the sentence transformer model
            embedding_dimension: Dimension of the embeddings
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Initialize MinHash generator for text preprocessing
        self.minhash_gen = MinHashGenerator()
    
    async def compute_article_embedding(self, article: NormalizedArticle) -> np.ndarray:
        """Compute embedding for an article.
        
        Args:
            article: Normalized article
            
        Returns:
            Article embedding vector
        """
        # Combine title and content for embedding
        text = f"{article.title} {article.summary or ''} {article.content[:1000]}"
        
        # Compute embedding
        embedding = await self._compute_embedding(text)
        
        return embedding
    
    async def compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding vector
        """
        return await self._compute_embedding(text)
    
    async def compute_similarity(
        self,
        article1: NormalizedArticle,
        article2: NormalizedArticle
    ) -> float:
        """Compute semantic similarity between two articles.
        
        Args:
            article1: First article
            article2: Second article
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Compute embeddings
        embedding1 = await self.compute_article_embedding(article1)
        embedding2 = await self.compute_article_embedding(article2)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    async def compute_batch_similarity(
        self,
        articles: List[NormalizedArticle],
        target_article: NormalizedArticle
    ) -> List[float]:
        """Compute similarity between target article and a batch of articles.
        
        Args:
            articles: List of articles to compare
            target_article: Target article for comparison
            
        Returns:
            List of similarity scores
        """
        # Compute target embedding
        target_embedding = await self.compute_article_embedding(target_article)
        
        # Compute embeddings for batch
        batch_embeddings = []
        for article in articles:
            embedding = await self.compute_article_embedding(article)
            batch_embeddings.append(embedding)
        
        # Convert to numpy array
        batch_embeddings = np.array(batch_embeddings)
        
        # Compute similarities
        similarities = cosine_similarity(
            target_embedding.reshape(1, -1),
            batch_embeddings
        )[0]
        
        return similarities.tolist()
    
    async def find_similar_articles(
        self,
        target_article: NormalizedArticle,
        candidate_articles: List[NormalizedArticle],
        threshold: float = 0.85
    ) -> List[Tuple[NormalizedArticle, float]]:
        """Find articles similar to target article.
        
        Args:
            target_article: Target article
            candidate_articles: List of candidate articles
            threshold: Similarity threshold
            
        Returns:
            List of (article, similarity) tuples
        """
        if not candidate_articles:
            return []
        
        # Compute similarities
        similarities = await self.compute_batch_similarity(
            candidate_articles, target_article
        )
        
        # Filter by threshold and sort by similarity
        similar_articles = []
        for article, similarity in zip(candidate_articles, similarities):
            if similarity >= threshold:
                similar_articles.append((article, similarity))
        
        # Sort by similarity (descending)
        similar_articles.sort(key=lambda x: x[1], reverse=True)
        
        return similar_articles
    
    async def compute_entity_similarity(
        self,
        entities1: List[Entity],
        entities2: List[Entity]
    ) -> float:
        """Compute similarity based on entity overlap.
        
        Args:
            entities1: First set of entities
            entities2: Second set of entities
            
        Returns:
            Entity similarity score (0.0 to 1.0)
        """
        if not entities1 or not entities2:
            return 0.0
        
        # Extract entity texts
        texts1 = [entity.text.lower() for entity in entities1]
        texts2 = [entity.text.lower() for entity in entities2]
        
        # Compute Jaccard similarity
        set1 = set(texts1)
        set2 = set(texts2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    async def compute_topic_similarity(
        self,
        topics1: List[Topic],
        topics2: List[Topic]
    ) -> float:
        """Compute similarity based on topic overlap.
        
        Args:
            topics1: First set of topics
            topics2: Second set of topics
            
        Returns:
            Topic similarity score (0.0 to 1.0)
        """
        if not topics1 or not topics2:
            return 0.0
        
        # Extract topic names
        names1 = [topic.name.lower() for topic in topics1]
        names2 = [topic.name.lower() for topic in topics2]
        
        # Compute Jaccard similarity
        set1 = set(names1)
        set2 = set(names2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    async def compute_location_similarity(
        self,
        locations1: List[Location],
        locations2: List[Location]
    ) -> float:
        """Compute similarity based on location overlap.
        
        Args:
            locations1: First set of locations
            locations2: Second set of locations
            
        Returns:
            Location similarity score (0.0 to 1.0)
        """
        if not locations1 or not locations2:
            return 0.0
        
        # Extract location names
        names1 = [loc.name.lower() for loc in locations1]
        names2 = [loc.name.lower() for loc in locations2]
        
        # Compute Jaccard similarity
        set1 = set(names1)
        set2 = set(names2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text using sentence transformer.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.model.encode, text
        )
        
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Basic text preprocessing
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text


class ContentSimilarityCalculator:
    """Content-based similarity calculator."""
    
    def __init__(self, minhash_generator: MinHashGenerator):
        """Initialize content similarity calculator.
        
        Args:
            minhash_generator: MinHash generator instance
        """
        self.minhash_gen = minhash_generator
    
    async def compute_title_similarity(
        self,
        title1: str,
        title2: str
    ) -> float:
        """Compute title similarity.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Title similarity score (0.0 to 1.0)
        """
        # Exact match
        if title1.lower().strip() == title2.lower().strip():
            return 1.0
        
        # MinHash similarity
        minhash1 = self.minhash_gen.generate_minhash(title1)
        minhash2 = self.minhash_gen.generate_minhash(title2)
        
        jaccard_sim = self.minhash_gen.jaccard_similarity(minhash1, minhash2)
        
        # Levenshtein distance for additional similarity
        levenshtein_sim = self._compute_levenshtein_similarity(title1, title2)
        
        # Weighted combination
        return 0.7 * jaccard_sim + 0.3 * levenshtein_sim
    
    async def compute_content_similarity(
        self,
        content1: str,
        content2: str
    ) -> float:
        """Compute content similarity.
        
        Args:
            content1: First content
            content2: Second content
            
        Returns:
            Content similarity score (0.0 to 1.0)
        """
        # MinHash similarity
        minhash1 = self.minhash_gen.generate_minhash(content1)
        minhash2 = self.minhash_gen.generate_minhash(content2)
        
        jaccard_sim = self.minhash_gen.jaccard_similarity(minhash1, minhash2)
        
        # Length similarity
        length_sim = self._compute_length_similarity(len(content1), len(content2))
        
        # Word overlap similarity
        word_sim = self._compute_word_overlap_similarity(content1, content2)
        
        # Weighted combination
        return 0.6 * jaccard_sim + 0.2 * length_sim + 0.2 * word_sim
    
    def _compute_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Compute Levenshtein distance similarity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein similarity (0.0 to 1.0)
        """
        if len(s1) < len(s2):
            return self._compute_levenshtein_similarity(s2, s1)
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        distance = previous_row[-1]
        return 1.0 - (distance / max_len)
    
    def _compute_length_similarity(self, len1: int, len2: int) -> float:
        """Compute length similarity.
        
        Args:
            len1: First length
            len2: Second length
            
        Returns:
            Length similarity (0.0 to 1.0)
        """
        if len1 == 0 and len2 == 0:
            return 1.0
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        return min(len1, len2) / max(len1, len2)
    
    def _compute_word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Compute word overlap similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Word overlap similarity (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union

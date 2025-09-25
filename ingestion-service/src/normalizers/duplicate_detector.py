"""
Duplicate detection and content deduplication.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models.content import DuplicateDetection, NormalizedArticle
from src.normalizers.content_normalizer import ContentNormalizer

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Duplicate detection and content deduplication utilities."""

    def __init__(self, content_normalizer: ContentNormalizer = None):
        self.content_normalizer = content_normalizer or ContentNormalizer()
        self.duplicate_threshold = 0.8
        self.check_period_days = 7

    async def detect_duplicates(
        self, article: NormalizedArticle, existing_articles: List[NormalizedArticle]
    ) -> List[DuplicateDetection]:
        """Detect duplicates for a given article."""
        duplicates = []

        # Filter articles within the check period
        cutoff_date = datetime.utcnow() - timedelta(days=self.check_period_days)
        recent_articles = [a for a in existing_articles if a.published_at and a.published_at >= cutoff_date]

        # Check for duplicates
        for existing_article in recent_articles:
            similarity = await self._calculate_similarity(article, existing_article)

            if similarity >= self.duplicate_threshold:
                duplicate = DuplicateDetection(
                    article_id=article.id,
                    duplicate_of=existing_article.id,
                    similarity_score=similarity,
                    detection_method="content_similarity",
                    detected_at=datetime.utcnow(),
                )
                duplicates.append(duplicate)

        return duplicates

    async def _calculate_similarity(self, article1: NormalizedArticle, article2: NormalizedArticle) -> float:
        """Calculate similarity between two articles."""
        # Check content hash first (exact match)
        if article1.content_hash == article2.content_hash:
            return 1.0

        # Calculate weighted similarity
        similarities = []

        # Title similarity (40% weight)
        title_similarity = self._text_similarity(article1.title, article2.title)
        similarities.append((title_similarity, 0.4))

        # Content similarity (40% weight)
        content_similarity = self._text_similarity(article1.content or "", article2.content or "")
        similarities.append((content_similarity, 0.4))

        # URL similarity (20% weight)
        url_similarity = self._url_similarity(article1.url, article2.url)
        similarities.append((url_similarity, 0.2))

        # Calculate weighted average
        weighted_similarity = sum(sim * weight for sim, weight in similarities)

        return weighted_similarity

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using multiple methods."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)

        if text1 == text2:
            return 1.0

        # Word-based similarity
        word_similarity = self._word_similarity(text1, text2)

        # Character-based similarity
        char_similarity = self._character_similarity(text1, text2)

        # N-gram similarity
        ngram_similarity = self._ngram_similarity(text1, text2)

        # Combine similarities
        combined_similarity = word_similarity * 0.5 + char_similarity * 0.3 + ngram_similarity * 0.2

        return combined_similarity

    def _word_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-based similarity using Jaccard index."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-based similarity using Levenshtein distance."""
        if not text1 or not text2:
            return 0.0

        # Simple character overlap
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())

        if not chars1 or not chars2:
            return 0.0

        intersection = chars1.intersection(chars2)
        union = chars1.union(chars2)

        return len(intersection) / len(union) if union else 0.0

    def _ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate n-gram similarity."""
        if not text1 or not text2:
            return 0.0

        ngrams1 = set(self._get_ngrams(text1.lower(), n))
        ngrams2 = set(self._get_ngrams(text2.lower(), n))

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        return len(intersection) / len(union) if union else 0.0

    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Generate n-grams from text."""
        if len(text) < n:
            return [text]

        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i : i + n])

        return ngrams

    def _url_similarity(self, url1: str, url2: str) -> float:
        """Calculate URL similarity."""
        if not url1 or not url2:
            return 0.0

        if url1 == url2:
            return 1.0

        # Check if URLs are from the same domain
        from src.utils.url_utils import url_utils

        domain1 = url_utils.extract_domain(url1)
        domain2 = url_utils.extract_domain(url2)

        if domain1 == domain2:
            return 0.8

        # Check if URLs are variations of each other
        if self._are_url_variations(url1, url2):
            return 0.6

        return 0.0

    def _are_url_variations(self, url1: str, url2: str) -> bool:
        """Check if URLs are variations of each other."""
        from src.utils.url_utils import url_utils

        # Normalize URLs
        clean_url1 = url_utils.normalize_url(url1)
        clean_url2 = url_utils.normalize_url(url2)

        # Check if they're the same after cleaning
        if clean_url1 == clean_url2:
            return True

        # Check if one is a subdirectory of the other
        if clean_url1.startswith(clean_url2) or clean_url2.startswith(clean_url1):
            return True

        return False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        import re

        text = re.sub(r"\s+", " ", text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        return text.strip()

    async def find_exact_duplicates(self, articles: List[NormalizedArticle]) -> List[List[NormalizedArticle]]:
        """Find exact duplicates based on content hash."""
        hash_groups = {}

        for article in articles:
            if article.content_hash in hash_groups:
                hash_groups[article.content_hash].append(article)
            else:
                hash_groups[article.content_hash] = [article]

        # Return groups with more than one article
        duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]

        return duplicate_groups

    async def find_similar_articles(
        self, articles: List[NormalizedArticle], threshold: float = None
    ) -> List[List[NormalizedArticle]]:
        """Find similar articles using content similarity."""
        if threshold is None:
            threshold = self.duplicate_threshold

        similar_groups = []
        processed = set()

        for i, article1 in enumerate(articles):
            if article1.id in processed:
                continue

            similar_group = [article1]
            processed.add(article1.id)

            for j, article2 in enumerate(articles[i + 1 :], i + 1):
                if article2.id in processed:
                    continue

                similarity = await self._calculate_similarity(article1, article2)

                if similarity >= threshold:
                    similar_group.append(article2)
                    processed.add(article2.id)

            if len(similar_group) > 1:
                similar_groups.append(similar_group)

        return similar_groups

    async def deduplicate_articles(self, articles: List[NormalizedArticle]) -> List[NormalizedArticle]:
        """Remove duplicates from a list of articles, keeping the best one."""
        if not articles:
            return []

        # Find similar groups
        similar_groups = await self.find_similar_articles(articles)

        # Keep the best article from each group
        kept_articles = []
        processed_ids = set()

        for group in similar_groups:
            # Sort by quality score (word count, recency, etc.)
            best_article = self._select_best_article(group)
            kept_articles.append(best_article)
            processed_ids.add(best_article.id)

        # Add articles that weren't in any group
        for article in articles:
            if article.id not in processed_ids:
                kept_articles.append(article)

        return kept_articles

    def _select_best_article(self, articles: List[NormalizedArticle]) -> NormalizedArticle:
        """Select the best article from a group of similar articles."""
        if len(articles) == 1:
            return articles[0]

        # Score articles based on multiple criteria
        best_article = articles[0]
        best_score = self._calculate_article_score(best_article)

        for article in articles[1:]:
            score = self._calculate_article_score(article)
            if score > best_score:
                best_article = article
                best_score = score

        return best_article

    def _calculate_article_score(self, article: NormalizedArticle) -> float:
        """Calculate quality score for an article."""
        score = 0.0

        # Word count score (0-1)
        if article.word_count > 0:
            word_score = min(article.word_count / 1000, 1.0)  # Normalize to 0-1
            score += word_score * 0.3

        # Title length score (0-1)
        if article.title:
            title_score = min(len(article.title) / 100, 1.0)  # Normalize to 0-1
            score += title_score * 0.2

        # Recency score (0-1)
        if article.published_at:
            days_old = (datetime.utcnow() - article.published_at).days
            recency_score = max(0, 1.0 - (days_old / 30))  # Decay over 30 days
            score += recency_score * 0.2

        # Content completeness score (0-1)
        completeness_score = 0.0
        if article.content:
            completeness_score += 0.5
        if article.summary:
            completeness_score += 0.3
        if article.author:
            completeness_score += 0.2
        score += completeness_score * 0.3

        return score

    async def get_duplicate_statistics(self, articles: List[NormalizedArticle]) -> Dict[str, Any]:
        """Get duplicate detection statistics."""
        total_articles = len(articles)

        # Find exact duplicates
        exact_duplicate_groups = await self.find_exact_duplicates(articles)
        exact_duplicates = sum(len(group) - 1 for group in exact_duplicate_groups)

        # Find similar articles
        similar_groups = await self.find_similar_articles(articles)
        similar_duplicates = sum(len(group) - 1 for group in similar_groups)

        # Calculate rates
        exact_duplicate_rate = exact_duplicates / total_articles if total_articles > 0 else 0
        similar_duplicate_rate = similar_duplicates / total_articles if total_articles > 0 else 0

        return {
            "total_articles": total_articles,
            "exact_duplicates": exact_duplicates,
            "similar_duplicates": similar_duplicates,
            "exact_duplicate_rate": exact_duplicate_rate,
            "similar_duplicate_rate": similar_duplicate_rate,
            "exact_duplicate_groups": len(exact_duplicate_groups),
            "similar_duplicate_groups": len(similar_groups),
        }

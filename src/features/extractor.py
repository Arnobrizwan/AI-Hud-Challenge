"""Comprehensive feature extraction for ranking."""

import math
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import structlog
from textstat import flesch_reading_ease

from ..optimization.cache import CacheManager
from ..schemas import Article, AuthorityScore, RankingRequest, TrendingScore

logger = structlog.get_logger(__name__)


class RankingFeatureExtractor:
    """Comprehensive feature extraction for ranking."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.trending_detector = TrendingDetector(cache_manager)
        self.authority_scorer = AuthorityScorer(cache_manager)
        self.geo_calculator = GeographicCalculator()

        # Feature cache keys
        self.feature_cache_prefix = "features"
        self.cache_ttl = 3600  # 1 hour

    async def compute_content_features(self, article: Article) -> Dict[str, float]:
        """Content quality and characteristics features."""
        try:
            features = {}

            # Basic content metrics
            features["title_length"] = len(article.title)
            features["content_length"] = len(article.content) if article.content else 0
            features["word_count"] = article.word_count
            features["reading_time"] = article.reading_time
            features["quality_score"] = article.quality_score

            # Readability features
            if article.content:
                features["readability_score"] = await self._compute_readability(article.content)
                features["avg_sentence_length"] = await self._compute_avg_sentence_length(article.content)
                features["paragraph_count"] = len(article.content.split("\n\n"))
            else:
                features["readability_score"] = 0.5
                features["avg_sentence_length"] = 0
                features["paragraph_count"] = 0

            # Sentiment features
            if article.sentiment:
                features["sentiment_polarity"] = article.sentiment.polarity
                features["sentiment_subjectivity"] = article.sentiment.subjectivity
            else:
                features["sentiment_polarity"] = 0.0
                features["sentiment_subjectivity"] = 0.0

            # Entity and topic features
            features["entity_count"] = len(article.entities)
            features["topic_count"] = len(article.topics)
            features["unique_topics"] = len(set(t.name for t in article.topics))

            # Media features
            features["has_image"] = 1.0 if article.image_url else 0.0
            features["has_video"] = 1.0 if article.videos else 0.0
            features["video_count"] = len(article.videos) if article.videos else 0

            # Language features
            features["language"] = 1.0 if article.language == "en" else 0.0
            features["title_caps_ratio"] = await self._compute_caps_ratio(article.title)
            features["title_question_mark"] = 1.0 if "?" in article.title else 0.0
            features["title_exclamation"] = 1.0 if "!" in article.title else 0.0

            # Content structure features
            features["has_summary"] = 1.0 if article.summary else 0.0
            features["summary_length"] = len(article.summary) if article.summary else 0

            return features

        except Exception as e:
            logger.error("Content feature extraction failed", error=str(e), article_id=article.id)
            return self._get_default_content_features()

    async def compute_freshness_features(self, article: Article) -> Dict[str, float]:
        """Time-based freshness and relevance features."""
        try:
            now = datetime.utcnow()
            age_hours = (now - article.published_at).total_seconds() / 3600
            age_days = age_hours / 24

            features = {}

            # Age features
            features["age_hours"] = age_hours
            features["age_days"] = age_days
            features["age_log"] = math.log(age_hours + 1)
            features["age_sqrt"] = math.sqrt(age_hours)

            # Freshness scores with different decay rates
            # 24-hour half-life
            features["freshness_24h"] = math.exp(-age_hours / 24)
            # 6-hour half-life
            features["freshness_6h"] = math.exp(-age_hours / 6)
            # 1-hour half-life
            features["freshness_1h"] = math.exp(-age_hours / 1)

            # Time-based categories
            features["is_breaking"] = 1.0 if age_hours < 1 else 0.0
            features["is_recent"] = 1.0 if age_hours < 6 else 0.0
            features["is_today"] = 1.0 if age_hours < 24 else 0.0
            features["is_this_week"] = 1.0 if age_days < 7 else 0.0
            features["is_this_month"] = 1.0 if age_days < 30 else 0.0

            # Temporal features
            features["hour_of_day"] = article.published_at.hour
            features["day_of_week"] = article.published_at.weekday()
            features["is_weekend"] = 1.0 if article.published_at.weekday() >= 5 else 0.0
            features["is_business_hours"] = 1.0 if 9 <= article.published_at.hour <= 17 else 0.0

            # Time-based engagement boost
            features["time_engagement_boost"] = await self._compute_time_engagement_boost(article)

            return features

        except Exception as e:
            logger.error("Freshness feature extraction failed", error=str(e), article_id=article.id)
            return self._get_default_freshness_features()

    async def compute_authority_features(self, article: Article) -> Dict[str, float]:
        """Source authority and credibility features."""
        try:
            # Get authority scores from cache or compute
            authority_scores = await self.authority_scorer.get_authority_scores(article.source.id)

            features = {}

            # Source authority features
            features["source_authority"] = authority_scores.authority_score
            features["source_reliability"] = authority_scores.reliability_score
            features["source_popularity"] = authority_scores.popularity_score
            features["source_recency"] = authority_scores.recency_score

            # Author authority features
            if article.author:
                author_authority = await self._get_author_authority(article.author.id)
                features["author_authority"] = author_authority
                features["has_author"] = 1.0
            else:
                features["author_authority"] = 0.0
                features["has_author"] = 0.0

            # Content credibility features
            features["citation_count"] = await self._get_citation_count(article.url)
            features["social_shares"] = await self._get_social_shares(article.url)
            features["backlink_count"] = await self._get_backlink_count(article.url)

            # Domain features
            domain = article.url.split("/")[2] if "://" in article.url else ""
            features["domain_length"] = len(domain)
            features["is_https"] = 1.0 if article.url.startswith("https://") else 0.0
            features["subdomain_count"] = len(domain.split(".")) - 2 if "." in domain else 0

            return features

        except Exception as e:
            logger.error("Authority feature extraction failed", error=str(e), article_id=article.id)
            return self._get_default_authority_features()

    async def compute_personalization_features(self, article: Article, user_id: str) -> Dict[str, float]:
        """Personalization features based on user preferences."""
        try:
            # Get user profile from cache
            user_profile = await self.cache_manager.get(f"user_profile:{user_id}")
            if not user_profile:
                return self._get_default_personalization_features()

            features = {}

            # Topic affinity features
            topic_scores = []
            for topic in article.topics:
                topic_name = topic.name.lower()
                affinity = user_profile.get("topic_preferences", {}).get(topic_name, 0.5)
                topic_scores.append(affinity * topic.confidence)

            features["avg_topic_affinity"] = np.mean(topic_scores) if topic_scores else 0.5
            features["max_topic_affinity"] = np.max(topic_scores) if topic_scores else 0.5
            features["topic_affinity_variance"] = np.var(topic_scores) if topic_scores else 0.0

            # Source preference features
            source_id = article.source.id
            source_preference = user_profile.get("source_preferences", {}).get(source_id, 0.5)
            features["source_preference"] = source_preference

            # Content preference features
            content_prefs = user_profile.get("content_preferences", {})
            preferred_length = content_prefs.get("preferred_length", 0)
            if preferred_length > 0:
                length_similarity = 1.0 - abs(preferred_length - article.word_count) / max(
                    preferred_length, article.word_count
                )
                features["length_preference"] = length_similarity
            else:
                features["length_preference"] = 0.5

            # Time preference features
            reading_patterns = user_profile.get("reading_patterns", {})
            preferred_hours = reading_patterns.get("preferred_hours", [])
            if preferred_hours:
                current_hour = article.published_at.hour
                hour_match = 1.0 if current_hour in preferred_hours else 0.0
                features["time_preference"] = hour_match
            else:
                features["time_preference"] = 0.5

            # Historical interaction features
            features["user_engagement_score"] = await self._get_user_engagement_score(user_id, article.id)

            return features

        except Exception as e:
            logger.error("Personalization feature extraction failed", error=str(e), user_id=user_id)
            return self._get_default_personalization_features()

    async def compute_contextual_features(self, article: Article, request: RankingRequest) -> Dict[str, float]:
        """Contextual features based on request context."""
        try:
            features = {}

            # Query relevance features
            if request.query:
                query_lower = request.query.lower()
                title_lower = article.title.lower()
                content_lower = article.content.lower() if article.content else ""

                # Text matching features
                features["title_query_match"] = 1.0 if query_lower in title_lower else 0.0
                features["content_query_match"] = 1.0 if query_lower in content_lower else 0.0
                features["query_word_count"] = len(query_lower.split())
                features["title_query_similarity"] = await self._compute_text_similarity(query_lower, title_lower)
            else:
                features["title_query_match"] = 0.0
                features["content_query_match"] = 0.0
                features["query_word_count"] = 0
                features["title_query_similarity"] = 0.0

            # Geographic features
            if request.location and article.country:
                geo_distance = await self.geo_calculator.compute_distance(request.location, article.country)
                features["geo_distance"] = geo_distance
                features["geo_relevance"] = 1.0 / (1.0 + geo_distance) if geo_distance > 0 else 1.0
            else:
                features["geo_distance"] = 0.0
                features["geo_relevance"] = 0.5

            # Content type features
            features["requested_content_type"] = 1.0 if article.content_type in request.content_types else 0.0

            # Topic filtering features
            if request.topics:
                article_topic_names = [t.name.lower() for t in article.topics]
                requested_topics = [t.lower() for t in request.topics]
                topic_match = len(set(article_topic_names).intersection(set(requested_topics)))
                features["topic_filter_match"] = topic_match / len(requested_topics) if requested_topics else 0.0
            else:
                features["topic_filter_match"] = 0.0

            # Source filtering features
            if request.sources:
                source_match = 1.0 if article.source.id in request.sources else 0.0
                features["source_filter_match"] = source_match
            else:
                features["source_filter_match"] = 0.0

            # Date range features
            if request.date_range:
                start_date = request.date_range.get("start")
                end_date = request.date_range.get("end")
                if start_date and end_date:
                    in_range = start_date <= article.published_at <= end_date
                    features["date_range_match"] = 1.0 if in_range else 0.0
                else:
                    features["date_range_match"] = 0.0
            else:
                features["date_range_match"] = 0.0

            return features

        except Exception as e:
            logger.error("Contextual feature extraction failed", error=str(e), article_id=article.id)
            return self._get_default_contextual_features()

    async def compute_interaction_features(self, article: Article, user_id: str) -> Dict[str, float]:
        """User interaction features."""
        try:
            features = {}

            # Engagement metrics
            features["view_count"] = article.view_count
            features["like_count"] = article.like_count
            features["share_count"] = article.share_count
            features["comment_count"] = article.comment_count

            # Engagement ratios
            total_engagement = article.view_count + article.like_count + article.share_count + article.comment_count
            features["total_engagement"] = total_engagement
            features["engagement_rate"] = total_engagement / max(article.view_count, 1)
            features["like_rate"] = article.like_count / max(article.view_count, 1)
            features["share_rate"] = article.share_count / max(article.view_count, 1)
            features["comment_rate"] = article.comment_count / max(article.view_count, 1)

            # User-specific interaction features
            user_interactions = await self._get_user_interactions(user_id, article.id)
            features["user_has_viewed"] = 1.0 if user_interactions.get("viewed", False) else 0.0
            features["user_has_liked"] = 1.0 if user_interactions.get("liked", False) else 0.0
            features["user_has_shared"] = 1.0 if user_interactions.get("shared", False) else 0.0
            features["user_has_commented"] = 1.0 if user_interactions.get("commented", False) else 0.0

            # Trending features
            trending_score = await self.trending_detector.compute_trending_score(article)
            features["trending_score"] = trending_score.trending_score
            features["trending_velocity"] = trending_score.velocity
            features["trending_acceleration"] = trending_score.acceleration

            return features

        except Exception as e:
            logger.error("Interaction feature extraction failed", error=str(e), article_id=article.id)
            return self._get_default_interaction_features()

    # Helper methods

    async def _compute_readability(self, content: str) -> float:
        """Compute readability score using Flesch Reading Ease."""
        try:
            score = flesch_reading_ease(content)
            # Normalize to 0-1 range
            return max(0, min(1, (score - 0) / 100))
        except BaseException:
            return 0.5

    async def _compute_avg_sentence_length(self, content: str) -> float:
        """Compute average sentence length."""
        sentences = re.split(r"[.!?]+", content)
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    async def _compute_caps_ratio(self, text: str) -> float:
        """Compute ratio of capital letters."""
        if not text:
            return 0
        return sum(1 for c in text if c.isupper()) / len(text)

    async def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity using Jaccard index."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union

    async def _compute_time_engagement_boost(self, article: Article) -> float:
        """Compute time-based engagement boost."""
        # This would analyze historical engagement patterns by time
        return 1.0  # Default neutral boost

    async def _get_author_authority(self, author_id: str) -> float:
        """Get author authority score."""
        # This would query author database
        return 0.5  # Default score

    async def _get_citation_count(self, url: str) -> int:
        """Get citation count for article."""
        # This would query citation database
        return 0

    async def _get_social_shares(self, url: str) -> int:
        """Get social media share count."""
        # This would query social media APIs
        return 0

    async def _get_backlink_count(self, url: str) -> int:
        """Get backlink count."""
        # This would query backlink database
        return 0

    async def _get_user_engagement_score(self, user_id: str, article_id: str) -> float:
        """Get user's historical engagement score."""
        # This would query user interaction database
        return 0.5

    async def _get_user_interactions(self, user_id: str, article_id: str) -> Dict[str, bool]:
        """Get user's interactions with article."""
        # This would query user interaction database
        return {"viewed": False, "liked": False, "shared": False, "commented": False}

    # Default feature methods

    def _get_default_content_features(self) -> Dict[str, float]:
        """Default content features when extraction fails."""
        return {
            "title_length": 0,
            "content_length": 0,
            "word_count": 0,
            "reading_time": 0,
            "quality_score": 0.5,
            "readability_score": 0.5,
            "avg_sentence_length": 0,
            "paragraph_count": 0,
            "sentiment_polarity": 0.0,
            "sentiment_subjectivity": 0.0,
            "entity_count": 0,
            "topic_count": 0,
            "unique_topics": 0,
            "has_image": 0.0,
            "has_video": 0.0,
            "video_count": 0,
            "language": 1.0,
            "title_caps_ratio": 0.0,
            "title_question_mark": 0.0,
            "title_exclamation": 0.0,
            "has_summary": 0.0,
            "summary_length": 0,
        }

    def _get_default_freshness_features(self) -> Dict[str, float]:
        """Default freshness features when extraction fails."""
        return {
            "age_hours": 24,
            "age_days": 1,
            "age_log": 3.2,
            "age_sqrt": 4.9,
            "freshness_24h": 0.37,
            "freshness_6h": 0.02,
            "freshness_1h": 0.0,
            "is_breaking": 0.0,
            "is_recent": 0.0,
            "is_today": 1.0,
            "is_this_week": 1.0,
            "is_this_month": 1.0,
            "hour_of_day": 12,
            "day_of_week": 0,
            "is_weekend": 0.0,
            "is_business_hours": 1.0,
            "time_engagement_boost": 1.0,
        }

    def _get_default_authority_features(self) -> Dict[str, float]:
        """Default authority features when extraction fails."""
        return {
            "source_authority": 0.5,
            "source_reliability": 0.5,
            "source_popularity": 0.5,
            "source_recency": 0.5,
            "author_authority": 0.5,
            "has_author": 0.0,
            "citation_count": 0,
            "social_shares": 0,
            "backlink_count": 0,
            "domain_length": 10,
            "is_https": 1.0,
            "subdomain_count": 0,
        }

    def _get_default_personalization_features(self) -> Dict[str, float]:
        """Default personalization features when extraction fails."""
        return {
            "avg_topic_affinity": 0.5,
            "max_topic_affinity": 0.5,
            "topic_affinity_variance": 0.0,
            "source_preference": 0.5,
            "length_preference": 0.5,
            "time_preference": 0.5,
            "user_engagement_score": 0.5,
        }

    def _get_default_contextual_features(self) -> Dict[str, float]:
        """Default contextual features when extraction fails."""
        return {
            "title_query_match": 0.0,
            "content_query_match": 0.0,
            "query_word_count": 0,
            "title_query_similarity": 0.0,
            "geo_distance": 0.0,
            "geo_relevance": 0.5,
            "requested_content_type": 1.0,
            "topic_filter_match": 0.0,
            "source_filter_match": 0.0,
            "date_range_match": 0.0,
        }

    def _get_default_interaction_features(self) -> Dict[str, float]:
        """Default interaction features when extraction fails."""
        return {
            "view_count": 0,
            "like_count": 0,
            "share_count": 0,
            "comment_count": 0,
            "total_engagement": 0,
            "engagement_rate": 0,
            "like_rate": 0,
            "share_rate": 0,
            "comment_rate": 0,
            "user_has_viewed": 0.0,
            "user_has_liked": 0.0,
            "user_has_shared": 0.0,
            "user_has_commented": 0.0,
            "trending_score": 0.0,
            "trending_velocity": 0.0,
            "trending_acceleration": 0.0,
        }


class TrendingDetector:
    """Real-time trending topic and content detection."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.time_windows = [1, 6, 24, 168]  # hours
        self.velocity_threshold = 2.0

    async def compute_trending_score(self, article: Article) -> TrendingScore:
        """Compute trending score based on velocity and acceleration."""
        try:
            scores = []

            for window_hours in self.time_windows:
                # Get article count in time window
                count = await self._get_article_count_in_window(article.topics, window_hours)

                # Get baseline count for same period previous week
                baseline = await self._get_baseline_count(article.topics, window_hours)

                # Compute velocity (relative increase)
                velocity = count / (baseline + 1)
                scores.append(velocity)

            # Combine velocities with decay weights
            weights = [0.4, 0.3, 0.2, 0.1]  # Favor recent windows
            trending_score = sum(s * w for s, w in zip(scores, weights))

            # Compute acceleration (change in velocity)
            acceleration = scores[0] - scores[1] if len(scores) > 1 else 0

            return TrendingScore(
                article_id=article.id,
                trending_score=min(trending_score, 5.0),  # Cap at 5x baseline
                velocity=scores[0] if scores else 0,
                acceleration=acceleration,
                time_window_hours=self.time_windows[0],
                baseline_comparison=scores[0] if scores else 0,
            )

        except Exception as e:
            logger.error("Trending score computation failed", error=str(e), article_id=article.id)
            return TrendingScore(
                article_id=article.id,
                trending_score=0.0,
                velocity=0.0,
                acceleration=0.0,
                time_window_hours=1,
                baseline_comparison=0.0,
            )

    async def _get_article_count_in_window(self, topics: List, window_hours: int) -> int:
        """Get article count in time window."""
        # This would query article database
        return 0

    async def _get_baseline_count(self, topics: List, window_hours: int) -> int:
        """Get baseline count for comparison."""
        # This would query historical data
        return 1


class AuthorityScorer:
    """Source authority and credibility scoring."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager

    async def get_authority_scores(self, source_id: str) -> AuthorityScore:
        """Get authority scores for source."""
        try:
            # Check cache first
            cached = await self.cache_manager.get(f"authority:{source_id}")
            if cached:
                return AuthorityScore(**cached)

            # Compute authority scores
            scores = await self._compute_authority_scores(source_id)

            # Cache the results
            await self.cache_manager.set(f"authority:{source_id}", scores.dict(), ttl=3600)

            return scores

        except Exception as e:
            logger.error("Authority scoring failed", error=str(e), source_id=source_id)
            return AuthorityScore(
                source_id=source_id,
                authority_score=0.5,
                reliability_score=0.5,
                popularity_score=0.5,
                recency_score=0.5,
                computed_at=datetime.utcnow(),
            )

    async def _compute_authority_scores(self, source_id: str) -> AuthorityScore:
        """Compute authority scores for source."""
        # This would implement actual authority scoring logic
        return AuthorityScore(
            source_id=source_id,
            authority_score=0.5,
            reliability_score=0.5,
            popularity_score=0.5,
            recency_score=0.5,
            computed_at=datetime.utcnow(),
        )


class GeographicCalculator:
    """Geographic distance and relevance calculations."""

    def __init__(self):
        self.earth_radius_km = 6371

    async def compute_distance(self, location1: Dict[str, float], location2: Dict[str, float]) -> float:
        """Compute distance between two locations in kilometers."""
        try:
            lat1, lon1 = location1.get("lat", 0), location1.get("lng", 0)
            lat2, lon2 = location2.get("lat", 0), location2.get("lng", 0)

            # Haversine formula
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
                math.radians(lat2)
            ) * math.sin(dlon / 2) * math.sin(dlon / 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return self.earth_radius_km * c

        except Exception as e:
            logger.error("Distance calculation failed", error=str(e))
            return 0.0

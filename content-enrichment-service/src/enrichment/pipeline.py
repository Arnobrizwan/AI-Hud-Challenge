"""Core content enrichment pipeline with parallel processing."""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from entities.extractor import EntityExtractor
from knowledge_base.entity_kb import EntityKnowledgeBase
from models.content import EnrichedContent, ExtractedContent, ModelVersion, ProcessingMode
from sentiment.analyzer import SentimentAnalyzer
from signals.extractor import SignalExtractor
from topics.classifier import TopicClassifier
from utils.language_detector import LanguageDetector

logger = structlog.get_logger(__name__)


class ContentEnrichmentPipeline:
    """Comprehensive content enrichment with AI models."""

    def __init__(self):
        """Initialize the enrichment pipeline."""
        self.entity_extractor = EntityExtractor()
        self.topic_classifier = TopicClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.signal_extractor = SignalExtractor()
        self.entity_kb = EntityKnowledgeBase()
        self.language_detector = LanguageDetector()

        # Model version tracking
        self.model_versions = self._get_model_versions()

    def _get_model_versions(self) -> Dict[str, ModelVersion]:
        """Get current model versions."""
        return {
            "entity_extractor": ModelVersion(
                name="spacy_ner",
                version="3.7.2",
                created_at=datetime.utcnow(),
                performance_metrics={"accuracy": 0.89, "f1": 0.87},
            ),
            "topic_classifier": ModelVersion(
                name="hierarchical_classifier",
                version="1.0.0",
                created_at=datetime.utcnow(),
                performance_metrics={"accuracy": 0.92, "f1": 0.90},
            ),
            "sentiment_analyzer": ModelVersion(
                name="multilingual_sentiment",
                version="2.1.0",
                created_at=datetime.utcnow(),
                performance_metrics={"accuracy": 0.88, "f1": 0.86},
            ),
            "signal_extractor": ModelVersion(
                name="content_signals",
                version="1.2.0",
                created_at=datetime.utcnow(),
                performance_metrics={"accuracy": 0.85, "f1": 0.83},
            ),
        }

    async def enrich_content(
        self,
        content: ExtractedContent,
        processing_mode: ProcessingMode = ProcessingMode.REALTIME,
        include_entities: bool = True,
        include_topics: bool = True,
        include_sentiment: bool = True,
        include_signals: bool = True,
        include_trust_score: bool = True,
        language_hint: Optional[str] = None,
    ) -> EnrichedContent:
        """Main enrichment pipeline with parallel processing."""
        start_time = time.time()

        try:
            # Detect language if not provided
            detected_language = language_hint or await self.language_detector.detect(content.content)

            # Prepare parallel tasks
            tasks = []
            task_names = []

            if include_entities:
                tasks.append(self._extract_entities(content, detected_language))
                task_names.append("entities")

            if include_topics:
                tasks.append(self._classify_topics(content, detected_language))
                task_names.append("topics")

            if include_sentiment:
                tasks.append(self._analyze_sentiment(content, detected_language))
                task_names.append("sentiment")

            if include_signals:
                tasks.append(self._extract_signals(content, detected_language))
                task_names.append("signals")

            if include_trust_score:
                tasks.append(self._compute_trustworthiness(content, detected_language))
                task_names.append("trust_score")

            # Execute tasks in parallel
            logger.info("Starting parallel enrichment tasks", content_id=content.id, task_count=len(tasks))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            entities, topics, sentiment, signals, trust_score = self._process_results(results, task_names)

            processing_time = int((time.time() - start_time) * 1000)

            # Create enriched content
            enriched_content = EnrichedContent(
                id=content.id,
                original_content=content,
                entities=entities or [],
                topics=topics or [],
                sentiment=sentiment,
                signals=signals,
                trust_score=trust_score,
                enrichment_timestamp=datetime.utcnow(),
                model_versions=self.model_versions,
                processing_time_ms=processing_time,
                language_detected=detected_language,
                processing_mode=processing_mode,
            )

            logger.info(
                "Content enrichment completed",
                content_id=content.id,
                processing_time_ms=processing_time,
                entities_count=len(entities) if entities else 0,
                topics_count=len(topics) if topics else 0,
            )

            return enriched_content

        except Exception as e:
            logger.error("Content enrichment failed", content_id=content.id, error=str(e), exc_info=True)
            raise

    async def _extract_entities(self, content: ExtractedContent, language: str) -> List[Any]:
        """Extract entities from content."""
        try:
            return await self.entity_extractor.extract_entities(content.content, language=language)
        except Exception as e:
            logger.error("Entity extraction failed", content_id=content.id, error=str(e))
            return []

    async def _classify_topics(self, content: ExtractedContent, language: str) -> List[Any]:
        """Classify topics for content."""
        try:
            return await self.topic_classifier.classify_topics(content, language=language)
        except Exception as e:
            logger.error("Topic classification failed", content_id=content.id, error=str(e))
            return []

    async def _analyze_sentiment(self, content: ExtractedContent, language: str) -> Any:
        """Analyze sentiment and emotions."""
        try:
            return await self.sentiment_analyzer.analyze_sentiment(content, language=language)
        except Exception as e:
            logger.error("Sentiment analysis failed", content_id=content.id, error=str(e))
            # Return neutral sentiment as fallback
            from ..models.content import SentimentAnalysis, SentimentLabel

            return SentimentAnalysis(sentiment=SentimentLabel.NEUTRAL, confidence=0.5, subjectivity=0.5, polarity=0.0)

    async def _extract_signals(self, content: ExtractedContent, language: str) -> Any:
        """Extract content quality signals."""
        try:
            return await self.signal_extractor.extract_signals(content, language=language)
        except Exception as e:
            logger.error("Signal extraction failed", content_id=content.id, error=str(e))
            # Return default signals as fallback
            from ..models.content import ContentSignal

            return ContentSignal(
                readability_score=0.5,
                factual_claims=0,
                citations_count=0,
                bias_score=0.0,
                engagement_prediction=0.5,
                virality_potential=0.5,
                content_freshness=0.5,
                authority_score=0.5,
            )

    async def _compute_trustworthiness(self, content: ExtractedContent, language: str) -> Any:
        """Compute trustworthiness score."""
        try:
            return await self.signal_extractor.compute_trustworthiness(content, language=language)
        except Exception as e:
            logger.error("Trustworthiness computation failed", content_id=content.id, error=str(e))
            # Return neutral trust score as fallback
            from ..models.content import TrustworthinessScore

            return TrustworthinessScore(
                overall_score=0.5,
                source_reliability=0.5,
                fact_checking_score=0.5,
                citation_quality=0.5,
                author_credibility=0.5,
                content_quality=0.5,
            )

    def _process_results(self, results: List[Any], task_names: List[str]) -> Tuple[List[Any], List[Any], Any, Any, Any]:
        """Process parallel task results."""
        entities = None
        topics = None
        sentiment = None
        signals = None
        trust_score = None

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Task failed", task_name=task_names[i], error=str(result))
                continue

            task_name = task_names[i]
            if task_name == "entities":
                entities = result
            elif task_name == "topics":
                topics = result
            elif task_name == "sentiment":
                sentiment = result
            elif task_name == "signals":
                signals = result
            elif task_name == "trust_score":
                trust_score = result

        return entities, topics, sentiment, signals, trust_score

    async def enrich_batch(
        self,
        contents: List[ExtractedContent],
        processing_mode: ProcessingMode = ProcessingMode.BATCH,
        **kwargs,
    ) -> List[EnrichedContent]:
        """Enrich multiple contents in batch."""
        logger.info("Starting batch enrichment", content_count=len(contents))

        # Process in smaller batches to avoid overwhelming the system
        batch_size = 10
        enriched_contents = []

        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]
            batch_tasks = [self.enrich_content(content, processing_mode, **kwargs) for content in batch]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Batch enrichment item failed", error=str(result))
                else:
                    enriched_contents.append(result)

        logger.info(
            "Batch enrichment completed",
            total_contents=len(contents),
            successful_contents=len(enriched_contents),
        )

        return enriched_contents

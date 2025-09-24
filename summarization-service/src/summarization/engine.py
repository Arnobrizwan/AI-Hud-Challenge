"""
Core Content Summarization Engine
Advanced multi-modal summarization with quality control
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings

from .ab_testing import ABTestManager
from .abstractive import VertexAISummarizer
from .bias import BiasDetector
from .consistency import FactualConsistencyChecker
from .extractive import BertExtractiveSummarizer
from .hybrid import HybridSummarizer
from .models import (
    BiasAnalysis,
    ConsistencyScores,
    ContentType,
    Language,
    ProcessedContent,
    ProcessingStats,
    QualityMetrics,
    SummarizationMethod,
    SummarizationRequest,
    Summary,
    SummaryResult,
)
from .preprocessing import ContentPreprocessor
from .quality import SummaryQualityScorer
from .translation import TranslationService

logger = logging.getLogger(__name__)


class ContentSummarizationEngine:
    """Advanced multi-modal summarization with quality control"""

    def __init__(self):
        """Initialize the summarization engine"""
        self.extractive_summarizer = BertExtractiveSummarizer()
        self.abstractive_summarizer = VertexAISummarizer()
        self.hybrid_summarizer = HybridSummarizer()
        self.preprocessor = ContentPreprocessor()
        self.quality_scorer = SummaryQualityScorer()
        self.bias_detector = BiasDetector()
        self.consistency_checker = FactualConsistencyChecker()
        self.ab_test_manager = ABTestManager()
        self.translation_service = TranslationService()

        self._initialized = False
        self._model_status = {}

    async def warm_up(self) -> Dict[str, Any]:
    """Warm up all models for faster inference"""
        logger.info("Warming up summarization models...")

        try:
            # Initialize models in parallel
            tasks = [
                self.extractive_summarizer.initialize(),
                self.abstractive_summarizer.initialize(),
                self.hybrid_summarizer.initialize(),
                self.preprocessor.initialize(),
                self.quality_scorer.initialize(),
                self.bias_detector.initialize(),
                self.consistency_checker.initialize(),
                self.translation_service.initialize(),
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            self._initialized = True
            self._model_status = {
                "extractive": True,
                "abstractive": True,
                "hybrid": True,
                "preprocessor": True,
                "quality_scorer": True,
                "bias_detector": True,
                "consistency_checker": True,
                "translation": True,
            }

            logger.info("All models warmed up successfully")

        except Exception as e:
            logger.error(f"Failed to warm up models: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Clean up resources"""
        logger.info("Cleaning up summarization engine...")

        try:
            # Cleanup all components
            cleanup_tasks = [
                self.extractive_summarizer.cleanup(),
                self.abstractive_summarizer.cleanup(),
                self.hybrid_summarizer.cleanup(),
                self.preprocessor.cleanup(),
                self.quality_scorer.cleanup(),
                self.bias_detector.cleanup(),
                self.consistency_checker.cleanup(),
                self.translation_service.cleanup(),
            ]

            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            logger.info("Cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def generate_summary(
            self,
            request: SummarizationRequest) -> SummaryResult:
        """Main summarization pipeline with quality validation"""

        if not self._initialized:
            raise RuntimeError("Summarization engine not initialized")

        start_time = time.time()
        processing_stats = ProcessingStats(
            total_time=0,
            preprocessing_time=0,
            summarization_time=0,
            headline_generation_time=0,
            quality_validation_time=0,
            bias_detection_time=0,
            tokens_processed=0,
            models_used=[],
        )

        try:
            # Step 1: Preprocess content
            preprocess_start = time.time()
            processed_content = await self.preprocess_content(request.content)
            processing_stats.preprocessing_time = time.time() - preprocess_start

            # Step 2: Detect language if not specified
            if processed_content.language == Language.ENGLISH:
                detected_language = await self.detect_language(processed_content.text)
                processed_content.language = detected_language

            # Step 3: Translate if needed
            if processed_content.language != Language.ENGLISH:
                processed_content = await self.translate_content(processed_content)

            # Step 4: Generate summary variants
            summarization_start = time.time()
            summaries = await self.generate_summary_variants(
                processed_content, request.target_lengths, request.methods
            )
            processing_stats.summarization_time = time.time() - summarization_start

            # Step 5: Generate headlines
            headline_start = time.time()
            headlines = await self.generate_headlines(processed_content, request.headline_styles)
            processing_stats.headline_generation_time = time.time() - headline_start

            # Step 6: Quality validation
            quality_start = time.time()
            quality_metrics = await self.validate_quality(processed_content, summaries, headlines)
            processing_stats.quality_validation_time = time.time() - quality_start

            # Step 7: Bias detection
            bias_start = time.time()
            bias_analysis = await self.analyze_bias(summaries + headlines)
            processing_stats.bias_detection_time = time.time() - bias_start

            # Step 8: Factual consistency checking
            consistency_scores = await self.check_consistency(
                processed_content, summaries + headlines
            )

            # Step 9: Select best variants
            best_summary = self.select_best_summary(summaries, quality_metrics)
            best_headline = self.select_best_headline(
                headlines, quality_metrics)

            # Step 10: Generate source attribution
            source_attribution = await self.generate_source_attribution(
                processed_content, best_summary
            )

            # Calculate final stats
            processing_stats.total_time = time.time() - start_time
            processing_stats.tokens_processed = len(
                processed_content.text.split())
            processing_stats.models_used = list(self._model_status.keys())

            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(
                quality_metrics, consistency_scores, bias_analysis
            )

            return SummaryResult(
                summary=best_summary,
                headline=best_headline,
                variants=summaries + headlines,
                quality_metrics=quality_metrics,
                consistency_scores=consistency_scores,
                bias_analysis=bias_analysis,
                processing_stats=processing_stats,
                source_attribution=source_attribution,
                language_detected=processed_content.language,
                confidence_score=confidence_score,
            )

        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return await self.generate_fallback_summary(request, processing_stats)

    async def preprocess_content(
            self, content: ProcessedContent) -> ProcessedContent:
        """Preprocess content for summarization"""
        try:
            return await self.preprocessor.preprocess(content)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return content  # Return original if preprocessing fails

    async def detect_language(self, text: str) -> Language:
        """Detect content language"""
        try:
            detected = await self.translation_service.detect_language(text)
            return Language(detected)
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return Language.ENGLISH

    async def translate_content(
            self, content: ProcessedContent) -> ProcessedContent:
        """Translate content to English if needed"""
        try:
            if content.language != Language.ENGLISH:
                translated_text = await self.translation_service.translate(
                    content.text, content.language.value, "en"
                )
                content.text = translated_text
                content.language = Language.ENGLISH
            return content
        except Exception as e:
            logger.warning(f"Translation failed: {str(e)}")
            return content

    async def generate_summary_variants(
        self,
        content: ProcessedContent,
        target_lengths: List[int],
        methods: List[SummarizationMethod],
    ) -> List[Summary]:
        """Generate multiple summary variants using different approaches"""

        variants = []

        for target_length in target_lengths:
            for method in methods:
                try:
                    if method == SummarizationMethod.EXTRACTIVE:
                        summary_text = await self.extractive_summarizer.summarize(
                            content.text, target_length=target_length
                        )
                    elif method == SummarizationMethod.ABSTRACTIVE:
                        summary_text = await self.abstractive_summarizer.summarize(
                            content, target_length=target_length
                        )
                    elif method == SummarizationMethod.HYBRID:
                        summary_text = await self.hybrid_summarizer.summarize(
                            content, target_length=target_length
                        )
                else:
                        continue

                    # Create summary object
                    summary = Summary(
                        text=summary_text,
                        method=method,
                        length=target_length,
                        word_count=len(summary_text.split()),
                    )

                    variants.append(summary)

                except Exception as e:
                    logger.error(
                        f"Failed to generate {method} summary: {str(e)}")
                    continue

        return variants

    async def generate_headlines(
        self, content: ProcessedContent, styles: List[str]
    ) -> List[Any]:  # Will be Headline type from headline_generation module
        """Generate headlines with different styles"""
        try:
            from headline_generation.generator import HeadlineGenerator

            headline_generator = HeadlineGenerator()
            return await headline_generator.generate_headlines(content, styles)
        except Exception as e:
            logger.error(f"Headline generation failed: {str(e)}")
            return []

    async def validate_quality(
            self,
            content: ProcessedContent,
            summaries: List[Summary],
            headlines: List[Any]) -> QualityMetrics:
        """Validate quality of generated content"""
        try:
            # Use the best summary for quality validation
            best_summary = max(summaries, key=lambda s: s.quality_score or 0)

            return await self.quality_scorer.score_quality(content.text, best_summary.text)
        except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            # Return default metrics
            return QualityMetrics(
                rouge1_f1=0.0,
                rouge2_f1=0.0,
                rougeL_f1=0.0,
                bertscore_f1=0.0,
                factual_consistency=0.0,
                readability=0.0,
                coverage=0.0,
                abstractiveness=0.0,
                overall_score=0.0,
            )

    async def analyze_bias(self, content_variants: List[Any]) -> BiasAnalysis:
        """Analyze bias in generated content"""
        try:
            # Combine all text for bias analysis
            all_text = " ".join([getattr(variant, "text", str(variant))
                                 for variant in content_variants])

            return await self.bias_detector.analyze_bias(all_text)
        except Exception as e:
            logger.error(f"Bias analysis failed: {str(e)}")
            return BiasAnalysis(
                political_bias=0.0,
                gender_bias=0.0,
                racial_bias=0.0,
                sentiment_bias=0.0,
                overall_bias=0.0,
                neutrality_score=1.0,
                detected_biases=[],
            )

    async def check_consistency(
        self, content: ProcessedContent, variants: List[Any]
    ) -> ConsistencyScores:
        """Check factual consistency"""
        try:
            # Use the first summary for consistency checking
            summary_text = next(
                (getattr(
                    v,
                    "text",
                    str(v)) for v in variants if hasattr(
                    v,
                    "text")),
                content.text)

            return await self.consistency_checker.check_consistency(content.text, summary_text)
        except Exception as e:
            logger.error(f"Consistency checking failed: {str(e)}")
            return ConsistencyScores(
                entity_consistency=0.0,
                numerical_consistency=0.0,
                temporal_consistency=0.0,
                entailment_score=0.0,
                overall_consistency=0.0,
            )

    def select_best_summary(
        self, summaries: List[Summary], quality_metrics: QualityMetrics
    ) -> Summary:
        """Select the best summary based on quality metrics"""
        if not summaries:
            raise ValueError("No summaries available for selection")

        # Score summaries based on quality metrics
        best_summary = None
        best_score = -1

        for summary in summaries:
            # Calculate composite score
            score = self.calculate_summary_score(summary, quality_metrics)
            if score > best_score:
                best_score = score
                best_summary = summary

        return best_summary or summaries[0]

    def select_best_headline(
            self,
            headlines: List[Any],
            quality_metrics: QualityMetrics) -> Any:
        """Select the best headline based on quality metrics"""
        if not headlines:
            return None

        # Return headline with highest score
        return max(headlines, key=lambda h: getattr(h, "score", 0))

    def calculate_summary_score(
            self,
            summary: Summary,
            quality_metrics: QualityMetrics) -> float:
        """Calculate composite score for summary selection"""
        # Weighted combination of factors
        length_score = 1.0 - abs(summary.word_count -
                                 summary.length) / summary.length
        quality_score = summary.quality_score or quality_metrics.overall_score

        return length_score * 0.3 + quality_score * 0.7

    def calculate_confidence_score(
        self,
        quality_metrics: QualityMetrics,
        consistency_scores: ConsistencyScores,
        bias_analysis: BiasAnalysis,
    ) -> float:
        """Calculate overall confidence score"""
        # Weighted combination of quality indicators
        confidence = (
            quality_metrics.overall_score * 0.4
            + consistency_scores.overall_consistency * 0.3
            + (1.0 - bias_analysis.overall_bias) * 0.2
            + bias_analysis.neutrality_score * 0.1
        )

        return round(confidence, 3)

    async def generate_source_attribution(
        self, content: ProcessedContent, summary: Summary
    ) -> List[str]:
        """Generate source attribution and citations"""
        attribution = []

        if content.source:
            attribution.append(f"Source: {content.source}")
        if content.author:
            attribution.append(f"Author: {content.author}")
        if content.published_at:
            attribution.append(
                f"Published: {content.published_at.strftime('%Y-%m-%d')}")

        attribution.append(f"Summarization method: {summary.method.value}")
        attribution.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return attribution

    async def generate_fallback_summary(
        self, request: SummarizationRequest, processing_stats: ProcessingStats
    ) -> SummaryResult:
        """Generate fallback summary when main pipeline fails"""
        logger.warning("Generating fallback summary due to pipeline failure")

        # Simple extractive fallback
        try:
            text = request.content.text
            sentences = text.split(".")
            fallback_text = ". ".join(sentences[:3]) + "."

            fallback_summary = Summary(
                text=fallback_text,
                method=SummarizationMethod.EXTRACTIVE,
                length=50,
                word_count=len(fallback_text.split()),
            )

            # Create minimal result
            return SummaryResult(
                summary=fallback_summary,
                headline=None,
                variants=[fallback_summary],
                quality_metrics=QualityMetrics(
                    rouge1_f1=0.0,
                    rouge2_f1=0.0,
                    rougeL_f1=0.0,
                    bertscore_f1=0.0,
                    factual_consistency=0.0,
                    readability=0.0,
                    coverage=0.0,
                    abstractiveness=0.0,
                    overall_score=0.5,
                ),
                consistency_scores=ConsistencyScores(
                    entity_consistency=0.0,
                    numerical_consistency=0.0,
                    temporal_consistency=0.0,
                    entailment_score=0.0,
                    overall_consistency=0.5,
                ),
                bias_analysis=BiasAnalysis(
                    political_bias=0.0,
                    gender_bias=0.0,
                    racial_bias=0.0,
                    sentiment_bias=0.0,
                    overall_bias=0.0,
                    neutrality_score=1.0,
                    detected_biases=[],
                ),
                processing_stats=processing_stats,
                source_attribution=[],
                language_detected=request.content.language,
                confidence_score=0.3,
            )

        except Exception as e:
            logger.error(f"Fallback summary generation failed: {str(e)}")
            raise

    async def get_status(self) -> Dict[str, Any]:
    """Get engine status and model information"""
        return {
            "initialized": self._initialized,
            "model_status": self._model_status,
            "settings": {
                "max_content_length": settings.MAX_CONTENT_LENGTH,
                "max_summary_length": settings.MAX_SUMMARY_LENGTH,
                "default_target_lengths": settings.DEFAULT_TARGET_LENGTHS,
                "supported_languages": settings.SUPPORTED_LANGUAGES,
            },
        }

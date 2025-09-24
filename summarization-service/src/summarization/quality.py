"""
Summary Quality Scoring Module
Comprehensive quality assessment and scoring for summaries
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from quality_validation.validator import SummaryQualityValidator

from config.settings import settings

from .models import ProcessedContent, QualityMetrics, Summary

logger = logging.getLogger(__name__)


class SummaryQualityScorer:
    """Advanced quality scoring for summaries"""

    def __init__(self):
        """Initialize the quality scorer"""
        self.validator = SummaryQualityValidator()
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
        """Initialize quality scoring tools"""
        try:
            logger.info("Initializing summary quality scorer...")

            await self.validator.initialize()

            self._initialized = True
            logger.info("Summary quality scorer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize quality scorer: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Clean up resources"""
        try:
    await self.validator.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def score_quality(
            self,
            original_text: str,
            summary_text: str) -> QualityMetrics:
        """Score quality of a summary"""

        if not self._initialized:
            raise RuntimeError("Quality scorer not initialized")

        try:
            return await self.validator.validate_summary_quality(original_text, summary_text)

        except Exception as e:
            logger.error(f"Quality scoring failed: {str(e)}")
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

    async def get_status(self) -> Dict[str, Any]:
        """Get quality scorer status"""
        return {
            "initialized": self._initialized,
            "validator_status":
    await self.validator.get_status(),
        }

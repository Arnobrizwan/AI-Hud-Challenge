"""
Basic tests for the safety service
"""

import asyncio
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abuse_detection.system import AbuseDetectionSystem
from content_moderation.engine import ContentModerationEngine
from drift_detection.detector import MultidimensionalDriftDetector
from safety_engine.core import SafetyMonitoringEngine
from safety_engine.models import SafetyMonitoringRequest, SafetyStatus


class TestSafetyMonitoringEngine:
    """Test the core safety monitoring engine"""

    @pytest_asyncio.fixture
    async def safety_engine(self) -> Dict[str, Any]:
        """Create a safety engine instance for testing"""
        engine = SafetyMonitoringEngine()
        await engine.initialize()
        yield engine
        await engine.cleanup()

    @pytest.fixture
    def sample_request(self):
        """Create a sample monitoring request"""
        return SafetyMonitoringRequest(
            user_id="test_user_123",
            content="This is a test message for safety monitoring",
            features={
                "text_length": 45,
                "sentiment": 0.2,
                "toxicity_score": 0.1},
            metadata={
                "source": "test",
                "timestamp": datetime.utcnow().isoformat()},
        )

    @pytest.mark.asyncio
    async def test_safety_engine_initialization(self, safety_engine) -> Dict[str, Any]:
        """Test that the safety engine initializes correctly"""
        assert safety_engine.is_initialized
        assert safety_engine.drift_detector is not None
        assert safety_engine.abuse_detector is not None
        assert safety_engine.content_moderator is not None

    @pytest.mark.asyncio
    async def test_safety_monitoring_basic(
            self, safety_engine, sample_request) -> Dict[str, Any]:
        """Test basic safety monitoring functionality"""
        # Mock the individual components to return predictable results
        safety_engine.drift_detector.detect_comprehensive_drift = AsyncMock(
            return_value=Mock(overall_severity=0.3, requires_action=False)
        )

        safety_engine.abuse_detector.detect_abuse = AsyncMock(
            return_value=Mock(abuse_score=0.2, threat_level="low")
        )

        safety_engine.content_moderator.moderate_content = AsyncMock(
            return_value=Mock(overall_safety_score=0.8, violations=[])
        )

        # Perform safety monitoring
        result = await safety_engine.monitor_system_safety(sample_request)

        # Verify the result
        assert isinstance(result, SafetyStatus)
        assert 0.0 <= result.overall_score <= 1.0
        assert isinstance(result.requires_intervention, bool)
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_safety_monitoring_with_intervention(
            self, safety_engine, sample_request) -> Dict[str, Any]:
        """Test safety monitoring when intervention is required"""
        # Mock components to return high-risk results
        safety_engine.drift_detector.detect_comprehensive_drift = AsyncMock(
            return_value=Mock(overall_severity=0.9, requires_action=True)
        )

        safety_engine.abuse_detector.detect_abuse = AsyncMock(
            return_value=Mock(abuse_score=0.8, threat_level="high")
        )

        safety_engine.content_moderator.moderate_content = AsyncMock(return_value=Mock(
            overall_safety_score=0.2, violations=["toxicity", "hate_speech"]))

        # Perform safety monitoring
        result = await safety_engine.monitor_system_safety(sample_request)

        # Verify the result indicates intervention is needed
        assert result.requires_intervention
        assert result.overall_score < 1.0  # Safety score should be less than perfect


class TestDriftDetection:
    """Test drift detection functionality"""

    @pytest_asyncio.fixture
    async def drift_detector(self) -> Dict[str, Any]:
        """Create a drift detector instance for testing"""
        detector = MultidimensionalDriftDetector()
        await detector.initialize()
        yield detector
        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_drift_detector_initialization(self, drift_detector) -> Dict[str, Any]:
        """Test that the drift detector initializes correctly"""
        assert drift_detector.is_initialized
        assert drift_detector.statistical_detectors is not None
        assert drift_detector.concept_drift_detector is not None
        assert drift_detector.prediction_drift_detector is not None

    @pytest.mark.asyncio
    async def test_drift_detection_basic(self, drift_detector) -> Dict[str, Any]:
        """Test basic drift detection functionality"""
        # Create mock data
        import numpy as np
        import pandas as pd

        # Reference data (normal distribution)
        reference_data = pd.DataFrame({"feature1": np.random.normal(
            0, 1, 1000), "feature2": np.random.normal(5, 2, 1000)})

        # Current data (slightly shifted distribution)
        current_data = pd.DataFrame({"feature1": np.random.normal(
            0.1, 1, 1000), "feature2": np.random.normal(5.1, 2, 1000)})

        # Create drift request
        from drift_detection.models import DriftDetectionRequest

        drift_request = DriftDetectionRequest(
            reference_data=reference_data,
            current_data=current_data,
            features_to_monitor=["feature1", "feature2"],
        )

        # Detect drift
        result = await drift_detector.detect_comprehensive_drift(drift_request)

        # Verify the result
        assert result is not None
        assert 0.0 <= result.overall_severity <= 1.0
        assert isinstance(result.requires_action, bool)


class TestAbuseDetection:
    """Test abuse detection functionality"""

    @pytest_asyncio.fixture
    async def abuse_detector(self) -> Dict[str, Any]:
        """Create an abuse detector instance for testing"""
        detector = AbuseDetectionSystem()
        await detector.initialize()
        yield detector
        await detector.cleanup()

    @pytest.mark.asyncio
    async def test_abuse_detector_initialization(self, abuse_detector) -> Dict[str, Any]:
        """Test that the abuse detector initializes correctly"""
        assert abuse_detector.is_initialized
        assert abuse_detector.behavioral_analyzer is not None
        assert abuse_detector.graph_analyzer is not None
        assert abuse_detector.ml_classifier is not None

    @pytest.mark.asyncio
    async def test_abuse_detection_basic(self, abuse_detector) -> Dict[str, Any]:
        """Test basic abuse detection functionality"""
        # Create mock activity data
        from abuse_detection.models import AbuseDetectionRequest, ActivityData

        activity_data = ActivityData(
            recent_activities=[
                {"action": "login", "timestamp": datetime.utcnow().isoformat()},
                {"action": "post", "timestamp": datetime.utcnow().isoformat()},
            ],
            user_features={"account_age": 30, "post_count": 5},
            activity_features={"login_frequency": 0.1, "post_frequency": 0.05},
        )

        abuse_request = AbuseDetectionRequest(
            user_id="test_user_123", activity_data=activity_data)

        # Detect abuse
        result = await abuse_detector.detect_abuse(abuse_request)

        # Verify the result
        assert result is not None
        assert result.user_id == "test_user_123"
        assert 0.0 <= result.abuse_score <= 1.0
        assert result.threat_level in ["low", "medium", "high", "critical"]


class TestContentModeration:
    """Test content moderation functionality"""

    @pytest_asyncio.fixture
    async def content_moderator(self) -> Dict[str, Any]:
        """Create a content moderator instance for testing"""
        moderator = ContentModerationEngine()
        await moderator.initialize()
        yield moderator
        await moderator.cleanup()

    @pytest.mark.asyncio
    async def test_content_moderator_initialization(self, content_moderator) -> Dict[str, Any]:
        """Test that the content moderator initializes correctly"""
        assert content_moderator.is_initialized
        assert content_moderator.toxicity_detector is not None
        assert content_moderator.hate_speech_detector is not None
        assert content_moderator.spam_detector is not None

    @pytest.mark.asyncio
    async def test_content_moderation_basic(self, content_moderator) -> Dict[str, Any]:
        """Test basic content moderation functionality"""
        # Create mock content
        from content_moderation.models import ContentItem

        content = ContentItem(
            id="content_123",
            text_content="This is a test message for content moderation",
            user_id="test_user_123",
        )

        # Moderate content
        result = await content_moderator.moderate_content(content)

        # Verify the result
        assert result is not None
        assert result.content_id == "content_123"
        assert 0.0 <= result.overall_safety_score <= 1.0
        assert isinstance(result.violations, list)


class TestIntegration:
    """Integration tests for the safety service"""

    @pytest.mark.asyncio
    async def test_end_to_end_safety_check(self) -> Dict[str, Any]:
        """Test end-to-end safety checking workflow"""
        # Initialize all components
        safety_engine = SafetyMonitoringEngine()
        await safety_engine.initialize()

        try:
            # Create a comprehensive test request
            request = SafetyMonitoringRequest(
                user_id="integration_test_user",
                content="This is an integration test message",
                features={
                    "text_length": 35,
                    "sentiment": 0.1,
                    "toxicity_score": 0.05,
                    "account_age": 100,
                    "post_count": 10,
                },
                metadata={
                    "source": "integration_test",
                    "timestamp": datetime.utcnow().isoformat(),
                    "test_mode": True,
                },
            )

            # Perform comprehensive safety monitoring
            result = await safety_engine.monitor_system_safety(request)

            # Verify the result structure
            assert isinstance(result, SafetyStatus)
            assert hasattr(result, "overall_score")
            assert hasattr(result, "requires_intervention")
            assert hasattr(result, "timestamp")
            assert hasattr(result, "drift_status")
            assert hasattr(result, "abuse_status")
            assert hasattr(result, "content_status")

            # Verify score is within valid range
            assert 0.0 <= result.overall_score <= 1.0

        finally:
            await safety_engine.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling in safety monitoring"""
        safety_engine = SafetyMonitoringEngine()
        await safety_engine.initialize()

        try:
            # Create a request that might cause errors
            request = SafetyMonitoringRequest(
                user_id="error_test_user",
                content="",  # Empty content
                features={},  # Empty features
                metadata={},  # Empty metadata
            )

            # This should not raise an exception
            result = await safety_engine.monitor_system_safety(request)

            # Should return a valid result even with empty inputs
            assert isinstance(result, SafetyStatus)

        finally:
            await safety_engine.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

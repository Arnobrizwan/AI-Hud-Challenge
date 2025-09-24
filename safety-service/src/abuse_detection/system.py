"""
Abuse Detection System
Multi-layered abuse detection and prevention
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from safety_engine.config import get_abuse_config
from safety_engine.models import (
    AbuseDetectionRequest,
    AbuseDetectionResult,
    BehavioralSignals,
    GraphSignals,
    MitigationAction,
    MitigationResult,
    MLPrediction,
    RuleViolation,
)

from .behavioral_analyzer import BehavioralAnomalyDetector
from .captcha_challenger import CaptchaChallenger
from .graph_analyzer import GraphBasedAbuseDetector
from .ml_classifier import AbuseClassificationModel
from .reputation_system import ReputationSystem
from .rule_engine import AbuseRuleEngine

logger = logging.getLogger(__name__)


class AbuseDetectionSystem:
    """Multi-layered abuse detection and prevention"""

    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False

        # Core components
        self.behavioral_analyzer = BehavioralAnomalyDetector()
        self.graph_analyzer = GraphBasedAbuseDetector()
        self.ml_classifier = AbuseClassificationModel()
        self.rule_engine = AbuseRuleEngine()
        self.reputation_system = ReputationSystem()
        self.captcha_challenger = CaptchaChallenger()

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the abuse detection system"""
        try:
            # Initialize all components
            await self.behavioral_analyzer.initialize()
            await self.graph_analyzer.initialize()
            await self.ml_classifier.initialize()
            await self.rule_engine.initialize()
            await self.reputation_system.initialize()
            await self.captcha_challenger.initialize()

            self.is_initialized = True
            logger.info("Abuse detection system initialized")

        except Exception as e:
            logger.error(
                f"Failed to initialize abuse detection system: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
    await self.behavioral_analyzer.cleanup()
            await self.graph_analyzer.cleanup()
            await self.ml_classifier.cleanup()
            await self.rule_engine.cleanup()
            await self.reputation_system.cleanup()
            await self.captcha_challenger.cleanup()

            self.is_initialized = False
            logger.info("Abuse detection system cleanup completed")

        except Exception as e:
            logger.error(
                f"Error during abuse detection system cleanup: {str(e)}")

    async def detect_abuse(
            self,
            abuse_request: AbuseDetectionRequest) -> AbuseDetectionResult:
        """Comprehensive abuse detection"""

        if not self.is_initialized:
            raise RuntimeError("Abuse detection system not initialized")

        try:
            user_id = abuse_request.user_id
            activity_data = abuse_request.activity_data

            # Run all detection methods in parallel
            detection_results = await asyncio.gather(
                self.analyze_behavior(user_id, activity_data),
                self.analyze_graph(user_id, activity_data),
                self.classify_with_ml(activity_data),
                self.check_rules(user_id, activity_data),
                self.get_reputation_score(user_id),
                return_exceptions=True,
            )

            behavioral_signals, graph_signals, ml_prediction, rule_violations, reputation_score = (
                detection_results)

            # Handle exceptions
            behavioral_signals = (
                behavioral_signals
                if not isinstance(behavioral_signals, Exception)
                else BehavioralSignals(anomaly_score=0.0)
            )
            graph_signals = (
                graph_signals
                if not isinstance(graph_signals, Exception)
                else GraphSignals(abuse_probability=0.0)
            )
            ml_prediction = (
                ml_prediction
                if not isinstance(ml_prediction, Exception)
                else MLPrediction(
                    abuse_probability=0.0,
                    confidence=0.0,
                    feature_importance={},
                    model_version="unknown",
                )
            )
            rule_violations = rule_violations if not isinstance(
                rule_violations, Exception) else []
            reputation_score = (
                reputation_score if not isinstance(
                    reputation_score,
                    Exception) else 1.0)

            # Calculate abuse score
            abuse_score = self.calculate_abuse_score(
                {
                    "behavioral": behavioral_signals.anomaly_score,
                    "graph": graph_signals.abuse_probability,
                    "ml_prediction": ml_prediction.abuse_probability,
                    "rule_violations": len(rule_violations),
                    "reputation": 1.0 - reputation_score,  # Higher reputation = lower abuse score
                }
            )

            # Determine threat level
            threat_level = self.determine_threat_level(
                abuse_score, rule_violations)

            # Generate response recommendations
            response_actions = await self.generate_abuse_response(
                abuse_score, threat_level, rule_violations
            )

            return AbuseDetectionResult(
                user_id=user_id,
                abuse_score=abuse_score,
                threat_level=threat_level,
                behavioral_signals=behavioral_signals,
                graph_signals=graph_signals,
                ml_prediction=ml_prediction,
                rule_violations=rule_violations,
                reputation_score=reputation_score,
                response_actions=response_actions,
                detection_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Abuse detection failed: {str(e)}")
            raise

    async def analyze_behavior(
            self,
            user_id: str,
            activity_data: Any) -> BehavioralSignals:
        """Analyze user behavior for anomalies"""
        try:
            return await self.behavioral_analyzer.analyze_behavior(
                user_id=user_id,
                recent_activities=activity_data.recent_activities,
                time_window=timedelta(seconds=self.config.analysis_window),
            )
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {str(e)}")
            return BehavioralSignals(anomaly_score=0.0)

    async def analyze_graph(
            self,
            user_id: str,
            activity_data: Any) -> GraphSignals:
        """Analyze user graph for abuse patterns"""
        try:
            return await self.graph_analyzer.analyze_user_graph(
                user_id=user_id, connection_data=activity_data.connection_data
            )
        except Exception as e:
            logger.error(f"Graph analysis failed: {str(e)}")
            return GraphSignals(abuse_probability=0.0)

    async def classify_with_ml(self, activity_data: Any) -> MLPrediction:
        """Classify abuse using ML models"""
        try:
            return await self.ml_classifier.predict_abuse_probability(
                user_features=activity_data.user_features,
                activity_features=activity_data.activity_features,
            )
        except Exception as e:
            logger.error(f"ML classification failed: {str(e)}")
            return MLPrediction(
                abuse_probability=0.0,
                confidence=0.0,
                feature_importance={},
                model_version="unknown",
            )

    async def check_rules(
            self,
            user_id: str,
            activity_data: Any) -> List[RuleViolation]:
        """Check abuse rules"""
        try:
            return await self.rule_engine.check_abuse_rules(
                user_id=user_id, activity_data=activity_data
            )
        except Exception as e:
            logger.error(f"Rule checking failed: {str(e)}")
            return []

    async def get_reputation_score(self, user_id: str) -> float:
        """Get user reputation score"""
        try:
            return await self.reputation_system.get_user_reputation(user_id)
        except Exception as e:
            logger.error(f"Reputation score retrieval failed: {str(e)}")
            return 1.0

    def calculate_abuse_score(self, signals: Dict[str, float]) -> float:
        """Calculate overall abuse score from individual signals"""
        try:
            # Weighted combination of signals
            weights = {
                "behavioral": 0.3,
                "graph": 0.25,
                "ml_prediction": 0.25,
                "rule_violations": 0.15,
                "reputation": 0.05,
            }

            # Normalize rule violations (0-1 scale)
            rule_violations = signals.get("rule_violations", 0)
            normalized_violations = min(
                rule_violations / 10.0,
                1.0)  # Cap at 10 violations

            # Calculate weighted score
            weighted_score = 0.0
            total_weight = 0.0

            for signal_name, weight in weights.items():
                if signal_name in signals:
                    if signal_name == "rule_violations":
                        signal_value = normalized_violations
                    else:
                        signal_value = signals[signal_name]

                    weighted_score += signal_value * weight
                    total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                abuse_score = weighted_score / total_weight
            else:
                abuse_score = 0.0

            return min(abuse_score, 1.0)

        except Exception as e:
            logger.error(f"Abuse score calculation failed: {str(e)}")
            return 0.0

    def determine_threat_level(
        self, abuse_score: float, rule_violations: List[RuleViolation]
    ) -> str:
        """Determine threat level based on abuse score and violations"""
        try:
            # Count high-severity violations
            high_severity_violations = sum(
                1 for v in rule_violations if v.severity in [
                    "high", "critical"])

            # Determine threat level
            if abuse_score >= 0.9 or high_severity_violations >= 3:
                return "critical"
            elif abuse_score >= 0.7 or high_severity_violations >= 2:
                return "high"
            elif abuse_score >= 0.5 or high_severity_violations >= 1:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Threat level determination failed: {str(e)}")
            return "low"

    async def generate_abuse_response(
            self,
            abuse_score: float,
            threat_level: str,
            rule_violations: List[RuleViolation]) -> List[MitigationAction]:
        """Generate response actions based on abuse detection"""
        try:
            response_actions = []

            # Rate limiting for medium+ threat
            if threat_level in ["medium", "high", "critical"]:
                response_actions.append(
                    MitigationAction(
                        action_type="rate_limit",
                        parameters={
                            "limit_multiplier": 0.5 if threat_level == "medium" else 0.1,
                            "duration_minutes": 60 if threat_level == "medium" else 1440,
                        },
                        priority=1,
                    ))

            # CAPTCHA challenge for high+ threat
            if threat_level in ["high", "critical"]:
                response_actions.append(
                    MitigationAction(
                        action_type="captcha_challenge",
                        parameters={
                            "difficulty": "high" if threat_level == "critical" else "medium",
                            "required_success_rate": 0.8,
                        },
                        priority=2,
                    ))

            # Content restrictions for high+ threat
            if threat_level in ["high", "critical"]:
                response_actions.append(
                    MitigationAction(
                        action_type="content_restriction",
                        parameters={
                            "restriction_level": "moderate" if threat_level == "high" else "strict",
                            "duration_hours": 24 if threat_level == "high" else 168,  # 1 week
                        },
                        priority=3,
                    )
                )

            # Temporary suspension for critical threat
            if threat_level == "critical":
                response_actions.append(
                    MitigationAction(
                        action_type="temporary_suspension",
                        parameters={
                            "duration_hours": 24,
                            "reason": "Critical abuse detected"},
                        priority=4,
                    ))

            # Manual review for high+ threat
            if threat_level in ["high", "critical"]:
                response_actions.append(
                    MitigationAction(
                        action_type="account_review",
                        parameters={
                            "priority": "high" if threat_level == "critical" else "medium",
                            "review_categories": [
                                "behavioral",
                                "content",
                                "network"],
                        },
                        priority=5,
                    ))

            return response_actions

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return []

    async def apply_abuse_mitigation(
        self, user_id: str, mitigation_actions: List[MitigationAction]
    ) -> MitigationResult:
        """Apply abuse mitigation measures"""
        try:
            applied_actions = []

            for action in mitigation_actions:
                try:
                    if action.action_type == "rate_limit":
    await self.apply_rate_limiting(user_id, action.parameters)
                    elif action.action_type == "captcha_challenge":
    await self.captcha_challenger.challenge_user(user_id)
                    elif action.action_type == "temporary_suspension":
    await self.apply_temporary_suspension(user_id, action.parameters)
                    elif action.action_type == "content_restriction":
    await self.apply_content_restrictions(user_id, action.parameters)
                    elif action.action_type == "account_review":
    await self.trigger_manual_review(user_id, action.parameters)

                    applied_actions.append(action)

                except Exception as e:
                    logger.error(
                        f"Failed to apply mitigation action {action.action_type}: {str(e)}"
                    )

            return MitigationResult(
                user_id=user_id,
                applied_actions=applied_actions,
                mitigation_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Abuse mitigation application failed: {str(e)}")
            raise

    async def apply_rate_limiting(
            self, user_id: str, parameters: Dict[str, Any]) -> None:
        """Apply rate limiting to user"""
        try:
            # This would integrate with the rate limiting system
            logger.info(
                f"Applying rate limiting to user {user_id} with parameters {parameters}")
            # Implementation would go here
        except Exception as e:
            logger.error(f"Rate limiting application failed: {str(e)}")
            raise

    async def apply_temporary_suspension(
            self, user_id: str, parameters: Dict[str, Any]) -> None:
        """Apply temporary suspension to user"""
        try:
            # This would integrate with user management system
            logger.info(
                f"Applying temporary suspension to user {user_id} with parameters {parameters}"
            )
            # Implementation would go here
        except Exception as e:
            logger.error(f"Temporary suspension application failed: {str(e)}")
            raise

    async def apply_content_restrictions(
            self, user_id: str, parameters: Dict[str, Any]) -> None:
        """Apply content restrictions to user"""
        try:
            # This would integrate with content management system
            logger.info(
                f"Applying content restrictions to user {user_id} with parameters {parameters}"
            )
            # Implementation would go here
        except Exception as e:
            logger.error(f"Content restrictions application failed: {str(e)}")
            raise

    async def trigger_manual_review(
            self, user_id: str, parameters: Dict[str, Any]) -> None:
        """Trigger manual review for user"""
        try:
            # This would integrate with review queue system
            logger.info(
                f"Triggering manual review for user {user_id} with parameters {parameters}")
            # Implementation would go here
        except Exception as e:
            logger.error(f"Manual review triggering failed: {str(e)}")
            raise

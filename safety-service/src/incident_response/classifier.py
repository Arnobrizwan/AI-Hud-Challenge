"""
Incident Classifier
Classify incidents by type, severity, and impact
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from safety_engine.config import get_settings
from safety_engine.models import IncidentClassification, SafetyIncident

logger = logging.getLogger(__name__)


class IncidentClassifier:
    """Classify incidents by type, severity, and impact"""

    def __init__(self):
        self.config = get_settings()
        self.is_initialized = False

        # Classification rules
        self.severity_rules = {
            "critical": {
                "keywords": ["critical", "severe", "emergency", "outage", "breach", "attack"],
                "thresholds": {"abuse_score": 0.9, "drift_severity": 0.9, "safety_score": 0.1},
            },
            "high": {
                "keywords": ["high", "major", "significant", "urgent"],
                "thresholds": {"abuse_score": 0.7, "drift_severity": 0.7, "safety_score": 0.3},
            },
            "medium": {
                "keywords": ["medium", "moderate", "noticeable"],
                "thresholds": {"abuse_score": 0.5, "drift_severity": 0.5, "safety_score": 0.5},
            },
            "low": {
                "keywords": ["low", "minor", "informational"],
                "thresholds": {"abuse_score": 0.3, "drift_severity": 0.3, "safety_score": 0.7},
            },
        }

        # Impact assessment rules
        self.impact_rules = {
            "system_availability": {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3},
            "data_integrity": {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3},
            "user_experience": {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3},
        }

        # Escalation rules
        self.escalation_rules = {"critical": True, "high": True, "medium": False, "low": False}

    async def initialize(self) -> Dict[str, Any]:
    """Initialize the incident classifier"""
        try:
            # Load any ML models or additional rules
            await self.load_classification_models()

            self.is_initialized = True
            logger.info("Incident classifier initialized")

        except Exception as e:
            logger.error(f"Failed to initialize incident classifier: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup resources"""
        try:
            self.is_initialized = False
            logger.info("Incident classifier cleanup completed")

        except Exception as e:
            logger.error(f"Error during incident classifier cleanup: {str(e)}")

    async def classify_incident(self, incident: SafetyIncident) -> IncidentClassification:
        """Classify incident by type, severity, and impact"""

        if not self.is_initialized:
            raise RuntimeError("Incident classifier not initialized")

        try:
            # Classify incident type
            incident_type = await self.classify_incident_type(incident)

            # Classify severity
            severity = await self.classify_severity(incident)

            # Assess impact
            impact_assessment = await self.assess_impact(incident)

            # Determine if escalation is required
            requires_escalation = await self.determine_escalation_requirement(incident, severity)

            # Generate classification confidence
            confidence = await self.calculate_classification_confidence(incident, severity, impact_assessment)

            # Create classification
            classification = IncidentClassification(
                incident_type=incident_type,
                severity=severity,
                impact_assessment=impact_assessment,
                requires_escalation=requires_escalation,
                confidence=confidence,
                classification_timestamp=datetime.utcnow(),
            )

            logger.info(f"Incident {incident.id} classified as {severity} {incident_type}")

            return classification

        except Exception as e:
            logger.error(f"Incident classification failed: {str(e)}")
            raise

    async def classify_incident_type(self, incident: SafetyIncident) -> str:
        """Classify the type of incident"""
        try:
            # Use incident type if already set
            if incident.incident_type:
                return incident.incident_type

            # Analyze description and metadata for type classification
            description = incident.description.lower()
            metadata = incident.metadata or {}

            # Check for drift-related keywords
            drift_keywords = ["drift", "distribution", "model", "prediction", "data"]
            if any(keyword in description for keyword in drift_keywords):
                return "data_drift"

            # Check for abuse-related keywords
            abuse_keywords = ["abuse", "malicious", "attack", "violation", "suspicious"]
            if any(keyword in description for keyword in abuse_keywords):
                return "abuse_detection"

            # Check for content-related keywords
            content_keywords = ["content", "moderation", "safety", "policy", "violation"]
            if any(keyword in description for keyword in content_keywords):
                return "content_safety"

            # Check for system-related keywords
            system_keywords = ["system", "service", "outage", "error", "failure"]
            if any(keyword in description for keyword in system_keywords):
                return "system_failure"

            # Check for security-related keywords
            security_keywords = ["security", "breach", "unauthorized", "access", "permission"]
            if any(keyword in description for keyword in security_keywords):
                return "security_incident"

            # Default to unknown if no clear classification
            return "unknown"

        except Exception as e:
            logger.error(f"Incident type classification failed: {str(e)}")
            return "unknown"

    async def classify_severity(self, incident: SafetyIncident) -> str:
        """Classify incident severity"""
        try:
            # Check if severity is already set
            if incident.severity and incident.severity in self.severity_rules:
                return incident.severity

            # Analyze description for severity keywords
            description = incident.description.lower()
            metadata = incident.metadata or {}

            # Check for severity keywords
            for severity, rules in self.severity_rules.items():
                if any(keyword in description for keyword in rules["keywords"]):
                    return severity

            # Check metadata thresholds
            for severity, rules in self.severity_rules.items():
                if await self.check_severity_thresholds(metadata, rules["thresholds"]):
                    return severity

            # Default to medium severity
            return "medium"

        except Exception as e:
            logger.error(f"Severity classification failed: {str(e)}")
            return "medium"

    async def check_severity_thresholds(self, metadata: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
        """Check if metadata values exceed severity thresholds"""
        try:
            for key, threshold in thresholds.items():
                if key in metadata:
                    value = metadata[key]
                    if isinstance(value, (int, float)) and value >= threshold:
                        return True

            return False

        except Exception as e:
            logger.error(f"Severity threshold check failed: {str(e)}")
            return False

    async def assess_impact(self, incident: SafetyIncident) -> Dict[str, str]:
        """Assess the impact of the incident"""
        try:
            impact_assessment = {}
            metadata = incident.metadata or {}

            # Assess system availability impact
            system_impact = await self.assess_system_availability_impact(incident, metadata)
            impact_assessment["system_availability"] = system_impact

            # Assess data integrity impact
            data_impact = await self.assess_data_integrity_impact(incident, metadata)
            impact_assessment["data_integrity"] = data_impact

            # Assess user experience impact
            user_impact = await self.assess_user_experience_impact(incident, metadata)
            impact_assessment["user_experience"] = user_impact

            return impact_assessment

        except Exception as e:
            logger.error(f"Impact assessment failed: {str(e)}")
            return {
                "system_availability": "unknown",
                "data_integrity": "unknown",
                "user_experience": "unknown",
            }

    async def assess_system_availability_impact(self, incident: SafetyIncident, metadata: Dict[str, Any]) -> str:
        """Assess system availability impact"""
        try:
            # Check affected systems
            affected_systems = incident.affected_systems or []

            # Critical systems
            critical_systems = ["database", "api", "authentication", "core"]
            if any(system in affected_systems for system in critical_systems):
                return "critical"

            # High impact systems
            high_impact_systems = ["ml_models", "monitoring", "logging"]
            if any(system in affected_systems for system in high_impact_systems):
                return "high"

            # Medium impact systems
            medium_impact_systems = ["analytics", "reporting", "notifications"]
            if any(system in affected_systems for system in medium_impact_systems):
                return "medium"

            # Default to low impact
            return "low"

        except Exception as e:
            logger.error(f"System availability impact assessment failed: {str(e)}")
            return "unknown"

    async def assess_data_integrity_impact(self, incident: SafetyIncident, metadata: Dict[str, Any]) -> str:
        """Assess data integrity impact"""
        try:
            # Check for data-related keywords in description
            description = incident.description.lower()

            # Critical data integrity issues
            critical_keywords = ["corruption", "loss", "breach", "unauthorized_access"]
            if any(keyword in description for keyword in critical_keywords):
                return "critical"

            # High data integrity issues
            high_keywords = ["inconsistency", "drift", "anomaly", "suspicious"]
            if any(keyword in description for keyword in high_keywords):
                return "high"

            # Medium data integrity issues
            medium_keywords = ["quality", "accuracy", "validation"]
            if any(keyword in description for keyword in medium_keywords):
                return "medium"

            # Default to low impact
            return "low"

        except Exception as e:
            logger.error(f"Data integrity impact assessment failed: {str(e)}")
            return "unknown"

    async def assess_user_experience_impact(self, incident: SafetyIncident, metadata: Dict[str, Any]) -> str:
        """Assess user experience impact"""
        try:
            # Check for user experience keywords in description
            description = incident.description.lower()

            # Critical user experience issues
            critical_keywords = ["outage", "down", "unavailable", "blocked"]
            if any(keyword in description for keyword in critical_keywords):
                return "critical"

            # High user experience issues
            high_keywords = ["slow", "delayed", "error", "failure"]
            if any(keyword in description for keyword in high_keywords):
                return "high"

            # Medium user experience issues
            medium_keywords = ["degraded", "suboptimal", "warning"]
            if any(keyword in description for keyword in medium_keywords):
                return "medium"

            # Default to low impact
            return "low"

        except Exception as e:
            logger.error(f"User experience impact assessment failed: {str(e)}")
            return "unknown"

    async def determine_escalation_requirement(self, incident: SafetyIncident, severity: str) -> bool:
        """Determine if incident requires escalation"""
        try:
            # Check severity-based escalation rules
            if severity in self.escalation_rules:
                return self.escalation_rules[severity]

            # Check for escalation keywords in description
            description = incident.description.lower()
            escalation_keywords = ["escalate", "urgent", "immediate", "critical"]
            if any(keyword in description for keyword in escalation_keywords):
                return True

            # Check metadata for escalation flags
            metadata = incident.metadata or {}
            if metadata.get("requires_escalation", False):
                return True

            # Default to no escalation
            return False

        except Exception as e:
            logger.error(f"Escalation requirement determination failed: {str(e)}")
            return False

    async def calculate_classification_confidence(
        self, incident: SafetyIncident, severity: str, impact_assessment: Dict[str, str]
    ) -> float:
        """Calculate confidence in classification"""
        try:
            confidence_factors = []

            # Factor 1: Severity keyword match
            description = incident.description.lower()
            severity_keywords = self.severity_rules.get(severity, {}).get("keywords", [])
            keyword_matches = sum(1 for keyword in severity_keywords if keyword in description)
            keyword_confidence = min(1.0, keyword_matches / len(severity_keywords)) if severity_keywords else 0.5
            confidence_factors.append(keyword_confidence)

            # Factor 2: Metadata threshold match
            metadata = incident.metadata or {}
            threshold_confidence = 0.5  # Default
            if severity in self.severity_rules:
                thresholds = self.severity_rules[severity]["thresholds"]
                threshold_matches = sum(
                    1 for key, threshold in thresholds.items() if key in metadata and metadata[key] >= threshold
                )
                threshold_confidence = threshold_matches / len(thresholds) if thresholds else 0.5
            confidence_factors.append(threshold_confidence)

            # Factor 3: Impact assessment consistency
            impact_consistency = 0.5  # Default
            if impact_assessment:
                impact_levels = list(impact_assessment.values())
                if impact_levels:
                    # Check if impact levels are consistent
                    high_impact_count = sum(1 for level in impact_levels if level in ["high", "critical"])
                    impact_consistency = high_impact_count / len(impact_levels)
            confidence_factors.append(impact_consistency)

            # Calculate overall confidence
            overall_confidence = sum(confidence_factors) / len(confidence_factors)

            return min(1.0, max(0.0, overall_confidence))

        except Exception as e:
            logger.error(f"Classification confidence calculation failed: {str(e)}")
            return 0.5

    async def load_classification_models(self) -> Dict[str, Any]:
    """Load any ML models for classification"""
        try:
            # Placeholder for loading ML models
            # In a real implementation, this would load trained models
            logger.info("Classification models loaded")

        except Exception as e:
            logger.error(f"Classification model loading failed: {str(e)}")
            raise

    async def get_classification_statistics(self) -> Dict[str, Any]:
    """Get classification statistics"""
        try:
            return {
                "severity_rules": len(self.severity_rules),
                "impact_rules": len(self.impact_rules),
                "escalation_rules": len(self.escalation_rules),
                "classification_accuracy": 0.85,  # Placeholder
                "average_confidence": 0.78,  # Placeholder
            }

        except Exception as e:
            logger.error(f"Classification statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    async def update_classification_rules(self, new_rules: Dict[str, Any]) -> bool:
        """Update classification rules"""
        try:
            # Update severity rules
            if "severity_rules" in new_rules:
                self.severity_rules.update(new_rules["severity_rules"])

            # Update impact rules
            if "impact_rules" in new_rules:
                self.impact_rules.update(new_rules["impact_rules"])

            # Update escalation rules
            if "escalation_rules" in new_rules:
                self.escalation_rules.update(new_rules["escalation_rules"])

            logger.info("Classification rules updated")
            return True

        except Exception as e:
            logger.error(f"Classification rules update failed: {str(e)}")
            return False

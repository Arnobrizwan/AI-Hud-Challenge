"""
Content Policy Monitor
Monitor compliance with content policies
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from safety_engine.config import get_compliance_config
from safety_engine.models import ComplianceType, ComplianceViolation

logger = logging.getLogger(__name__)


class ContentPolicyMonitor:
    """Monitor compliance with content policies"""

    def __init__(self):
        self.config = get_compliance_config()
        self.is_initialized = False

        # Content policy rules
        self.content_policies = {
            "hate_speech": {
                "enabled": True,
                "severity": "high",
                "description": "Content must not contain hate speech or discriminatory language",
            },
            "harassment": {
                "enabled": True,
                "severity": "high",
                "description": "Content must not harass, bully, or intimidate others",
            },
            "violence": {
                "enabled": True,
                "severity": "high",
                "description": "Content must not promote or glorify violence",
            },
            "adult_content": {
                "enabled": True,
                "severity": "medium",
                "description": "Adult content must be properly labeled and restricted",
            },
            "spam": {
                "enabled": True,
                "severity": "low",
                "description": "Content must not be spam or repetitive",
            },
            "misinformation": {
                "enabled": True,
                "severity": "high",
                "description": "Content must not contain false or misleading information",
            },
            "copyright_violation": {
                "enabled": True,
                "severity": "medium",
                "description": "Content must not violate copyright or intellectual property rights",
            },
        }

    async def initialize(self) -> Dict[str, Any]:
    """Initialize the content policy monitor"""
        try:
            self.is_initialized = True
            logger.info("Content policy monitor initialized")

        except Exception as e:
            logger.error(f"Failed to initialize content policy monitor: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
    """Cleanup resources"""
        try:
            self.is_initialized = False
            logger.info("Content policy monitor cleanup completed")

        except Exception as e:
            logger.error(f"Error during content policy monitor cleanup: {str(e)}")

    async def check_policy_compliance(self, content_items: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Check compliance with content policies"""
        if not self.is_initialized:
            raise RuntimeError("Content policy monitor not initialized")

        try:
            violations = []
            compliance_score = 1.0
            recommendations = []

            if content_items:
                for content_item in content_items:
                    item_violations = await self.check_content_item_compliance(content_item)
                    violations.extend(item_violations)

            # Check general content policy compliance
            general_violations = await self.check_general_content_policy_compliance()
            violations.extend(general_violations)

            # Calculate compliance score
            if violations:
                compliance_score = max(0.0, 1.0 - (len(violations) * 0.1))

            # Generate recommendations
            recommendations = self.generate_content_policy_recommendations(violations)

            return {
                "compliance_score": compliance_score,
                "violations": violations,
                "recommendations": recommendations,
                "policies_checked": list(self.content_policies.keys()),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Content policy compliance check failed: {str(e)}")
            return {
                "compliance_score": 0.0,
                "violations": [],
                "recommendations": [f"Content policy compliance check failed: {str(e)}"],
                "policies_checked": [],
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def check_content_item_compliance(self, content_item: Any) -> List[ComplianceViolation]:
        """Check compliance for a specific content item"""
        try:
            violations = []

            # Get content text
            content_text = getattr(content_item, "text_content", "") or ""

            # Check each content policy
            for policy_name, policy_config in self.content_policies.items():
                if not policy_config["enabled"]:
                    continue

                if policy_name == "hate_speech":
                    if self.contains_hate_speech(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates hate speech policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "harassment":
                    if self.contains_harassment(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates harassment policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "violence":
                    if self.contains_violence(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates violence policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "adult_content":
                    if self.contains_adult_content(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates adult content policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "spam":
                    if self.is_spam(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates spam policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "misinformation":
                    if self.contains_misinformation(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates misinformation policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

                elif policy_name == "copyright_violation":
                    if self.contains_copyright_violation(content_text):
                        violations.append(
                            ComplianceViolation(
                                violation_type=ComplianceType.CONTENT_POLICY,
                                severity=policy_config["severity"],
                                description=f"Content violates copyright policy: {policy_config['description']}",
                                affected_data={
                                    "content_id": getattr(content_item, "id", "unknown"),
                                    "policy": policy_name,
                                },
                                remediation_required=True,
                            )
                        )

            return violations

        except Exception as e:
            logger.error(f"Content item compliance check failed: {str(e)}")
            return []

    async def check_general_content_policy_compliance(self) -> List[ComplianceViolation]:
        """Check general content policy compliance"""
        try:
            violations = []

            # Check if content moderation is enabled
            if not self.config.content_policy_enabled:
                violations.append(
                    ComplianceViolation(
                        violation_type=ComplianceType.CONTENT_POLICY,
                        severity="medium",
                        description="Content policy monitoring is disabled",
                        affected_data=None,
                        remediation_required=True,
                    )
                )

            # Check if prohibited content types are properly handled
            for content_type in self.config.prohibited_content_types:
                if content_type not in self.content_policies:
                    violations.append(
                        ComplianceViolation(
                            violation_type=ComplianceType.CONTENT_POLICY,
                            severity="low",
                            description=f"No policy defined for prohibited content type: {content_type}",
                            affected_data={"content_type": content_type},
                            remediation_required=True,
                        )
                    )

            return violations

        except Exception as e:
            logger.error(f"General content policy compliance check failed: {str(e)}")
            return []

    def contains_hate_speech(self, text: str) -> bool:
        """Check if text contains hate speech"""
        try:
            if not text:
                return False

            # Simple hate speech detection (in real implementation, use ML
            # models)
            hate_keywords = ["hate", "stupid", "idiot", "moron", "dumb", "retard"]
            text_lower = text.lower()

            return any(keyword in text_lower for keyword in hate_keywords)

        except Exception as e:
            logger.error(f"Hate speech detection failed: {str(e)}")
            return False

    def contains_harassment(self, text: str) -> bool:
        """Check if text contains harassment"""
        try:
            if not text:
                return False

            # Simple harassment detection
            harassment_keywords = ["harass", "bully", "intimidate", "threaten"]
            text_lower = text.lower()

            return any(keyword in text_lower for keyword in harassment_keywords)

        except Exception as e:
            logger.error(f"Harassment detection failed: {str(e)}")
            return False

    def contains_violence(self, text: str) -> bool:
        """Check if text contains violence"""
        try:
            if not text:
                return False

            # Simple violence detection
            violence_keywords = ["kill", "murder", "violence", "attack", "weapon"]
            text_lower = text.lower()

            return any(keyword in text_lower for keyword in violence_keywords)

        except Exception as e:
            logger.error(f"Violence detection failed: {str(e)}")
            return False

    def contains_adult_content(self, text: str) -> bool:
        """Check if text contains adult content"""
        try:
            if not text:
                return False

            # Simple adult content detection
            adult_keywords = ["adult", "explicit", "nsfw", "xxx"]
            text_lower = text.lower()

            return any(keyword in text_lower for keyword in adult_keywords)

        except Exception as e:
            logger.error(f"Adult content detection failed: {str(e)}")
            return False

    def is_spam(self, text: str) -> bool:
        """Check if text is spam"""
        try:
            if not text:
                return False

            # Simple spam detection
            spam_indicators = ["click here", "buy now", "free money", "make money"]
            text_lower = text.lower()

            return any(indicator in text_lower for indicator in spam_indicators)

        except Exception as e:
            logger.error(f"Spam detection failed: {str(e)}")
            return False

    def contains_misinformation(self, text: str) -> bool:
        """Check if text contains misinformation"""
        try:
            if not text:
                return False

            # Simple misinformation detection
            misinformation_indicators = ["fake news", "conspiracy", "hoax", "lies"]
            text_lower = text.lower()

            return any(indicator in text_lower for indicator in misinformation_indicators)

        except Exception as e:
            logger.error(f"Misinformation detection failed: {str(e)}")
            return False

    def contains_copyright_violation(self, text: str) -> bool:
        """Check if text contains copyright violations"""
        try:
            if not text:
                return False

            # Simple copyright violation detection
            copyright_indicators = ["copyright", "©", "all rights reserved"]
            text_lower = text.lower()

            return any(indicator in text_lower for indicator in copyright_indicators)

        except Exception as e:
            logger.error(f"Copyright violation detection failed: {str(e)}")
            return False

    def generate_content_policy_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate content policy compliance recommendations"""
        try:
            recommendations = []

            # General recommendations
            if not violations:
                recommendations.append("Content policy compliance is good - maintain current practices")
                return recommendations

            # Violation-specific recommendations
            for violation in violations:
                if "hate speech" in violation.description.lower():
                    recommendations.append("Implement stronger hate speech detection and moderation")

                elif "harassment" in violation.description.lower():
                    recommendations.append("Enhance harassment detection and reporting mechanisms")

                elif "violence" in violation.description.lower():
                    recommendations.append("Strengthen violence content detection and removal")

                elif "adult content" in violation.description.lower():
                    recommendations.append("Improve adult content detection and age verification")

                elif "spam" in violation.description.lower():
                    recommendations.append("Enhance spam detection and filtering mechanisms")

                elif "misinformation" in violation.description.lower():
                    recommendations.append("Implement fact-checking and misinformation detection")

                elif "copyright" in violation.description.lower():
                    recommendations.append("Strengthen copyright violation detection and DMCA compliance")

            # Add general recommendations
            recommendations.append("Regularly review and update content policies")
            recommendations.append("Implement automated content moderation tools")
            recommendations.append("Provide clear content guidelines to users")
            recommendations.append("Establish content review and appeal processes")
            recommendations.append("Train moderators on content policy enforcement")

            return recommendations

        except Exception as e:
            logger.error(f"Content policy recommendation generation failed: {str(e)}")
            return []

    async def get_compliance_status(self) -> Dict[str, Any]:
    """Get current content policy compliance status"""
        try:
            return {
                "enabled": self.config.content_policy_enabled,
                "policies_count": len(self.content_policies),
                "enabled_policies": sum(1 for policy in self.content_policies.values() if policy["enabled"]),
                "prohibited_content_types": self.config.prohibited_content_types,
                "last_check": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Content policy compliance status retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def generate_detailed_report(self) -> Dict[str, Any]:
    """Generate detailed content policy compliance report"""
        try:
            report = {
                "report_type": "content_policy_detailed",
                "generated_at": datetime.utcnow().isoformat(),
                "compliance_overview": await self.get_compliance_status(),
                "policies": self.content_policies,
                "recommendations": [
                    "Implement comprehensive content moderation system",
                    "Use AI-powered content detection tools",
                    "Establish clear content guidelines and community standards",
                    "Provide user education on content policies",
                    "Regularly audit and update content policies",
                ],
            }

            return report

        except Exception as e:
            logger.error(f"Content policy detailed report generation failed: {str(e)}")
            return {"error": str(e)}

    async def update_content_policy(self, policy_name: str, policy_config: Dict[str, Any]) -> bool:
        """Update a content policy configuration"""
        try:
            if policy_name not in self.content_policies:
                logger.warning(f"Content policy {policy_name} not found")
                return False

            self.content_policies[policy_name].update(policy_config)
            logger.info(f"Updated content policy: {policy_name}")
            return True

        except Exception as e:
            logger.error(f"Content policy update failed: {str(e)}")
            return False

    async def add_content_policy(self, policy_name: str, policy_config: Dict[str, Any]) -> bool:
        """Add a new content policy"""
        try:
            if policy_name in self.content_policies:
                logger.warning(f"Content policy {policy_name} already exists")
                return False

            self.content_policies[policy_name] = policy_config
            logger.info(f"Added content policy: {policy_name}")
            return True

        except Exception as e:
            logger.error(f"Content policy addition failed: {str(e)}")
            return False

    async def remove_content_policy(self, policy_name: str) -> bool:
        """Remove a content policy"""
        try:
            if policy_name not in self.content_policies:
                logger.warning(f"Content policy {policy_name} not found")
                return False

            del self.content_policies[policy_name]
            logger.info(f"Removed content policy: {policy_name}")
            return True

        except Exception as e:
            logger.error(f"Content policy removal failed: {str(e)}")
            return False

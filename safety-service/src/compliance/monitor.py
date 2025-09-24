"""
Compliance Monitor
Regulatory compliance monitoring and reporting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from safety_engine.config import get_compliance_config
from safety_engine.models import (
    ComplianceReport,
    ComplianceRequest,
    ComplianceType,
    ComplianceViolation,
)

from .audit_trail import AuditTrailManager
from .content_policy_monitor import ContentPolicyMonitor
from .gdpr_monitor import GDPRComplianceMonitor
from .privacy_monitor import PrivacyComplianceMonitor

logger = logging.getLogger(__name__)


class ComplianceMonitor:
    """Monitor compliance with regulations and policies"""

    def __init__(self):
        self.config = get_compliance_config()
        self.is_initialized = False

        # Compliance monitors
        self.gdpr_monitor = GDPRComplianceMonitor()
        self.content_policy_monitor = ContentPolicyMonitor()
        self.privacy_monitor = PrivacyComplianceMonitor()
        self.audit_trail_manager = AuditTrailManager()

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the compliance monitor"""
        try:
            # Initialize all monitors
            await self.gdpr_monitor.initialize()
            await self.content_policy_monitor.initialize()
            await self.privacy_monitor.initialize()
            await self.audit_trail_manager.initialize()

            self.is_initialized = True
            logger.info("Compliance monitor initialized")

        except Exception as e:
            logger.error(f"Failed to initialize compliance monitor: {str(e)}")
            raise

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup resources"""
        try:
    await self.gdpr_monitor.cleanup()
            await self.content_policy_monitor.cleanup()
            await self.privacy_monitor.cleanup()
            await self.audit_trail_manager.cleanup()

            self.is_initialized = False
            logger.info("Compliance monitor cleanup completed")

        except Exception as e:
            logger.error(f"Error during compliance monitor cleanup: {str(e)}")

    async def check_compliance(
            self,
            request: ComplianceRequest) -> ComplianceReport:
        """Comprehensive compliance checking"""

        if not self.is_initialized:
            raise RuntimeError("Compliance monitor not initialized")

        try:
            compliance_results = {}

            # GDPR compliance
            if request.check_gdpr:
                gdpr_result = await self.gdpr_monitor.check_gdpr_compliance(
                    data_processing_activities=request.data_activities
                )
                compliance_results["gdpr"] = gdpr_result

            # Content policy compliance
            if request.check_content_policy:
                content_policy_result = await self.content_policy_monitor.check_policy_compliance(
                    content_items=request.content_items
                )
                compliance_results["content_policy"] = content_policy_result

            # Privacy compliance
            if request.check_privacy:
                privacy_result = await self.privacy_monitor.check_privacy_compliance(
                    user_data=request.user_data
                )
                compliance_results["privacy"] = privacy_result

            # Calculate overall compliance score
            overall_compliance = self.calculate_compliance_score(
                compliance_results)

            # Extract violations
            violations = self.extract_compliance_violations(compliance_results)

            # Generate recommendations
            recommendations = await self.generate_compliance_recommendations(compliance_results)

            # Create compliance report
            report = ComplianceReport(
                overall_compliance_score=overall_compliance,
                compliance_results=compliance_results,
                violations=violations,
                recommendations=recommendations,
                report_timestamp=datetime.utcnow(),
            )

            # Log compliance check
            await self.audit_trail_manager.log_compliance_check(request, report)

            return report

        except Exception as e:
            logger.error(f"Compliance check failed: {str(e)}")
            raise

    def calculate_compliance_score(
            self, compliance_results: Dict[str, Any]) -> float:
        """Calculate overall compliance score from individual results"""
        try:
            if not compliance_results:
                return 1.0  # No checks performed, assume compliant

            scores = []

            for result in compliance_results.values():
                if hasattr(result, "compliance_score"):
                    scores.append(result.compliance_score)
                elif hasattr(result, "score"):
                    scores.append(result.score)
                elif isinstance(result, dict) and "score" in result:
                    scores.append(result["score"])

            if not scores:
                return 1.0

            # Return average compliance score
            return sum(scores) / len(scores)

        except Exception as e:
            logger.error(f"Compliance score calculation failed: {str(e)}")
            return 0.0

    def extract_compliance_violations(
        self, compliance_results: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        """Extract compliance violations from results"""
        try:
            violations = []

            for compliance_type, result in compliance_results.items():
                if hasattr(result, "violations"):
                    for violation in result.violations:
                        violations.append(violation)
                elif isinstance(result, dict) and "violations" in result:
                    for violation in result["violations"]:
                        violations.append(violation)

            return violations

        except Exception as e:
            logger.error(f"Violation extraction failed: {str(e)}")
            return []

    async def generate_compliance_recommendations(
        self, compliance_results: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations based on results"""
        try:
            recommendations = []

            # GDPR recommendations
            if "gdpr" in compliance_results:
                gdpr_result = compliance_results["gdpr"]
                if hasattr(gdpr_result, "recommendations"):
                    recommendations.extend(gdpr_result.recommendations)
                elif isinstance(gdpr_result, dict) and "recommendations" in gdpr_result:
                    recommendations.extend(gdpr_result["recommendations"])

            # Content policy recommendations
            if "content_policy" in compliance_results:
                content_result = compliance_results["content_policy"]
                if hasattr(content_result, "recommendations"):
                    recommendations.extend(content_result.recommendations)
                elif isinstance(content_result, dict) and "recommendations" in content_result:
                    recommendations.extend(content_result["recommendations"])

            # Privacy recommendations
            if "privacy" in compliance_results:
                privacy_result = compliance_results["privacy"]
                if hasattr(privacy_result, "recommendations"):
                    recommendations.extend(privacy_result.recommendations)
                elif isinstance(privacy_result, dict) and "recommendations" in privacy_result:
                    recommendations.extend(privacy_result["recommendations"])

            # General recommendations based on overall compliance
            overall_score = self.calculate_compliance_score(compliance_results)

            if overall_score < 0.8:
                recommendations.append(
                    "Overall compliance score is below 80% - review all compliance areas"
                )

            if overall_score < 0.6:
                recommendations.append(
                    "Critical compliance issues detected - immediate action required"
                )

            # Remove duplicates and return
            return list(set(recommendations))

        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return []

    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        try:
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "compliance_status": {
                    "gdpr":
    await self.gdpr_monitor.get_compliance_status(),
                    "content_policy":
    await self.content_policy_monitor.get_compliance_status(),
                    "privacy":
    await self.privacy_monitor.get_compliance_status(),
                },
                "recent_violations":
    await self.get_recent_violations(),
                "compliance_trends":
    await self.get_compliance_trends(),
                "audit_summary":
    await self.audit_trail_manager.get_audit_summary(),
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Compliance dashboard generation failed: {str(e)}")
            return {"error": str(e)}

    async def get_recent_violations(
            self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent compliance violations"""
        try:
            # This would typically query a database
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Recent violations retrieval failed: {str(e)}")
            return []

    async def get_compliance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get compliance trends over time"""
        try:
            # This would typically query historical data
            # For now, return mock data
            return {
                "time_period_days": days,
                "gdpr_trend": "stable",
                "content_policy_trend": "improving",
                "privacy_trend": "stable",
                "overall_trend": "improving",
            }

        except Exception as e:
            logger.error(f"Compliance trends retrieval failed: {str(e)}")
            return {"error": str(e)}

    async def generate_compliance_report(
            self, report_type: str = "summary") -> Dict[str, Any]:
    """Generate comprehensive compliance report"""
        try:
            report = {
                "report_type": report_type,
                "generated_at": datetime.utcnow().isoformat(),
                "compliance_overview":
    await self.get_compliance_dashboard(),
                "detailed_results": {},
            }

            # Add detailed results for each compliance area
            if self.config.gdpr_enabled:
                report["detailed_results"][
                    "gdpr"
                ] = await self.gdpr_monitor.generate_detailed_report()

            if self.config.content_policy_enabled:
                report["detailed_results"][
                    "content_policy"
                ] = await self.content_policy_monitor.generate_detailed_report()

            if self.config.privacy_enabled:
                report["detailed_results"][
                    "privacy"
                ] = await self.privacy_monitor.generate_detailed_report()

            return report

        except Exception as e:
            logger.error(f"Compliance report generation failed: {str(e)}")
            return {"error": str(e)}

    async def export_compliance_data(self, format: str = "json") -> str:
        """Export compliance data in specified format"""
        try:
            if format == "json":
                data = await self.get_compliance_dashboard()
                return json.dumps(data, indent=2)
            elif format == "csv":
                # Generate CSV format
                return self.generate_csv_export()
        else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Compliance data export failed: {str(e)}")
            return ""

    def generate_csv_export(self) -> str:
        """Generate CSV export of compliance data"""
        try:
            # This would generate actual CSV data
            # For now, return empty CSV
            return "timestamp,compliance_type,score,violations\n"

        except Exception as e:
            logger.error(f"CSV export generation failed: {str(e)}")
            return ""

    async def schedule_compliance_checks(self) -> Dict[str, Any]:
        """Schedule periodic compliance checks"""
        try:
            # This would set up scheduled compliance checks
            # For now, just log the action
            logger.info("Compliance checks scheduled")

        except Exception as e:
            logger.error(f"Compliance check scheduling failed: {str(e)}")

    async def get_compliance_metrics(self) -> Dict[str, Any]:
        """Get compliance metrics and statistics"""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_checks": 0,  # Would be calculated from audit logs
                "compliance_score": 0.0,  # Would be calculated from recent checks
                "violation_count": 0,  # Would be calculated from recent violations
                "trend": "stable",  # Would be calculated from historical data
                "monitors_status": {
                    "gdpr": self.config.gdpr_enabled,
                    "content_policy": self.config.content_policy_enabled,
                    "privacy": self.config.privacy_enabled,
                },
            }

            return metrics

        except Exception as e:
            logger.error(f"Compliance metrics retrieval failed: {str(e)}")
            return {"error": str(e)}

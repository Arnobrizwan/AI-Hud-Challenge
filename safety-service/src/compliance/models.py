"""
Compliance Models
Data models for compliance monitoring system
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BaseComplianceModel(BaseModel):
    """Base model for compliance monitoring"""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ComplianceRequest(BaseComplianceModel):
    """Request for compliance checking"""

    check_gdpr: bool = True
    check_content_policy: bool = True
    check_privacy: bool = True
    data_activities: Optional[Dict[str, Any]] = None
    content_items: Optional[List[Any]] = None
    user_data: Optional[Dict[str, Any]] = None


class GDPRComplianceResult(BaseComplianceModel):
    """GDPR compliance check result"""

    is_compliant: bool
    violations: List[str]
    recommendations: List[str]
    data_processing_lawful: bool
    consent_obtained: bool
    data_subject_rights: Dict[str, bool]


class ContentPolicyResult(BaseComplianceModel):
    """Content policy compliance result"""

    is_compliant: bool
    violations: List[str]
    recommendations: List[str]
    policy_checks: Dict[str, bool]


class PrivacyComplianceResult(BaseComplianceModel):
    """Privacy compliance check result"""

    is_compliant: bool
    violations: List[str]
    recommendations: List[str]
    data_minimization: bool
    purpose_limitation: bool
    storage_limitation: bool


class ComplianceReport(BaseComplianceModel):
    """Comprehensive compliance report"""

    overall_compliance_score: float
    compliance_results: Dict[str, Any]
    violations: List[str]
    recommendations: List[str]
    report_timestamp: datetime

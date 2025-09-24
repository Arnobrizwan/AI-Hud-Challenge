"""
Custom exceptions for MLOps Orchestration Service
"""


class MLOpsException(Exception):
    """Base exception for MLOps service"""

    pass


class PipelineError(MLOpsException):
    """Pipeline-related errors"""

    pass


class TrainingError(MLOpsException):
    """Training-related errors"""

    pass


class DeploymentError(MLOpsException):
    """Deployment-related errors"""

    pass


class MonitoringError(MLOpsException):
    """Monitoring-related errors"""

    pass


class FeatureStoreError(MLOpsException):
    """Feature store-related errors"""

    pass


class RetrainingError(MLOpsException):
    """Retraining-related errors"""

    pass


class ValidationError(MLOpsException):
    """Validation-related errors"""

    pass


class ResourceError(MLOpsException):
    """Resource-related errors"""

    pass


class AuthenticationError(MLOpsException):
    """Authentication-related errors"""

    pass


class AuthorizationError(MLOpsException):
    """Authorization-related errors"""

    pass


class ConfigurationError(MLOpsException):
    """Configuration-related errors"""

    pass


class ExternalServiceError(MLOpsException):
    """External service-related errors"""

    pass

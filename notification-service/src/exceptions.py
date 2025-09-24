"""
Custom exceptions for notification decisioning service.
"""


class NotificationError(Exception):
    """Base exception for notification-related errors."""
    pass


class NotificationDeliveryError(NotificationError):
    """Exception raised when notification delivery fails."""
    pass


class NotificationDecisionError(NotificationError):
    """Exception raised when notification decisioning fails."""
    pass


class FatigueDetectionError(NotificationError):
    """Exception raised when fatigue detection fails."""
    pass


class TimingPredictionError(NotificationError):
    """Exception raised when timing prediction fails."""
    pass


class RelevanceScoringError(NotificationError):
    """Exception raised when relevance scoring fails."""
    pass


class PreferenceManagementError(NotificationError):
    """Exception raised when preference management fails."""
    pass


class ContentOptimizationError(NotificationError):
    """Exception raised when content optimization fails."""
    pass


class ABTestingError(NotificationError):
    """Exception raised when A/B testing fails."""
    pass


class DeliveryChannelError(NotificationError):
    """Exception raised when delivery channel operations fail."""
    pass

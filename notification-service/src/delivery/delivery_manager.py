"""
Multi-channel delivery system for notifications.
"""

import asyncio
import logging
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx
import structlog
from firebase_admin import messaging

from ..config import get_settings
from ..exceptions import DeliveryChannelError
from ..models.schemas import DeliveryChannel, DeliveryResult, NotificationDecision, NotificationType

logger = structlog.get_logger()


class FCMClient:
    """Firebase Cloud Messaging client for push notifications."""

    def __init__(self):
        self.settings = get_settings()
        self.app = None

    async def initialize(self) -> None:
        """Initialize FCM client."""
        logger.info("Initializing FCM client")

        try:
            import firebase_admin
            from firebase_admin import credentials

            if self.settings.FIREBASE_CREDENTIALS_PATH:
                cred = credentials.Certificate(self.settings.FIREBASE_CREDENTIALS_PATH)
                self.app = firebase_admin.initialize_app(cred)
            else:
                # Use default credentials (e.g., from environment)
                self.app = firebase_admin.initialize_app()

            logger.info("FCM client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize FCM client: {e}")
            raise DeliveryChannelError(f"FCM initialization failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup FCM client."""
        logger.info("Cleaning up FCM client")
        # FCM cleanup if needed

    async def send_notification(self, decision: NotificationDecision) -> DeliveryResult:
        """Send push notification via FCM."""

        try:
            # Get user's FCM token (in real implementation, fetch from database)
            fcm_token = await self._get_user_fcm_token(decision.user_id)

            if not fcm_token:
                raise DeliveryChannelError(f"No FCM token found for user {decision.user_id}")

            # Create FCM message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=decision.content.title,
                    body=decision.content.body,
                    image=decision.content.image_url,
                ),
                data={
                    "action_url": decision.content.action_url or "",
                    "category": decision.content.category,
                    "priority": decision.content.priority.value,
                    "notification_id": str(uuid.uuid4()),
                },
                token=fcm_token,
                android=messaging.AndroidConfig(
                    priority=(
                        "high"
                        if decision.content.priority.value in ["high", "urgent"]
                        else "normal"
                    ),
                    notification=messaging.AndroidNotification(
                        channel_id="default",
                        priority=(
                            "high"
                            if decision.content.priority.value in ["high", "urgent"]
                            else "normal"
                        ),
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=decision.content.title, body=decision.content.body
                            ),
                            badge=1,
                            sound="default",
                        )
                    )
                ),
            )

            # Send message
            response = messaging.send(message, app=self.app)

            logger.info(
                "FCM notification sent successfully", user_id=decision.user_id, message_id=response
            )

            return DeliveryResult(
                success=True,
                delivery_id=response,
                channel=DeliveryChannel.PUSH,
                delivered_at=datetime.utcnow(),
                was_engaged=False,  # Will be updated when user interacts
            )

        except Exception as e:
            logger.error(
                "FCM notification failed", user_id=decision.user_id, error=str(e), exc_info=True
            )
            raise DeliveryChannelError(f"FCM delivery failed: {e}")

    async def _get_user_fcm_token(self, user_id: str) -> Optional[str]:
        """Get user's FCM token."""
        # Mock implementation - would fetch from database
        return f"mock_fcm_token_{user_id}"


class EmailClient:
    """Email client for email notifications."""

    def __init__(self):
        self.settings = get_settings()
        self.smtp_client = None

    async def initialize(self) -> None:
        """Initialize email client."""
        logger.info("Initializing email client")

        if not all(
            [self.settings.SMTP_HOST, self.settings.SMTP_USERNAME, self.settings.SMTP_PASSWORD]
        ):
            logger.warning("Email configuration incomplete, email notifications disabled")
            return

        try:
            self.smtp_client = smtplib.SMTP(self.settings.SMTP_HOST, self.settings.SMTP_PORT)

            if self.settings.SMTP_USE_TLS:
                self.smtp_client.starttls()

            self.smtp_client.login(self.settings.SMTP_USERNAME, self.settings.SMTP_PASSWORD)

            logger.info("Email client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize email client: {e}")
            raise DeliveryChannelError(f"Email client initialization failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup email client."""
        logger.info("Cleaning up email client")
        if self.smtp_client:
            self.smtp_client.quit()

    async def send_notification(self, decision: NotificationDecision) -> DeliveryResult:
        """Send email notification."""

        try:
            if not self.smtp_client:
                raise DeliveryChannelError("Email client not initialized")

            # Get user's email address (in real implementation, fetch from database)
            user_email = await self._get_user_email(decision.user_id)

            if not user_email:
                raise DeliveryChannelError(f"No email address found for user {decision.user_id}")

            # Create email message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = decision.content.title
            msg["From"] = self.settings.SMTP_USERNAME
            msg["To"] = user_email

            # Create HTML content
            html_content = self._create_email_html(decision)
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Create plain text content
            text_content = self._create_email_text(decision)
            text_part = MIMEText(text_content, "plain")
            msg.attach(text_part)

            # Send email
            self.smtp_client.send_message(msg)

            delivery_id = str(uuid.uuid4())

            logger.info(
                "Email notification sent successfully",
                user_id=decision.user_id,
                delivery_id=delivery_id,
            )

            return DeliveryResult(
                success=True,
                delivery_id=delivery_id,
                channel=DeliveryChannel.EMAIL,
                delivered_at=datetime.utcnow(),
                was_engaged=False,
            )

        except Exception as e:
            logger.error(
                "Email notification failed", user_id=decision.user_id, error=str(e), exc_info=True
            )
            raise DeliveryChannelError(f"Email delivery failed: {e}")

    async def _get_user_email(self, user_id: str) -> Optional[str]:
        """Get user's email address."""
        # Mock implementation - would fetch from database
        return f"user_{user_id}@example.com"

    def _create_email_html(self, decision: NotificationDecision) -> str:
        """Create HTML email content."""
        return f"""
        <html>
        <body>
            <h2>{decision.content.title}</h2>
            <p>{decision.content.body}</p>
            {f'<img src="{decision.content.image_url}" alt="Notification Image" style="max-width: 100%;">' if decision.content.image_url else ''}
            {f'<p><a href="{decision.content.action_url}">Read More</a></p>' if decision.content.action_url else ''}
            <hr>
            <p><small>Category: {decision.content.category} | Priority: {decision.content.priority.value}</small></p>
        </body>
        </html>
        """

    def _create_email_text(self, decision: NotificationDecision) -> str:
        """Create plain text email content."""
        return f"""
        {decision.content.title}
        
        {decision.content.body}
        
        {f'Read more: {decision.content.action_url}' if decision.content.action_url else ''}
        
        Category: {decision.content.category}
        Priority: {decision.content.priority.value}
        """


class SMSClient:
    """SMS client for SMS notifications."""

    def __init__(self):
        self.settings = get_settings()
        self.client = None

    async def initialize(self) -> None:
        """Initialize SMS client."""
        logger.info("Initializing SMS client")

        if not all(
            [
                self.settings.TWILIO_ACCOUNT_SID,
                self.settings.TWILIO_AUTH_TOKEN,
                self.settings.TWILIO_PHONE_NUMBER,
            ]
        ):
            logger.warning("SMS configuration incomplete, SMS notifications disabled")
            return

        try:
            # In real implementation, would use Twilio client
            # from twilio.rest import Client
            # self.client = Client(
            #     self.settings.TWILIO_ACCOUNT_SID,
            #     self.settings.TWILIO_AUTH_TOKEN
            # )

            logger.info("SMS client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SMS client: {e}")
            raise DeliveryChannelError(f"SMS client initialization failed: {e}")

    async def cleanup(self) -> None:
        """Cleanup SMS client."""
        logger.info("Cleaning up SMS client")
        # SMS cleanup if needed

    async def send_notification(self, decision: NotificationDecision) -> DeliveryResult:
        """Send SMS notification."""

        try:
            if not self.client:
                raise DeliveryChannelError("SMS client not initialized")

            # Get user's phone number (in real implementation, fetch from database)
            user_phone = await self._get_user_phone(decision.user_id)

            if not user_phone:
                raise DeliveryChannelError(f"No phone number found for user {decision.user_id}")

            # Create SMS message
            sms_body = f"{decision.content.title}\n\n{decision.content.body}"
            if decision.content.action_url:
                sms_body += f"\n\n{decision.content.action_url}"

            # Send SMS (mock implementation)
            # message = self.client.messages.create(
            #     body=sms_body,
            #     from_=self.settings.TWILIO_PHONE_NUMBER,
            #     to=user_phone
            # )

            delivery_id = str(uuid.uuid4())

            logger.info(
                "SMS notification sent successfully",
                user_id=decision.user_id,
                delivery_id=delivery_id,
            )

            return DeliveryResult(
                success=True,
                delivery_id=delivery_id,
                channel=DeliveryChannel.SMS,
                delivered_at=datetime.utcnow(),
                was_engaged=False,
            )

        except Exception as e:
            logger.error(
                "SMS notification failed", user_id=decision.user_id, error=str(e), exc_info=True
            )
            raise DeliveryChannelError(f"SMS delivery failed: {e}")

    async def _get_user_phone(self, user_id: str) -> Optional[str]:
        """Get user's phone number."""
        # Mock implementation - would fetch from database
        return f"+1234567890"  # Mock phone number


class MultiChannelDelivery:
    """Manage delivery across multiple channels."""

    def __init__(self):
        self.fcm_client = FCMClient()
        self.email_client = EmailClient()
        self.sms_client = SMSClient()
        self.channel_performance = {}  # Cache for channel performance data

    async def initialize(self) -> None:
        """Initialize multi-channel delivery system."""
        logger.info("Initializing multi-channel delivery system")

        # Initialize all clients
        await self.fcm_client.initialize()
        await self.email_client.initialize()
        await self.sms_client.initialize()

        logger.info("Multi-channel delivery system initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup multi-channel delivery system."""
        logger.info("Cleaning up multi-channel delivery system")

        # Cleanup all clients
        await self.fcm_client.cleanup()
        await self.email_client.cleanup()
        await self.sms_client.cleanup()

    async def deliver_notification(self, decision: NotificationDecision) -> DeliveryResult:
        """Deliver notification through the specified channel."""

        try:
            logger.info(
                "Delivering notification",
                user_id=decision.user_id,
                channel=decision.delivery_channel.value,
            )

            if decision.delivery_channel == DeliveryChannel.PUSH:
                result = await self.fcm_client.send_notification(decision)
            elif decision.delivery_channel == DeliveryChannel.EMAIL:
                result = await self.email_client.send_notification(decision)
            elif decision.delivery_channel == DeliveryChannel.SMS:
                result = await self.sms_client.send_notification(decision)
            else:
                raise DeliveryChannelError(f"Unknown delivery channel: {decision.delivery_channel}")

            # Update channel performance
            await self._update_channel_performance(
                decision.user_id, decision.delivery_channel, True
            )

            return result

        except Exception as e:
            logger.error(
                "Notification delivery failed",
                user_id=decision.user_id,
                channel=decision.delivery_channel.value,
                error=str(e),
            )

            # Update channel performance
            await self._update_channel_performance(
                decision.user_id, decision.delivery_channel, False
            )

            raise

    async def select_optimal_channel(
        self, user_id: str, notification_type: NotificationType, urgency: float
    ) -> DeliveryChannel:
        """Select best delivery channel based on user behavior and context."""

        try:
            # Get user's available channels
            user_channels = await self._get_user_channels(user_id)

            if not user_channels:
                # Default to push if no channels available
                return DeliveryChannel.PUSH

            # For urgent notifications, prefer immediate channels
            if urgency > 0.8:
                if DeliveryChannel.PUSH in user_channels and await self._is_device_online(user_id):
                    return DeliveryChannel.PUSH
                elif DeliveryChannel.SMS in user_channels:
                    return DeliveryChannel.SMS
                elif DeliveryChannel.EMAIL in user_channels:
                    return DeliveryChannel.EMAIL

            # For regular notifications, use engagement-based selection
            channel_performance = await self._get_channel_performance(user_id)

            # Select channel with best performance
            available_performance = {
                channel: channel_performance.get(channel, 0.5) for channel in user_channels
            }

            optimal_channel = max(available_performance, key=available_performance.get)

            logger.debug(
                "Selected optimal channel",
                user_id=user_id,
                channel=optimal_channel.value,
                performance=available_performance,
            )

            return optimal_channel

        except Exception as e:
            logger.error(f"Error selecting optimal channel: {e}")
            # Fallback to push
            return DeliveryChannel.PUSH

    async def _get_user_channels(self, user_id: str) -> List[DeliveryChannel]:
        """Get user's available delivery channels."""
        # Mock implementation - would fetch from user preferences
        return [DeliveryChannel.PUSH, DeliveryChannel.EMAIL, DeliveryChannel.SMS]

    async def _is_device_online(self, user_id: str) -> bool:
        """Check if user's device is online."""
        # Mock implementation - would check device status
        return True

    async def _get_channel_performance(self, user_id: str) -> Dict[DeliveryChannel, float]:
        """Get channel performance scores for user."""

        # Check cache first
        cache_key = f"channel_performance:{user_id}"
        if cache_key in self.channel_performance:
            return self.channel_performance[cache_key]

        # Mock implementation - would calculate from historical data
        performance = {
            DeliveryChannel.PUSH: 0.8,
            DeliveryChannel.EMAIL: 0.6,
            DeliveryChannel.SMS: 0.9,
        }

        # Cache performance data
        self.channel_performance[cache_key] = performance

        return performance

    async def _update_channel_performance(
        self, user_id: str, channel: DeliveryChannel, success: bool
    ) -> None:
        """Update channel performance based on delivery result."""

        try:
            # In real implementation, would update performance metrics in database
            cache_key = f"channel_performance:{user_id}"

            if cache_key in self.channel_performance:
                # Adjust performance based on success/failure
                adjustment = 0.01 if success else -0.01
                self.channel_performance[cache_key][channel] = max(
                    0.0, min(1.0, self.channel_performance[cache_key][channel] + adjustment)
                )

            logger.debug(
                "Updated channel performance",
                user_id=user_id,
                channel=channel.value,
                success=success,
            )

        except Exception as e:
            logger.error(f"Error updating channel performance: {e}")

    async def get_delivery_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get delivery analytics for user."""

        try:
            analytics = {
                "user_id": user_id,
                "channel_performance": await self._get_channel_performance(user_id),
                "available_channels": await self._get_user_channels(user_id),
                "device_online": await self._is_device_online(user_id),
                "delivery_history": await self._get_delivery_history(user_id),
            }

            return analytics

        except Exception as e:
            logger.error(f"Error getting delivery analytics: {e}")
            return {}

    async def _get_delivery_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get delivery history for user."""
        # Mock implementation - would fetch from database
        return [
            {
                "channel": "push",
                "success": True,
                "delivered_at": datetime.utcnow().isoformat(),
                "engagement": True,
            }
        ]

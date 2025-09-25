"""
Notification policy engine with audit logs and escalation rules.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import structlog
from redis.asyncio import Redis

from ..models.schemas import NotificationDecision, PolicyRule, EscalationRule, AuditLog

logger = structlog.get_logger()


class PolicyEngine:
    """Manages notification policies and escalation rules."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client
        self.policies: Dict[str, PolicyRule] = {}
        self.escalation_rules: List[EscalationRule] = []
        self.audit_logs: List[AuditLog] = []

    async def initialize(self) -> None:
        """Initialize the policy engine."""
        logger.info("Initializing policy engine")
        
        # Load default policies
        await self._load_default_policies()
        
        # Load escalation rules
        await self._load_escalation_rules()
        
        logger.info("Policy engine initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up policy engine")

    async def evaluate_notification_policy(
        self, 
        decision: NotificationDecision,
        user_id: str,
        notification_type: str
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Evaluate notification against policies.
        
        Args:
            decision: Notification decision
            user_id: User ID
            notification_type: Type of notification
            
        Returns:
            Tuple of (allowed, reason, metadata)
        """
        try:
            # Check cooldown policies
            cooldown_check = await self._check_cooldown_policy(user_id, notification_type)
            if not cooldown_check[0]:
                await self._log_policy_violation(
                    user_id, notification_type, "cooldown_violation", cooldown_check[1]
                )
                return False, "cooldown_violation", cooldown_check[1]

            # Check frequency policies
            frequency_check = await self._check_frequency_policy(user_id, notification_type)
            if not frequency_check[0]:
                await self._log_policy_violation(
                    user_id, notification_type, "frequency_violation", frequency_check[1]
                )
                return False, "frequency_violation", frequency_check[1]

            # Check quiet hours
            quiet_hours_check = await self._check_quiet_hours_policy(user_id)
            if not quiet_hours_check[0]:
                await self._log_policy_violation(
                    user_id, notification_type, "quiet_hours_violation", quiet_hours_check[1]
                )
                return False, "quiet_hours_violation", quiet_hours_check[1]

            # Check escalation rules
            escalation_check = await self._check_escalation_rules(decision, user_id)
            if escalation_check[0]:
                await self._log_escalation(
                    user_id, notification_type, escalation_check[1]
                )
                return True, "escalation_triggered", escalation_check[1]

            # Check content policies
            content_check = await self._check_content_policy(decision)
            if not content_check[0]:
                await self._log_policy_violation(
                    user_id, notification_type, "content_violation", content_check[1]
                )
                return False, "content_violation", content_check[1]

            # All policies passed
            await self._log_policy_success(user_id, notification_type)
            return True, "policy_passed", {}

        except Exception as e:
            logger.error(
                "Error evaluating notification policy",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return False, "policy_evaluation_error", {"error": str(e)}

    async def _check_cooldown_policy(self, user_id: str, notification_type: str) -> Tuple[bool, Dict]:
        """Check cooldown policy for user and notification type."""
        try:
            cooldown_key = f"cooldown:{user_id}:{notification_type}"
            last_notification = await self.redis_client.get(cooldown_key)
            
            if last_notification:
                last_time = datetime.fromisoformat(last_notification.decode())
                cooldown_duration = await self._get_cooldown_duration(notification_type)
                
                if datetime.utcnow() - last_time < cooldown_duration:
                    remaining_time = cooldown_duration - (datetime.utcnow() - last_time)
                    return False, {
                        "last_notification": last_time.isoformat(),
                        "cooldown_duration": cooldown_duration.total_seconds(),
                        "remaining_time": remaining_time.total_seconds()
                    }
            
            return True, {}
            
        except Exception as e:
            logger.error("Error checking cooldown policy", user_id=user_id, error=str(e))
            return True, {}  # Allow on error

    async def _check_frequency_policy(self, user_id: str, notification_type: str) -> Tuple[bool, Dict]:
        """Check frequency policy for user and notification type."""
        try:
            frequency_key = f"frequency:{user_id}:{notification_type}"
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            hour_key = f"{frequency_key}:{current_hour.isoformat()}"
            
            # Get current count for this hour
            current_count = await self.redis_client.get(hour_key)
            count = int(current_count) if current_count else 0
            
            # Get frequency limit
            frequency_limit = await self._get_frequency_limit(notification_type)
            
            if count >= frequency_limit:
                return False, {
                    "current_count": count,
                    "frequency_limit": frequency_limit,
                    "time_window": "1_hour"
                }
            
            # Increment counter
            await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)  # 1 hour TTL
            
            return True, {"current_count": count + 1, "frequency_limit": frequency_limit}
            
        except Exception as e:
            logger.error("Error checking frequency policy", user_id=user_id, error=str(e))
            return True, {}  # Allow on error

    async def _check_quiet_hours_policy(self, user_id: str) -> Tuple[bool, Dict]:
        """Check quiet hours policy for user."""
        try:
            # Get user's quiet hours preference
            quiet_hours_key = f"quiet_hours:{user_id}"
            quiet_hours = await self.redis_client.hgetall(quiet_hours_key)
            
            if not quiet_hours:
                return True, {}  # No quiet hours set
            
            # Parse quiet hours
            start_hour = int(quiet_hours.get(b"start_hour", b"22").decode())
            end_hour = int(quiet_hours.get(b"end_hour", b"8").decode())
            timezone = quiet_hours.get(b"timezone", b"UTC").decode()
            
            current_hour = datetime.utcnow().hour
            
            # Check if current time is within quiet hours
            if start_hour > end_hour:  # Overnight quiet hours
                is_quiet = current_hour >= start_hour or current_hour < end_hour
            else:  # Same day quiet hours
                is_quiet = start_hour <= current_hour < end_hour
            
            if is_quiet:
                return False, {
                    "quiet_hours_start": start_hour,
                    "quiet_hours_end": end_hour,
                    "current_hour": current_hour,
                    "timezone": timezone
                }
            
            return True, {}
            
        except Exception as e:
            logger.error("Error checking quiet hours policy", user_id=user_id, error=str(e))
            return True, {}  # Allow on error

    async def _check_escalation_rules(self, decision: NotificationDecision, user_id: str) -> Tuple[bool, Dict]:
        """Check escalation rules."""
        try:
            for rule in self.escalation_rules:
                if await self._evaluate_escalation_rule(rule, decision, user_id):
                    return True, {
                        "rule_id": rule.rule_id,
                        "rule_name": rule.rule_name,
                        "escalation_action": rule.action,
                        "priority": rule.priority
                    }
            
            return False, {}
            
        except Exception as e:
            logger.error("Error checking escalation rules", user_id=user_id, error=str(e))
            return False, {}

    async def _check_content_policy(self, decision: NotificationDecision) -> Tuple[bool, Dict]:
        """Check content policy."""
        try:
            # Check for blocked content
            if decision.content and hasattr(decision.content, 'topics'):
                blocked_topics = await self._get_blocked_topics()
                for topic in decision.content.topics:
                    if topic in blocked_topics:
                        return False, {
                            "blocked_topic": topic,
                            "blocked_topics": blocked_topics
                        }
            
            # Check for sensitive content
            if decision.content and hasattr(decision.content, 'sentiment'):
                if decision.content.sentiment and decision.content.sentiment < -0.8:
                    return False, {
                        "reason": "negative_sentiment",
                        "sentiment_score": decision.content.sentiment
                    }
            
            return True, {}
            
        except Exception as e:
            logger.error("Error checking content policy", error=str(e))
            return True, {}  # Allow on error

    async def _load_default_policies(self) -> None:
        """Load default notification policies."""
        # Cooldown policies
        self.policies["cooldown"] = PolicyRule(
            rule_id="cooldown",
            rule_name="Notification Cooldown",
            rule_type="cooldown",
            conditions={"enabled": True},
            actions={"cooldown_duration": 300}  # 5 minutes
        )
        
        # Frequency policies
        self.policies["frequency"] = PolicyRule(
            rule_id="frequency",
            rule_name="Notification Frequency",
            rule_type="frequency",
            conditions={"enabled": True},
            actions={"max_per_hour": 5}
        )
        
        # Quiet hours policies
        self.policies["quiet_hours"] = PolicyRule(
            rule_id="quiet_hours",
            rule_name="Quiet Hours",
            rule_type="quiet_hours",
            conditions={"enabled": True},
            actions={"start_hour": 22, "end_hour": 8}
        )

    async def _load_escalation_rules(self) -> None:
        """Load escalation rules."""
        # Breaking news escalation
        self.escalation_rules.append(EscalationRule(
            rule_id="breaking_news_escalation",
            rule_name="Breaking News Escalation",
            conditions={"is_breaking": True, "urgency_score": 0.8},
            action="immediate_delivery",
            priority="high"
        ))
        
        # High priority escalation
        self.escalation_rules.append(EscalationRule(
            rule_id="high_priority_escalation",
            rule_name="High Priority Escalation",
            conditions={"priority": "urgent", "relevance_score": 0.9},
            action="escalate_to_manager",
            priority="medium"
        ))

    async def _get_cooldown_duration(self, notification_type: str) -> timedelta:
        """Get cooldown duration for notification type."""
        cooldown_durations = {
            "breaking_news": timedelta(minutes=5),
            "urgent": timedelta(minutes=10),
            "normal": timedelta(minutes=30),
            "low_priority": timedelta(hours=1)
        }
        return cooldown_durations.get(notification_type, timedelta(minutes=30))

    async def _get_frequency_limit(self, notification_type: str) -> int:
        """Get frequency limit for notification type."""
        frequency_limits = {
            "breaking_news": 10,
            "urgent": 5,
            "normal": 3,
            "low_priority": 1
        }
        return frequency_limits.get(notification_type, 3)

    async def _get_blocked_topics(self) -> List[str]:
        """Get list of blocked topics."""
        try:
            blocked_topics = await self.redis_client.smembers("blocked_topics")
            return [topic.decode() for topic in blocked_topics]
        except Exception as e:
            logger.error("Error getting blocked topics", error=str(e))
            return []

    async def _evaluate_escalation_rule(
        self, 
        rule: EscalationRule, 
        decision: NotificationDecision, 
        user_id: str
    ) -> bool:
        """Evaluate if escalation rule should trigger."""
        try:
            # Check conditions
            for condition, value in rule.conditions.items():
                if condition == "is_breaking":
                    if not getattr(decision, "is_breaking", False):
                        return False
                elif condition == "urgency_score":
                    if decision.score < value:
                        return False
                elif condition == "priority":
                    if decision.priority.value != value:
                        return False
                elif condition == "relevance_score":
                    if decision.score < value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error("Error evaluating escalation rule", rule_id=rule.rule_id, error=str(e))
            return False

    async def _log_policy_violation(
        self, 
        user_id: str, 
        notification_type: str, 
        violation_type: str, 
        metadata: Dict
    ) -> None:
        """Log policy violation."""
        audit_log = AuditLog(
            log_id=f"policy_violation_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            action="policy_violation",
            resource_type="notification",
            resource_id=f"{user_id}:{notification_type}",
            details={
                "violation_type": violation_type,
                "notification_type": notification_type,
                "metadata": metadata
            },
            timestamp=datetime.utcnow()
        )
        
        await self._store_audit_log(audit_log)

    async def _log_policy_success(self, user_id: str, notification_type: str) -> None:
        """Log successful policy evaluation."""
        audit_log = AuditLog(
            log_id=f"policy_success_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            action="policy_success",
            resource_type="notification",
            resource_id=f"{user_id}:{notification_type}",
            details={"notification_type": notification_type},
            timestamp=datetime.utcnow()
        )
        
        await self._store_audit_log(audit_log)

    async def _log_escalation(
        self, 
        user_id: str, 
        notification_type: str, 
        escalation_data: Dict
    ) -> None:
        """Log escalation event."""
        audit_log = AuditLog(
            log_id=f"escalation_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            action="escalation_triggered",
            resource_type="notification",
            resource_id=f"{user_id}:{notification_type}",
            details={
                "notification_type": notification_type,
                "escalation_data": escalation_data
            },
            timestamp=datetime.utcnow()
        )
        
        await self._store_audit_log(audit_log)

    async def _store_audit_log(self, audit_log: AuditLog) -> None:
        """Store audit log in Redis."""
        try:
            log_key = f"audit_log:{audit_log.log_id}"
            log_data = {
                "log_id": audit_log.log_id,
                "user_id": audit_log.user_id,
                "action": audit_log.action,
                "resource_type": audit_log.resource_type,
                "resource_id": audit_log.resource_id,
                "details": str(audit_log.details),
                "timestamp": audit_log.timestamp.isoformat()
            }
            
            await self.redis_client.hset(log_key, mapping=log_data)
            await self.redis_client.expire(log_key, 2592000)  # 30 days TTL
            
            # Add to audit log list
            await self.redis_client.lpush("audit_logs", audit_log.log_id)
            await self.redis_client.ltrim("audit_logs", 0, 9999)  # Keep last 10000 logs
            
        except Exception as e:
            logger.error("Error storing audit log", error=str(e))

    async def get_audit_logs(
        self, 
        user_id: Optional[str] = None, 
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs with optional filtering."""
        try:
            log_ids = await self.redis_client.lrange("audit_logs", 0, limit - 1)
            logs = []
            
            for log_id in log_ids:
                log_key = f"audit_log:{log_id.decode()}"
                log_data = await self.redis_client.hgetall(log_key)
                
                if log_data:
                    # Apply filters
                    if user_id and log_data.get(b"user_id").decode() != user_id:
                        continue
                    if action and log_data.get(b"action").decode() != action:
                        continue
                    
                    audit_log = AuditLog(
                        log_id=log_data[b"log_id"].decode(),
                        user_id=log_data[b"user_id"].decode(),
                        action=log_data[b"action"].decode(),
                        resource_type=log_data[b"resource_type"].decode(),
                        resource_id=log_data[b"resource_id"].decode(),
                        details=eval(log_data[b"details"].decode()) if log_data.get(b"details") else {},
                        timestamp=datetime.fromisoformat(log_data[b"timestamp"].decode())
                    )
                    logs.append(audit_log)
            
            return logs
            
        except Exception as e:
            logger.error("Error getting audit logs", error=str(e))
            return []

"""
Abuse Rule Engine
Rule-based abuse detection and prevention
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import re
from dataclasses import dataclass

from safety_engine.models import RuleViolation, ThreatLevel
from safety_engine.config import get_abuse_config

logger = logging.getLogger(__name__)

@dataclass
class AbuseRule:
    """Definition of an abuse rule"""
    rule_id: str
    rule_name: str
    description: str
    severity: ThreatLevel
    enabled: bool = True
    condition: Callable = None
    parameters: Dict[str, Any] = None

class AbuseRuleEngine:
    """Rule-based abuse detection engine"""
    
    def __init__(self):
        self.config = get_abuse_config()
        self.is_initialized = False
        
        # Rules storage
        self.rules: Dict[str, AbuseRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        
        # Rule execution statistics
        self.rule_stats = {}
        
    async def initialize(self):
        """Initialize the rule engine"""
        try:
            # Load predefined rules
            await self.load_predefined_rules()
            
            # Load custom rules from configuration
            await self.load_custom_rules()
            
            self.is_initialized = True
            logger.info(f"Abuse rule engine initialized with {len(self.rules)} rules")
            
        except Exception as e:
            logger.error(f"Failed to initialize rule engine: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.rules.clear()
            self.rule_groups.clear()
            self.rule_stats.clear()
            
            self.is_initialized = False
            logger.info("Rule engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during rule engine cleanup: {str(e)}")
    
    async def check_abuse_rules(self, user_id: str, activity_data: Any) -> List[RuleViolation]:
        """Check all applicable abuse rules"""
        
        if not self.is_initialized:
            raise RuntimeError("Rule engine not initialized")
        
        try:
            violations = []
            
            # Check each enabled rule
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Execute rule condition
                    is_violated, violation_details = await self.execute_rule(rule, user_id, activity_data)
                    
                    if is_violated:
                        violation = RuleViolation(
                            rule_id=rule.rule_id,
                            rule_name=rule.rule_name,
                            violation_type=rule.rule_name,
                            severity=rule.severity,
                            description=f"{rule.description}: {violation_details}",
                            timestamp=datetime.utcnow()
                        )
                        violations.append(violation)
                        
                        # Update rule statistics
                        self.update_rule_stats(rule_id, True)
                    else:
                        self.update_rule_stats(rule_id, False)
                        
                except Exception as e:
                    logger.error(f"Rule {rule_id} execution failed: {str(e)}")
                    self.update_rule_stats(rule_id, False, error=True)
            
            return violations
            
        except Exception as e:
            logger.error(f"Rule checking failed: {str(e)}")
            return []
    
    async def execute_rule(self, rule: AbuseRule, user_id: str, activity_data: Any) -> tuple[bool, str]:
        """Execute a specific rule and return violation status"""
        try:
            if rule.condition is None:
                return False, "No condition defined"
            
            # Prepare rule context
            context = {
                'user_id': user_id,
                'activity_data': activity_data,
                'rule_parameters': rule.parameters or {},
                'timestamp': datetime.utcnow()
            }
            
            # Execute rule condition
            result = rule.condition(context)
            
            if isinstance(result, bool):
                return result, "Rule condition triggered" if result else "Rule condition not triggered"
            elif isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1]
            else:
                return False, "Invalid rule result"
                
        except Exception as e:
            logger.error(f"Rule execution failed for {rule.rule_id}: {str(e)}")
            return False, f"Rule execution error: {str(e)}"
    
    async def load_predefined_rules(self):
        """Load predefined abuse detection rules"""
        try:
            # High frequency requests rule
            self.add_rule(AbuseRule(
                rule_id="high_frequency_requests",
                rule_name="High Frequency Requests",
                description="User making too many requests in a short time",
                severity=ThreatLevel.MEDIUM,
                condition=self.check_high_frequency_requests,
                parameters={"max_requests_per_minute": 60, "max_requests_per_hour": 1000}
            ))
            
            # Unusual request patterns rule
            self.add_rule(AbuseRule(
                rule_id="unusual_request_patterns",
                rule_name="Unusual Request Patterns",
                description="User exhibiting unusual request patterns",
                severity=ThreatLevel.LOW,
                condition=self.check_unusual_request_patterns,
                parameters={"pattern_threshold": 0.8}
            ))
            
            # Suspicious user agent rule
            self.add_rule(AbuseRule(
                rule_id="suspicious_user_agent",
                rule_name="Suspicious User Agent",
                description="User using suspicious or automated user agent",
                severity=ThreatLevel.MEDIUM,
                condition=self.check_suspicious_user_agent,
                parameters={"suspicious_patterns": ["bot", "crawler", "scraper", "automated"]}
            ))
            
            # Geographic anomalies rule
            self.add_rule(AbuseRule(
                rule_id="geographic_anomalies",
                rule_name="Geographic Anomalies",
                description="User accessing from unusual geographic locations",
                severity=ThreatLevel.MEDIUM,
                condition=self.check_geographic_anomalies,
                parameters={"max_countries_per_hour": 3, "suspicious_countries": []}
            ))
            
            # Account age violations rule
            self.add_rule(AbuseRule(
                rule_id="account_age_violations",
                rule_name="Account Age Violations",
                description="New account exhibiting suspicious behavior",
                severity=ThreatLevel.HIGH,
                condition=self.check_account_age_violations,
                parameters={"min_account_age_hours": 24, "max_activity_for_new_account": 100}
            ))
            
            # Content violations rule
            self.add_rule(AbuseRule(
                rule_id="content_violations",
                rule_name="Content Violations",
                description="User posting or accessing prohibited content",
                severity=ThreatLevel.HIGH,
                condition=self.check_content_violations,
                parameters={"max_violations_per_hour": 5}
            ))
            
            # Authentication failures rule
            self.add_rule(AbuseRule(
                rule_id="authentication_failures",
                rule_name="Authentication Failures",
                description="Multiple failed authentication attempts",
                severity=ThreatLevel.MEDIUM,
                condition=self.check_authentication_failures,
                parameters={"max_failed_attempts": 5, "time_window_minutes": 15}
            ))
            
            # Resource abuse rule
            self.add_rule(AbuseRule(
                rule_id="resource_abuse",
                rule_name="Resource Abuse",
                description="User consuming excessive system resources",
                severity=ThreatLevel.MEDIUM,
                condition=self.check_resource_abuse,
                parameters={"max_cpu_usage": 0.8, "max_memory_usage": 0.8}
            ))
            
            logger.info("Predefined rules loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load predefined rules: {str(e)}")
            raise
    
    async def load_custom_rules(self):
        """Load custom rules from configuration"""
        try:
            # This would load custom rules from a configuration file or database
            # For now, we'll add some example custom rules
            
            # Custom rule: API key abuse
            self.add_rule(AbuseRule(
                rule_id="api_key_abuse",
                rule_name="API Key Abuse",
                description="Suspicious API key usage patterns",
                severity=ThreatLevel.HIGH,
                condition=self.check_api_key_abuse,
                parameters={"max_requests_per_key_per_hour": 10000}
            ))
            
            # Custom rule: Data exfiltration
            self.add_rule(AbuseRule(
                rule_id="data_exfiltration",
                rule_name="Data Exfiltration",
                description="Suspicious data access patterns",
                severity=ThreatLevel.CRITICAL,
                condition=self.check_data_exfiltration,
                parameters={"max_data_access_per_hour": 1000}
            ))
            
            logger.info("Custom rules loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load custom rules: {str(e)}")
    
    def add_rule(self, rule: AbuseRule):
        """Add a rule to the engine"""
        try:
            self.rules[rule.rule_id] = rule
            
            # Add to rule groups
            if rule.rule_id not in self.rule_groups:
                self.rule_groups[rule.rule_id] = []
            
            # Initialize rule statistics
            if rule.rule_id not in self.rule_stats:
                self.rule_stats[rule.rule_id] = {
                    'total_executions': 0,
                    'violations': 0,
                    'errors': 0,
                    'last_execution': None
                }
            
        except Exception as e:
            logger.error(f"Failed to add rule {rule.rule_id}: {str(e)}")
    
    def remove_rule(self, rule_id: str):
        """Remove a rule from the engine"""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                if rule_id in self.rule_groups:
                    del self.rule_groups[rule_id]
                if rule_id in self.rule_stats:
                    del self.rule_stats[rule_id]
                
                logger.info(f"Rule {rule_id} removed successfully")
            
        except Exception as e:
            logger.error(f"Failed to remove rule {rule_id}: {str(e)}")
    
    def update_rule_stats(self, rule_id: str, violated: bool, error: bool = False):
        """Update rule execution statistics"""
        try:
            if rule_id not in self.rule_stats:
                self.rule_stats[rule_id] = {
                    'total_executions': 0,
                    'violations': 0,
                    'errors': 0,
                    'last_execution': None
                }
            
            stats = self.rule_stats[rule_id]
            stats['total_executions'] += 1
            stats['last_execution'] = datetime.utcnow()
            
            if error:
                stats['errors'] += 1
            elif violated:
                stats['violations'] += 1
                
        except Exception as e:
            logger.error(f"Failed to update rule stats for {rule_id}: {str(e)}")
    
    # Rule condition functions
    def check_high_frequency_requests(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for high frequency requests"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            # Get request frequency from activity data
            request_frequency = getattr(activity_data, 'request_frequency', 0)
            max_requests_per_minute = params.get('max_requests_per_minute', 60)
            
            if request_frequency > max_requests_per_minute:
                return True, f"Request frequency {request_frequency} exceeds limit {max_requests_per_minute}"
            
            return False, "Request frequency within limits"
            
        except Exception as e:
            logger.error(f"High frequency check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_unusual_request_patterns(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for unusual request patterns"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            # Get pattern score from activity data
            pattern_score = getattr(activity_data, 'pattern_anomaly_score', 0)
            threshold = params.get('pattern_threshold', 0.8)
            
            if pattern_score > threshold:
                return True, f"Pattern anomaly score {pattern_score} exceeds threshold {threshold}"
            
            return False, "Request patterns appear normal"
            
        except Exception as e:
            logger.error(f"Unusual pattern check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_suspicious_user_agent(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for suspicious user agent"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            user_agent = getattr(activity_data, 'user_agent', '')
            suspicious_patterns = params.get('suspicious_patterns', [])
            
            if user_agent:
                user_agent_lower = user_agent.lower()
                for pattern in suspicious_patterns:
                    if pattern.lower() in user_agent_lower:
                        return True, f"User agent '{user_agent}' matches suspicious pattern '{pattern}'"
            
            return False, "User agent appears legitimate"
            
        except Exception as e:
            logger.error(f"Suspicious user agent check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_geographic_anomalies(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for geographic anomalies"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            # Get geographic data from activity data
            countries = getattr(activity_data, 'countries_accessed', [])
            max_countries = params.get('max_countries_per_hour', 3)
            
            if len(countries) > max_countries:
                return True, f"Accessed {len(countries)} countries, exceeds limit {max_countries}"
            
            return False, "Geographic access patterns appear normal"
            
        except Exception as e:
            logger.error(f"Geographic anomaly check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_account_age_violations(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for account age violations"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            account_age_hours = getattr(activity_data, 'account_age_hours', 0)
            activity_level = getattr(activity_data, 'activity_level', 0)
            
            min_age = params.get('min_account_age_hours', 24)
            max_activity = params.get('max_activity_for_new_account', 100)
            
            if account_age_hours < min_age and activity_level > max_activity:
                return True, f"New account ({account_age_hours}h) with high activity ({activity_level})"
            
            return False, "Account age and activity levels are appropriate"
            
        except Exception as e:
            logger.error(f"Account age violation check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_content_violations(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for content violations"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            violations_count = getattr(activity_data, 'content_violations_count', 0)
            max_violations = params.get('max_violations_per_hour', 5)
            
            if violations_count > max_violations:
                return True, f"Content violations count {violations_count} exceeds limit {max_violations}"
            
            return False, "Content violations within acceptable limits"
            
        except Exception as e:
            logger.error(f"Content violation check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_authentication_failures(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for authentication failures"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            failed_attempts = getattr(activity_data, 'failed_auth_attempts', 0)
            max_attempts = params.get('max_failed_attempts', 5)
            
            if failed_attempts > max_attempts:
                return True, f"Failed authentication attempts {failed_attempts} exceeds limit {max_attempts}"
            
            return False, "Authentication failures within acceptable limits"
            
        except Exception as e:
            logger.error(f"Authentication failure check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_resource_abuse(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for resource abuse"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            cpu_usage = getattr(activity_data, 'cpu_usage', 0)
            memory_usage = getattr(activity_data, 'memory_usage', 0)
            
            max_cpu = params.get('max_cpu_usage', 0.8)
            max_memory = params.get('max_memory_usage', 0.8)
            
            if cpu_usage > max_cpu:
                return True, f"CPU usage {cpu_usage} exceeds limit {max_cpu}"
            
            if memory_usage > max_memory:
                return True, f"Memory usage {memory_usage} exceeds limit {max_memory}"
            
            return False, "Resource usage within acceptable limits"
            
        except Exception as e:
            logger.error(f"Resource abuse check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_api_key_abuse(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for API key abuse"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            api_requests = getattr(activity_data, 'api_requests_count', 0)
            max_requests = params.get('max_requests_per_key_per_hour', 10000)
            
            if api_requests > max_requests:
                return True, f"API requests {api_requests} exceeds limit {max_requests}"
            
            return False, "API usage within acceptable limits"
            
        except Exception as e:
            logger.error(f"API key abuse check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    def check_data_exfiltration(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check for data exfiltration patterns"""
        try:
            activity_data = context['activity_data']
            params = context['rule_parameters']
            
            data_access = getattr(activity_data, 'data_access_count', 0)
            max_access = params.get('max_data_access_per_hour', 1000)
            
            if data_access > max_access:
                return True, f"Data access {data_access} exceeds limit {max_access}"
            
            return False, "Data access patterns appear normal"
            
        except Exception as e:
            logger.error(f"Data exfiltration check failed: {str(e)}")
            return False, f"Check failed: {str(e)}"
    
    async def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule execution statistics"""
        try:
            return {
                'total_rules': len(self.rules),
                'enabled_rules': sum(1 for rule in self.rules.values() if rule.enabled),
                'rule_stats': self.rule_stats,
                'rule_groups': self.rule_groups
            }
            
        except Exception as e:
            logger.error(f"Rule statistics retrieval failed: {str(e)}")
            return {}

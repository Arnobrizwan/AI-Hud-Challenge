"""
Audit Trail Manager
Comprehensive audit logging and reporting
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass, asdict

from safety_engine.config import get_compliance_config

logger = logging.getLogger(__name__)

@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    result: str
    severity: str

class AuditTrailManager:
    """Comprehensive audit logging and reporting system"""
    
    def __init__(self):
        self.config = get_compliance_config()
        self.is_initialized = False
        
        # Audit event storage
        self.audit_events: List[AuditEvent] = []
        self.event_counter = 0
        
        # Event types and their configurations
        self.event_types = {
            'safety_check': {
                'retention_days': 365,
                'severity': 'info',
                'description': 'Safety monitoring check performed'
            },
            'abuse_detection': {
                'retention_days': 365,
                'severity': 'warning',
                'description': 'Abuse detection triggered'
            },
            'content_moderation': {
                'retention_days': 365,
                'severity': 'info',
                'description': 'Content moderation performed'
            },
            'rate_limiting': {
                'retention_days': 90,
                'severity': 'info',
                'description': 'Rate limiting applied'
            },
            'compliance_check': {
                'retention_days': 2555,  # 7 years
                'severity': 'info',
                'description': 'Compliance check performed'
            },
            'data_access': {
                'retention_days': 2555,  # 7 years
                'severity': 'info',
                'description': 'Data access event'
            },
            'data_modification': {
                'retention_days': 2555,  # 7 years
                'severity': 'warning',
                'description': 'Data modification event'
            },
            'system_event': {
                'retention_days': 90,
                'severity': 'info',
                'description': 'System event occurred'
            },
            'security_event': {
                'retention_days': 2555,  # 7 years
                'severity': 'critical',
                'description': 'Security event occurred'
            }
        }
        
    async def initialize(self):
        """Initialize the audit trail manager"""
        try:
            # Start cleanup task
            asyncio.create_task(self.cleanup_old_events())
            
            self.is_initialized = True
            logger.info("Audit trail manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit trail manager: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.audit_events.clear()
            self.is_initialized = False
            logger.info("Audit trail manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during audit trail manager cleanup: {str(e)}")
    
    async def log_event(self, event_type: str, action: str, resource: str,
                       user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       result: str = 'success', severity: Optional[str] = None) -> str:
        """Log an audit event"""
        
        if not self.is_initialized:
            raise RuntimeError("Audit trail manager not initialized")
        
        try:
            # Generate event ID
            self.event_counter += 1
            event_id = f"audit_{self.event_counter}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine severity
            if severity is None:
                severity = self.event_types.get(event_type, {}).get('severity', 'info')
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                user_id=user_id,
                action=action,
                resource=resource,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                result=result,
                severity=severity
            )
            
            # Store event
            self.audit_events.append(audit_event)
            
            # Log to system logger
            logger.info(f"Audit event logged: {event_id} - {event_type}:{action}")
            
            return event_id
            
        except Exception as e:
            logger.error(f"Audit event logging failed: {str(e)}")
            raise
    
    async def log_safety_check(self, request: Any, result: Any) -> str:
        """Log safety check event"""
        try:
            return await self.log_event(
                event_type='safety_check',
                action='monitor_system_safety',
                resource='safety_service',
                user_id=getattr(request, 'user_id', None),
                details={
                    'request_id': getattr(request, 'request_id', None),
                    'overall_score': getattr(result, 'overall_score', None),
                    'requires_intervention': getattr(result, 'requires_intervention', None)
                },
                result='success' if getattr(result, 'overall_score', 0) > 0.8 else 'warning'
            )
            
        except Exception as e:
            logger.error(f"Safety check logging failed: {str(e)}")
            return ""
    
    async def log_abuse_detection(self, user_id: str, abuse_result: Any) -> str:
        """Log abuse detection event"""
        try:
            return await self.log_event(
                event_type='abuse_detection',
                action='detect_abuse',
                resource='abuse_detection',
                user_id=user_id,
                details={
                    'abuse_score': getattr(abuse_result, 'abuse_score', None),
                    'threat_level': getattr(abuse_result, 'threat_level', None),
                    'rule_violations': len(getattr(abuse_result, 'rule_violations', []))
                },
                result='success' if getattr(abuse_result, 'abuse_score', 0) < 0.7 else 'warning',
                severity='warning' if getattr(abuse_result, 'abuse_score', 0) > 0.7 else 'info'
            )
            
        except Exception as e:
            logger.error(f"Abuse detection logging failed: {str(e)}")
            return ""
    
    async def log_content_moderation(self, content_id: str, moderation_result: Any) -> str:
        """Log content moderation event"""
        try:
            return await self.log_event(
                event_type='content_moderation',
                action='moderate_content',
                resource='content_moderation',
                details={
                    'content_id': content_id,
                    'safety_score': getattr(moderation_result, 'overall_safety_score', None),
                    'recommended_action': getattr(moderation_result, 'recommended_action', None),
                    'violations': len(getattr(moderation_result, 'violations', []))
                },
                result='success' if getattr(moderation_result, 'overall_safety_score', 0) > 0.8 else 'warning'
            )
            
        except Exception as e:
            logger.error(f"Content moderation logging failed: {str(e)}")
            return ""
    
    async def log_rate_limiting(self, user_id: str, rate_limit_result: Any) -> str:
        """Log rate limiting event"""
        try:
            return await self.log_event(
                event_type='rate_limiting',
                action='check_rate_limits',
                resource='rate_limiting',
                user_id=user_id,
                details={
                    'is_rate_limited': getattr(rate_limit_result, 'is_rate_limited', None),
                    'triggered_limits': getattr(rate_limit_result, 'triggered_limits', []),
                    'remaining_capacity': getattr(rate_limit_result, 'remaining_capacity', None)
                },
                result='success' if not getattr(rate_limit_result, 'is_rate_limited', False) else 'warning'
            )
            
        except Exception as e:
            logger.error(f"Rate limiting logging failed: {str(e)}")
            return ""
    
    async def log_compliance_check(self, request: Any, report: Any) -> str:
        """Log compliance check event"""
        try:
            return await self.log_event(
                event_type='compliance_check',
                action='check_compliance',
                resource='compliance_monitor',
                details={
                    'compliance_score': getattr(report, 'overall_compliance_score', None),
                    'violations': len(getattr(report, 'violations', [])),
                    'recommendations': len(getattr(report, 'recommendations', []))
                },
                result='success' if getattr(report, 'overall_compliance_score', 0) > 0.8 else 'warning'
            )
            
        except Exception as e:
            logger.error(f"Compliance check logging failed: {str(e)}")
            return ""
    
    async def log_data_access(self, user_id: str, resource: str, action: str, details: Dict[str, Any]) -> str:
        """Log data access event"""
        try:
            return await self.log_event(
                event_type='data_access',
                action=action,
                resource=resource,
                user_id=user_id,
                details=details,
                result='success'
            )
            
        except Exception as e:
            logger.error(f"Data access logging failed: {str(e)}")
            return ""
    
    async def log_data_modification(self, user_id: str, resource: str, action: str, details: Dict[str, Any]) -> str:
        """Log data modification event"""
        try:
            return await self.log_event(
                event_type='data_modification',
                action=action,
                resource=resource,
                user_id=user_id,
                details=details,
                result='success',
                severity='warning'
            )
            
        except Exception as e:
            logger.error(f"Data modification logging failed: {str(e)}")
            return ""
    
    async def log_security_event(self, event_type: str, action: str, resource: str,
                                user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                                ip_address: Optional[str] = None) -> str:
        """Log security event"""
        try:
            return await self.log_event(
                event_type='security_event',
                action=action,
                resource=resource,
                user_id=user_id,
                details=details,
                ip_address=ip_address,
                result='warning',
                severity='critical'
            )
            
        except Exception as e:
            logger.error(f"Security event logging failed: {str(e)}")
            return ""
    
    async def get_audit_logs(self, limit: int = 100, offset: int = 0,
                           event_type: Optional[str] = None, user_id: Optional[str] = None,
                           severity: Optional[str] = None, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit logs with filtering"""
        try:
            # Filter events
            filtered_events = self.audit_events
            
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
            if user_id:
                filtered_events = [e for e in filtered_events if e.user_id == user_id]
            
            if severity:
                filtered_events = [e for e in filtered_events if e.severity == severity]
            
            if start_date:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
            
            if end_date:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            paginated_events = filtered_events[offset:offset + limit]
            
            # Convert to dictionaries
            return [asdict(event) for event in paginated_events]
            
        except Exception as e:
            logger.error(f"Audit logs retrieval failed: {str(e)}")
            return []
    
    async def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary"""
        try:
            if not self.audit_events:
                return {"message": "No audit events available"}
            
            # Calculate summary statistics
            total_events = len(self.audit_events)
            
            # Count by event type
            event_type_counts = {}
            for event in self.audit_events:
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
            
            # Count by severity
            severity_counts = {}
            for event in self.audit_events:
                severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # Count by result
            result_counts = {}
            for event in self.audit_events:
                result_counts[event.result] = result_counts.get(event.result, 0) + 1
            
            # Get recent events (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_events = [e for e in self.audit_events if e.timestamp > cutoff_time]
            
            return {
                'total_events': total_events,
                'recent_events_24h': len(recent_events),
                'event_type_counts': event_type_counts,
                'severity_counts': severity_counts,
                'result_counts': result_counts,
                'oldest_event': min(e.timestamp for e in self.audit_events).isoformat(),
                'newest_event': max(e.timestamp for e in self.audit_events).isoformat(),
                'retention_days': self.config.audit_log_retention_days
            }
            
        except Exception as e:
            logger.error(f"Audit summary generation failed: {str(e)}")
            return {'error': str(e)}
    
    async def export_audit_logs(self, format: str = 'json', start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> str:
        """Export audit logs in specified format"""
        try:
            # Get filtered events
            events = await self.get_audit_logs(
                limit=10000,  # Large limit for export
                start_date=start_date,
                end_date=end_date
            )
            
            if format == 'json':
                return json.dumps(events, indent=2, default=str)
            elif format == 'csv':
                return self.generate_csv_export(events)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Audit logs export failed: {str(e)}")
            return ""
    
    def generate_csv_export(self, events: List[Dict[str, Any]]) -> str:
        """Generate CSV export of audit logs"""
        try:
            if not events:
                return "event_id,event_type,timestamp,user_id,action,resource,result,severity\n"
            
            # Get all unique keys
            all_keys = set()
            for event in events:
                all_keys.update(event.keys())
            
            # Create CSV header
            csv_lines = [','.join(sorted(all_keys))]
            
            # Add data rows
            for event in events:
                row = []
                for key in sorted(all_keys):
                    value = event.get(key, '')
                    # Escape commas and quotes
                    if isinstance(value, str):
                        value = value.replace('"', '""')
                        if ',' in value or '"' in value:
                            value = f'"{value}"'
                    row.append(str(value))
                csv_lines.append(','.join(row))
            
            return '\n'.join(csv_lines)
            
        except Exception as e:
            logger.error(f"CSV export generation failed: {str(e)}")
            return ""
    
    async def cleanup_old_events(self):
        """Background task to clean up old audit events"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if not self.is_initialized:
                    break
                
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=self.config.audit_log_retention_days)
                
                # Remove old events
                initial_count = len(self.audit_events)
                self.audit_events = [e for e in self.audit_events if e.timestamp > cutoff_date]
                removed_count = initial_count - len(self.audit_events)
                
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} old audit events")
                
            except Exception as e:
                logger.error(f"Audit cleanup task failed: {str(e)}")
                await asyncio.sleep(3600)
    
    async def search_audit_logs(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search audit logs by query"""
        try:
            # Simple text search in event details
            matching_events = []
            
            for event in self.audit_events:
                # Search in various fields
                searchable_text = f"{event.event_type} {event.action} {event.resource} {event.result} {event.severity}"
                if event.user_id:
                    searchable_text += f" {event.user_id}"
                if event.details:
                    searchable_text += f" {json.dumps(event.details)}"
                
                if query.lower() in searchable_text.lower():
                    matching_events.append(asdict(event))
            
            # Sort by timestamp (newest first)
            matching_events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return matching_events[:limit]
            
        except Exception as e:
            logger.error(f"Audit search failed: {str(e)}")
            return []
    
    async def get_user_activity(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get activity for a specific user"""
        try:
            user_events = [e for e in self.audit_events if e.user_id == user_id]
            user_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(event) for event in user_events[:limit]]
            
        except Exception as e:
            logger.error(f"User activity retrieval failed: {str(e)}")
            return []
    
    async def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security-related events"""
        try:
            security_events = [e for e in self.audit_events if e.event_type == 'security_event' or e.severity == 'critical']
            security_events.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(event) for event in security_events[:limit]]
            
        except Exception as e:
            logger.error(f"Security events retrieval failed: {str(e)}")
            return []

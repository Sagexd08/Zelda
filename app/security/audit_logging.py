"""
Comprehensive audit logging for security events
"""
import logging
from datetime import datetime
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Audit logging for security and compliance
    """
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
    
    def log_authentication_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: str,
        user_agent: Optional[str] = None,
        additional_data: Optional[Dict] = None
    ):
        """Log authentication attempt"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'authentication_attempt',
            'user_id': user_id,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent,
            **(additional_data or {})
        }
        self._log_event(event)
    
    def log_user_registration(
        self,
        user_id: str,
        ip_address: str,
        source: str = "web"
    ):
        """Log user registration"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'user_registration',
            'user_id': user_id,
            'ip_address': ip_address,
            'source': source
        }
        self._log_event(event)
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str
    ):
        """Log data access"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'data_access',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'ip_address': ip_address
        }
        self._log_event(event)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        details: Optional[Dict] = None
    ):
        """Log security event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'details': details or {}
        }
        self._log_event(event, log_level=logging.WARNING if severity == 'high' else logging.INFO)
    
    def log_gdpr_deletion(
        self,
        user_id: str,
        requested_by: str,
        ip_address: str
    ):
        """Log GDPR data deletion"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'gdpr_deletion',
            'user_id': user_id,
            'requested_by': requested_by,
            'ip_address': ip_address
        }
        self._log_event(event)
    
    def _log_event(self, event: Dict[str, Any], log_level: int = logging.INFO):
        """Internal method to log event"""
        message = json.dumps(event)
        self.logger.log(log_level, message)


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


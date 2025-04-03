from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """Alert information"""
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

class AlertNotifier(ABC):
    """Alert notification interface"""
    
    @abstractmethod
    def notify(self, alert: Alert) -> None:
        """Send alert"""
        pass

class ConsoleAlertNotifier(AlertNotifier):
    """Console alert notifier"""
    
    def notify(self, alert: Alert) -> None:
        """Print alert information to console"""
        print(f"[{alert.severity.value}] {alert.title}")
        print(f"Source: {alert.source}")
        print(f"Time: {alert.timestamp}")
        print(f"Message: {alert.message}")
        if alert.metadata:
            print("Metadata:", alert.metadata)
        print("-" * 80)

class AlertManager:
    """Alert manager"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.notifiers: List[AlertNotifier] = []
        self.alerts: List[Alert] = []
    
    def add_notifier(self, notifier: AlertNotifier) -> None:
        """Add alert notifier"""
        self.notifiers.append(notifier)
    
    def trigger_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Trigger alert"""
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        for notifier in self.notifiers:
            notifier.notify(alert)
    
    def info(
        self,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record info level alert"""
        self.trigger_alert(
            title=title,
            message=message,
            severity=AlertSeverity.INFO,
            source=source,
            metadata=metadata
        )
    
    def warning(
        self,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record warning level alert"""
        self.trigger_alert(
            title=title,
            message=message,
            severity=AlertSeverity.WARNING,
            source=source,
            metadata=metadata
        )
    
    def error(
        self,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record error level alert"""
        self.trigger_alert(
            title=title,
            message=message,
            severity=AlertSeverity.ERROR,
            source=source,
            metadata=metadata
        )
    
    def critical(
        self,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record critical level alert"""
        self.trigger_alert(
            title=title,
            message=message,
            severity=AlertSeverity.CRITICAL,
            source=source,
            metadata=metadata
        )
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.alerts

    def get_alerts(self, 
                  severity: Optional[AlertSeverity] = None,
                  source: Optional[str] = None) -> List[Alert]:
        """Get alert history"""
        alerts = []
        for notifier in self.notifiers:
            alerts.extend(notifier.get_alerts())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        return alerts 
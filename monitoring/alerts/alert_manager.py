from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    """告警信息"""
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    
class AlertNotifier(ABC):
    """告警通知接口"""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        pass

class ConsoleAlertNotifier(AlertNotifier):
    """控制台告警通知器"""
    
    def send_alert(self, alert: Alert) -> bool:
        """在控制台打印告警信息"""
        print(f"\n[ALERT] {alert.severity.value} - {alert.title}")
        print(f"Source: {alert.source}")
        print(f"Message: {alert.message}")
        if alert.metadata:
            print("Metadata:", alert.metadata)
        print(f"Time: {alert.timestamp}\n")
        return True

class AlertManager:
    """告警管理器"""
    
    def __init__(self, notifiers: Optional[List[AlertNotifier]] = None):
        """初始化告警管理器"""
        self.notifiers = notifiers or []
    
    def trigger_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """触发告警"""
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        for notifier in self.notifiers:
            try:
                notifier.send_alert(alert)
            except Exception as e:
                print(f"Failed to send alert via {notifier.__class__.__name__}: {e}")

    def info(self, source: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """记录信息级别告警"""
        self.trigger_alert(
            title="Information",
            message=message,
            severity=AlertSeverity.INFO,
            source=source,
            metadata=metadata
        )
    
    def warning(self, source: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """记录警告级别告警"""
        self.trigger_alert(
            title="Warning",
            message=message,
            severity=AlertSeverity.WARNING,
            source=source,
            metadata=metadata
        )
    
    def error(self, source: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """记录错误级别告警"""
        self.trigger_alert(
            title="Error",
            message=message,
            severity=AlertSeverity.ERROR,
            source=source,
            metadata=metadata
        )
    
    def critical(self, source: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """记录严重级别告警"""
        self.trigger_alert(
            title="Critical",
            message=message,
            severity=AlertSeverity.CRITICAL,
            source=source,
            metadata=metadata
        )

    def get_alerts(self, 
                  severity: Optional[AlertSeverity] = None,
                  source: Optional[str] = None) -> List[Alert]:
        """获取告警历史"""
        alerts = []
        for notifier in self.notifiers:
            alerts.extend(notifier.get_alerts())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        return alerts 
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

class AlertSeverity(Enum):
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
    """控制台告警通知实现"""
    
    def send_alert(self, alert: Alert) -> bool:
        print(f"[{alert.timestamp}] {alert.severity.value}: {alert.title}")
        print(f"Source: {alert.source}")
        print(f"Message: {alert.message}")
        print(f"Metadata: {alert.metadata}")
        return True

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self._notifiers: List[AlertNotifier] = []
        self._alerts: List[Alert] = []

    def add_notifier(self, notifier: AlertNotifier) -> None:
        """添加告警通知器"""
        self._notifiers.append(notifier)

    def trigger_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
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
        self._alerts.append(alert)
        
        for notifier in self._notifiers:
            try:
                notifier.send_alert(alert)
            except Exception as e:
                print(f"Failed to send alert via {notifier.__class__.__name__}: {e}")

    def get_alerts(self, 
                  severity: Optional[AlertSeverity] = None,
                  source: Optional[str] = None) -> List[Alert]:
        """获取告警历史"""
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        return alerts 
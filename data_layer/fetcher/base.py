from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from monitoring.metrics.collector import MetricsCollector, DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, AlertSeverity
from monitoring.lineage.tracker import LineageTracker, DataNode, Operation, OperationType

class DataFetcherBase(ABC):
    """Base class for data fetching"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        lineage_tracker: LineageTracker
    ):
        """Initialize data fetcher base class"""
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.lineage_tracker = lineage_tracker
        self.source_node = None

    def set_lineage(self, source_id: str, source_name: str) -> None:
        """Set data lineage"""
        self.source_node = self.lineage_tracker.create_source_node(
            source_id=source_id,
            source_name=source_name,
            metadata={
                "fetcher": self.__class__.__name__,
                "timestamp": datetime.now().isoformat()
            }
        )

    def record_lineage(self, operation: str, output_data: pd.DataFrame) -> None:
        """Record data lineage"""
        self.lineage_tracker.record_operation(
            source_node=self.source_node,
            operation=operation,
            input_data={},
            output_data=output_data
        )

    def log_warning(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Record warning log"""
        self.alert_manager.warning(
            title="Data Fetching Warning",
            message=message,
            source=self.__class__.__name__,
            metadata=metadata
        )

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Handle data fetching error"""
        self.alert_manager.error(
            title="Data Fetching Error",
            message=str(error),
            source=self.__class__.__name__,
            metadata=context
        )
        raise error

    def standardize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format"""
        # Ensure required fields exist
        required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Data missing required fields: {missing_fields}")
        
        # Standardize date format
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure numeric types are correct
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df

    def validate_date_format(self, date_str: str) -> datetime:
        """Validate and standardize date format"""
        try:
            return pd.to_datetime(date_str)
        except Exception as e:
            raise ValueError(f"Invalid date format: {date_str}") from e

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data format

        Args:
            df: Original DataFrame

        Returns:
            pd.DataFrame: Standardized data
        """
        # Ensure required fields exist
        required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Standardize date format
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Ensure numeric types are correct
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce').astype(np.float64)
        
        return df

    def validate_dates(self, start_date: Union[str, datetime], 
                      end_date: Union[str, datetime]) -> tuple:
        """
        Validate and standardize date format

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            tuple: (datetime, datetime) Standardized start and end dates
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        if start_date > end_date:
            raise ValueError("Start date cannot be later than end date")
            
        return start_date, end_date

    def _record_fetch_metrics(self, start_time: datetime, data: Dict[str, Any], source: str) -> None:
        """Record metrics related to data fetching"""
        duration = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics_collector.record_latency(f"fetch_{source}", duration)
        
        # Record data volume
        if isinstance(data.get('data'), (list, tuple)):
            self.metrics_collector.record_data_volume(source, len(data['data']))
            
    @abstractmethod
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """Get historical data"""
        pass

    @abstractmethod
    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """Get latest data"""
        pass

    @abstractmethod
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate stock symbols"""
        pass

    def validate_response(self, data: Dict[str, Any]) -> bool:
        """Validate response data structure"""
        return True  # Subclasses can override this method to implement specific validation logic 
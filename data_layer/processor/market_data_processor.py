from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

from data_layer.processor.base import DataProcessorBase
from monitoring.metrics.collector import MetricsCollector
from monitoring.alerts.alert_manager import AlertManager
from monitoring.lineage.tracker import LineageTracker

class MarketDataProcessor(DataProcessorBase):
    """Market data processor"""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        lineage_tracker: LineageTracker
    ):
        """Initialize market data processor"""
        super().__init__(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            lineage_tracker=lineage_tracker
        )
        self._setup_lineage(
            source_id="market_data_processor",
            source_name="Market Data Processor",
            metadata={"processor": "market_data"}
        )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data"""
        try:
            # Record processing start time
            start_time = datetime.now()
            
            df = data['data']
            
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Calculate daily returns
            df['daily_return'] = df.groupby('symbol')['close'].pct_change()
            
            # Calculate moving averages
            df['ma_5'] = df.groupby('symbol')['close'].rolling(window=5).mean().reset_index(0, drop=True)
            df['ma_20'] = df.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
            
            # Calculate volatility
            df['volatility'] = df.groupby('symbol')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)
            
            # Calculate volume changes
            df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
            
            # Record processing completion time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "market_data_processing",
                processing_time
            )
            
            # Record data lineage
            self._record_lineage(
                operation="market_data_processing",
                input_data=data,
                output_data=df
            )
            
            return {
                'data': df,
                'metadata': {
                    'processing_time_ms': processing_time,
                    'added_features': [
                        'daily_return', 'ma_5', 'ma_20',
                        'volatility', 'volume_change'
                    ]
                }
            }
            
        except Exception as e:
            self.alert_manager.error(
                title="Market Data Processing Error",
                message=f"Error processing market data: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            raise 
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from data_layer.processor.base import DataProcessorBase
from data_layer.monitoring.metrics import MetricsCollector
from data_layer.monitoring.alerts import AlertManager
from data_layer.monitoring.lineage import LineageTracker

class MarketDataProcessor(DataProcessorBase):
    """Market data processor implementation"""
    
    def __init__(self,
                 metrics_collector: MetricsCollector = None,
                 alert_manager: AlertManager = None,
                 lineage_tracker: LineageTracker = None):
        """Initialize market data processor"""
        super().__init__(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            lineage_tracker=lineage_tracker
        )
        self.set_lineage(
            source_id="market_data_processor",
            source_name="Market Data Processor"
        )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data"""
        try:
            # Record processing start time
            start_time = datetime.now()
            
            df = data['data']
            available_fields = set(df.columns)
            added_features = []
            
            # Calculate daily returns if 'close' is available
            if 'close' in available_fields:
                df['daily_return'] = df.groupby('symbol')['close'].pct_change()
                added_features.append('daily_return')
                
                # Calculate moving averages
                df['ma_5'] = df.groupby('symbol')['close'].rolling(window=5).mean().reset_index(0, drop=True)
                df['ma_20'] = df.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
                added_features.extend(['ma_5', 'ma_20'])
                
                # Calculate volatility (requires daily_return)
                df['volatility'] = df.groupby('symbol')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)
                added_features.append('volatility')
            
            # Calculate volume changes if 'volume' is available
            if 'volume' in available_fields:
                df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
                added_features.append('volume_change')
            
            # Calculate true range if high, low, close are available
            if all(field in available_fields for field in ['high', 'low', 'close']):
                df['true_range'] = df.groupby('symbol').apply(
                    lambda x: pd.DataFrame({
                        'tr': np.maximum(
                            x['high'] - x['low'],
                            np.maximum(
                                abs(x['high'] - x['close'].shift(1)),
                                abs(x['low'] - x['close'].shift(1))
                            )
                        )
                    })
                ).reset_index(level=0, drop=True)
                added_features.append('true_range')
            
            # Record processing completion time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "market_data_processing",
                processing_time
            )
            
            # Record data lineage
            self.record_lineage(
                operation="market_data_processing",
                input_data=data,
                output_data=df
            )
            
            return {
                'data': df,
                'metadata': {
                    'processing_time_ms': processing_time,
                    'added_features': added_features,
                    'available_fields': list(available_fields),
                    'processed_fields': list(set(df.columns) - available_fields)
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

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data - implementation of abstract method"""
        try:
            # Record processing start time
            start_time = datetime.now()
            
            available_fields = set(data.columns)
            added_features = []
            
            # Calculate daily returns if 'close' is available
            if 'close' in available_fields:
                data['daily_return'] = data.groupby('symbol')['close'].pct_change()
                added_features.append('daily_return')
                
                # Calculate moving averages
                data['ma_5'] = data.groupby('symbol')['close'].rolling(window=5).mean().reset_index(0, drop=True)
                data['ma_20'] = data.groupby('symbol')['close'].rolling(window=20).mean().reset_index(0, drop=True)
                added_features.extend(['ma_5', 'ma_20'])
                
                # Calculate volatility (requires daily_return)
                data['volatility'] = data.groupby('symbol')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)
                added_features.append('volatility')
            
            # Calculate volume changes if 'volume' is available
            if 'volume' in available_fields:
                data['volume_change'] = data.groupby('symbol')['volume'].pct_change()
                added_features.append('volume_change')
            
            # Calculate true range if high, low, close are available
            if all(field in available_fields for field in ['high', 'low', 'close']):
                data['true_range'] = data.groupby('symbol').apply(
                    lambda x: pd.DataFrame({
                        'tr': np.maximum(
                            x['high'] - x['low'],
                            np.maximum(
                                abs(x['high'] - x['close'].shift(1)),
                                abs(x['low'] - x['close'].shift(1))
                            )
                        )
                    })
                ).reset_index(level=0, drop=True)
                added_features.append('true_range')
            
            # Record processing completion time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            if self.metrics_collector:
                self.metrics_collector.record_latency(
                    "market_data_processing",
                    processing_time
                )
            
            return data
            
        except Exception as e:
            if self.alert_manager:
                self.alert_manager.error(
                    title="Market Data Processing Error",
                    message=f"Error processing market data: {str(e)}",
                    source=self.__class__.__name__,
                    metadata={'error_type': e.__class__.__name__}
                )
            raise 
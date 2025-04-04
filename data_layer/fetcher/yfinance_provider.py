from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yfinance as yf

from data_layer.fetcher.base import DataFetcherBase
from monitoring.metrics.collector import DefaultMetricsCollector
from monitoring.alerts.alert_manager import AlertManager, AlertSeverity
from monitoring.lineage.tracker import LineageTracker

class YFinanceProvider(DataFetcherBase):
    """Yahoo Finance data provider implementation"""
    
    def __init__(self,
                 metrics_collector: DefaultMetricsCollector = None,
                 alert_manager: AlertManager = None,
                 lineage_tracker: LineageTracker = None):
        """Initialize Yahoo Finance provider"""
        if metrics_collector is None:
            metrics_collector = DefaultMetricsCollector()
        if alert_manager is None:
            alert_manager = AlertManager()
        if lineage_tracker is None:
            lineage_tracker = LineageTracker()
            
        super().__init__(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            lineage_tracker=lineage_tracker
        )
        self.set_lineage(
            source_id="yfinance",
            source_name="YFinance API"
        )
        # Add provider-specific metadata
        if self.source_node and self.source_node.metadata:
            self.source_node.metadata.update({
                "provider": "Yahoo Finance",
                "description": "YFinance API data source"
            })
    
    def get_historical_data(self, symbols: List[str], start_date: datetime, end_date: datetime, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get historical market data"""
        try:
            # Record fetching start time
            start_time = datetime.now()
            
            # Fetch data for each symbol
            all_data = []
            last_error = None
            
            for symbol in symbols:
                try:
                    # Get data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    # Skip if no data returned
                    if df.empty:
                        self.alert_manager.warning(
                            title="Data Fetching Warning",
                            message=f"No data returned for symbol {symbol}",
                            source=self.__class__.__name__,
                            metadata={
                                'symbol': symbol,
                                'provider': 'Yahoo Finance',
                                'start_date': start_date,
                                'end_date': end_date
                            }
                        )
                        continue
                    
                    # Debug: Print DataFrame info
                    self.alert_manager.warning(
                        title="DataFrame Info",
                        message=f"DataFrame info for {symbol}:\n{df.info()}",
                        source=self.__class__.__name__,
                        metadata={
                            'symbol': symbol,
                            'columns': list(df.columns),
                            'index': str(df.index)
                        }
                    )
                    
                    # Add symbol column
                    df['symbol'] = symbol
                    
                    # Reset index to make date a column
                    df = df.reset_index()
                    
                    # Filter fields if specified
                    if fields:
                        # Always include symbol and date columns
                        required_columns = ['symbol', 'Date']  # Note: 'Date' is the column name after reset_index
                        # Add any requested fields that exist in the DataFrame
                        available_fields = [f for f in fields if f in df.columns]
                        df = df[required_columns + available_fields]
                    
                    all_data.append(df)
                    
                except Exception as e:
                    last_error = e
                    self.alert_manager.warning(
                        title="Data Fetching Warning",
                        message=f"Error fetching data for {symbol}: {str(e)}",
                        source=self.__class__.__name__,
                        metadata={
                            'symbol': symbol,
                            'provider': 'Yahoo Finance',
                            'error': str(e)
                        }
                    )
            
            if not all_data:
                if last_error:
                    raise last_error
                raise ValueError("Failed to fetch data for any symbols")
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Record fetching completion time
            fetching_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "yfinance_data_fetching",
                fetching_time
            )
            
            # Record data lineage
            self.record_lineage(
                operation="historical_data_fetching",
                output_data=combined_df
            )
            
            return {
                'data': combined_df,
                'metadata': {
                    'fetching_time_ms': fetching_time,
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'fields': fields
                }
            }
            
        except Exception as e:
            self.alert_manager.error(
                title="Data Fetching Error",
                message=f"Error fetching historical data: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            raise
    
    def get_latest_data(self, symbols: List[str], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get latest market data"""
        try:
            # Record fetching start time
            start_time = datetime.now()
            
            # Get data for last 2 days to ensure we have latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            # Fetch data for each symbol
            all_data = []
            for symbol in symbols:
                try:
                    # Get data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    # Add symbol column
                    df['symbol'] = symbol
                    
                    # Reset index to make date a column
                    df = df.reset_index()
                    
                    # Filter fields if specified
                    if fields:
                        # Always include symbol and date columns
                        required_columns = ['symbol', 'Date']  # Note: 'Date' is the column name after reset_index
                        # Add any requested fields that exist in the DataFrame
                        available_fields = [f for f in fields if f in df.columns]
                        df = df[required_columns + available_fields]
                    
                    # Get the latest row
                    latest_data = df.iloc[-1]
                    all_data.append(latest_data)
                    
                except Exception as e:
                    self.alert_manager.warning(
                        title="Data Fetching Warning",
                        message=f"Error fetching latest data for {symbol}: {str(e)}",
                        source=self.__class__.__name__,
                        metadata={
                            'symbol': symbol,
                            'provider': 'Yahoo Finance',
                            'error': str(e)
                        }
                    )
            
            if not all_data:
                raise ValueError("Failed to fetch latest data for all symbols")
            
            # Create DataFrame from latest data
            latest_df = pd.DataFrame(all_data)
            
            # Record fetching completion time
            fetching_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "yfinance_latest_data_fetching",
                fetching_time
            )
            
            # Record data lineage
            self.record_lineage(
                operation="latest_data_fetching",
                output_data=latest_df
            )
            
            return {
                'data': latest_df,
                'metadata': {
                    'fetching_time_ms': fetching_time,
                    'symbols': symbols,
                    'timestamp': datetime.now(),
                    'fields': fields
                }
            }
            
        except Exception as e:
            self.alert_manager.error(
                title="Data Fetching Error",
                message=f"Error fetching latest data: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            raise
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate stock symbols"""
        if not symbols:
            raise ValueError("No symbols provided")
            
        if isinstance(symbols, str):
            symbols = [symbols]
            
        valid_symbols = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                if ticker.info and 'symbol' in ticker.info:
                    valid_symbols.append(symbol)
                else:
                    self.alert_manager.warning(
                        title="Invalid Symbol",
                        message=f"Symbol {symbol} not found",
                        source=self.__class__.__name__,
                        metadata={'symbol': symbol}
                    )
            except Exception as e:
                self.alert_manager.warning(
                    title="Symbol Validation Error",
                    message=f"Error validating symbol {symbol}: {str(e)}",
                    source=self.__class__.__name__,
                    metadata={'symbol': symbol, 'error': str(e)}
                )
                
        return valid_symbols
    
    def _validate_market_data(self, data: Dict[str, Any]) -> None:
        """Validate market data quality"""
        if data.get('data') is None or (isinstance(data['data'], pd.DataFrame) and data['data'].empty):
            if self.alert_manager:
                self.alert_manager.trigger_alert(
                    title="Empty Market Data",
                    message=f"No data returned for symbols: {data.get('symbols')}",
                    severity=AlertSeverity.WARNING,
                    source=self.__class__.__name__,
                    metadata=data.get('metadata', {})
                )
            return
            
        # Check data point count
        data_len = len(data['data']) if isinstance(data['data'], pd.DataFrame) else len(data['data'].keys())
        self.metrics_collector.record_data_quality(
            "market_data_volume",
            success=data_len > 0,
            details={"record_count": data_len}
        )
        
        # Check required fields only if no fields were specified
        if not data.get('metadata', {}).get('fields'):
            required_fields = {'open', 'high', 'low', 'close', 'volume'}
            df = pd.DataFrame(data['data'])
            actual_fields = set(df.columns)
            
            self.metrics_collector.record_data_quality(
                "required_fields_check",
                success=required_fields.issubset(actual_fields),
                details={
                    "missing_fields": list(required_fields - actual_fields),
                    "available_fields": list(actual_fields)
                }
            ) 
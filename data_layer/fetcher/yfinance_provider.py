from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yfinance as yf

from data_layer.fetcher.base import DataFetcherBase
from monitoring.alerts.alert_manager import AlertSeverity

class YFinanceProvider(DataFetcherBase):
    """YFinance data provider"""
    
    def __init__(self, metrics_collector=None, alert_manager=None, lineage_tracker=None):
        """Initialize YFinance data provider"""
        super().__init__(metrics_collector, alert_manager, lineage_tracker)
        self._setup_lineage(
            source_id="yfinance",
            source_name="YFinance API",
            metadata={
                "provider": "Yahoo Finance",
                "description": "YFinance API data source"
            }
        )
    
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """Get historical data"""
        # Validate dates
        start_date, end_date = self.validate_dates(start_date, end_date)
        
        # Handle single symbol case
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Validate symbols
        symbols = self.validate_symbols(symbols)
        if not symbols:
            raise ValueError("No valid symbols provided")
        
        all_data = []
        
        # Get data
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d"
                )
                
                if not data.empty:
                    data = data.reset_index()
                    data['symbol'] = symbol
                    all_data.append(data)
                
            except Exception as e:
                self._log_warning(f"Error getting historical data for {symbol}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("Failed to fetch data for all symbols")
        
        # Merge data
        result = pd.concat(all_data, ignore_index=True)
        result = self.normalize_data(result)
        
        # Filter fields
        if fields:
            available_fields = set(result.columns) - {'symbol', 'date'}
            valid_fields = [f for f in fields if f in available_fields]
            result = result[['symbol', 'date'] + valid_fields]
        
        # Record lineage
        self._record_lineage(
            operation="fetch_historical",
            input_data={
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "fields": fields
            },
            output_data=result
        )
        
        return {
            'data': result,
            'metadata': {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'fields': list(result.columns)
            }
        }
    
    def get_latest_data(self, symbols: Union[str, List[str]], fields: List[str] = None) -> Dict[str, Any]:
        """Get latest data"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        symbols = self.validate_symbols(symbols)
        if not symbols:
            raise ValueError("No valid symbols provided")
        
        latest_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                
                if not data.empty:
                    data = data.reset_index()
                    data['symbol'] = symbol
                    
                    if fields:
                        available_fields = set(data.columns) - {'symbol', 'date'}
                        valid_fields = [f for f in fields if f in available_fields]
                        data = data[['symbol', 'date'] + valid_fields]
                    
                    latest_data[symbol] = data.iloc[-1]
                
            except Exception as e:
                self._log_warning(f"Error getting latest data for {symbol}: {str(e)}")
                continue
        
        if not latest_data:
            raise ValueError("Failed to fetch latest data for all symbols")
        
        return {
            'data': latest_data,
            'metadata': {
                'symbols': symbols,
                'timestamp': datetime.now(),
                'fields': fields
            }
        }
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols"""
        if not symbols:
            raise ValueError("No symbols provided")
        
        if isinstance(symbols, str):
            symbols = [symbols]
        
        valid_symbols = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info:
                    valid_symbols.append(symbol)
            except Exception as e:
                self._log_warning(f"Invalid symbol: {symbol}, error: {str(e)}")
        
        if not valid_symbols:
            raise ValueError("No valid symbols provided")
        
        return valid_symbols
    
    def _validate_market_data(self, data: Dict[str, Any]) -> None:
        """Validate market data quality"""
        if not data.get('data'):
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
        self.metrics_collector.record_data_quality(
            "market_data_volume",
            success=len(data['data']) > 0,
            details={"record_count": len(data['data'])}
        )
        
        # Check required fields
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
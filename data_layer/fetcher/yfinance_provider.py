from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yfinance as yf

from .base import DataFetcherBase
from monitoring.alerts.alert_manager import AlertSeverity

class YFinanceProvider(DataFetcherBase):
    """YFinance数据提供者"""
    
    def __init__(self, metrics_collector=None, alert_manager=None, lineage_tracker=None):
        """初始化YFinance数据提供者"""
        super().__init__(metrics_collector, alert_manager, lineage_tracker)
        self._setup_lineage(
            source_id="yfinance",
            source_name="YFinance API",
            metadata={
                "provider": "yfinance",
                "description": "YFinance API数据源"
            }
        )
    
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """获取历史数据"""
        # 验证日期
        self.validate_dates(start_date, end_date)

        # 处理单个股票代码的情况
        if isinstance(symbols, str):
            symbols = [symbols]

        # 验证股票代码
        valid_symbols = self.validate_symbols(symbols)
        if not valid_symbols:
            return {
                'data': pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']),
                'symbols': []
            }

        # 获取数据
        all_data = []
        for symbol in valid_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    data = data.reset_index()
                    data['symbol'] = symbol
                    all_data.append(data)
            except Exception as e:
                self._log_warning(f"获取 {symbol} 的历史数据时出错: {str(e)}")

        if not all_data:
            return {
                'data': pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']),
                'symbols': []
            }

        # 合并数据
        result_data = pd.concat(all_data, ignore_index=True)
        result_data = result_data.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # 过滤字段
        if fields:
            result_data = result_data[['symbol', 'date'] + [f for f in fields if f in result_data.columns]]
        else:
            result_data = result_data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]

        # 记录血缘关系
        if self.lineage_tracker:
            self._record_lineage(
                operation="get_historical_data",
                input_data={
                    "symbols": valid_symbols,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "fields": fields
                },
                output_data=result_data
            )

        return {
            'data': result_data,
            'symbols': valid_symbols
        }
    
    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        fields: List[str] = None
    ) -> Dict[str, Any]:
        """获取最新数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        result = self.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )

        if result['data'].empty:
            return {
                'data': pd.DataFrame(columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']),
                'symbols': []
            }

        # 获取每个股票的最新数据
        latest_data = result['data'].sort_values('date').groupby('symbol').last().reset_index()

        # 过滤字段
        if fields:
            latest_data = latest_data[['symbol', 'date'] + [f for f in fields if f in latest_data.columns]]
        else:
            latest_data = latest_data[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]

        return {
            'data': latest_data,
            'symbols': result['symbols']
        }
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """验证股票代码"""
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
        """验证市场数据的质量"""
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
            
        # 检查数据点数量
        self.metrics_collector.record_data_quality(
            "market_data_volume",
            success=len(data['data']) > 0,
            details={"record_count": len(data['data'])}
        )
        
        # 检查必要字段
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
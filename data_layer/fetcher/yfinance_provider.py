from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yfinance as yf

from .base import DataFetcherBase
from monitoring.alerts.alert_manager import AlertSeverity

class YFinanceProvider(DataFetcherBase):
    """YFinance数据提供者"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_lineage(
            source_id="yfinance_api",
            source_name="YFinance API Data",
            metadata={"provider": "yfinance", "data_type": "market_data"}
        )
    
    def get_historical_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """获取历史数据"""
        try:
            start_time = datetime.now()
            
            # 标准化日期
            start_date, end_date = self.validate_dates(start_date, end_date)
            
            # 标准化symbols为列表
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # 验证symbols
            valid_symbols = self.validate_symbols(symbols)
            if not valid_symbols:
                raise ValueError("No valid symbols provided")
            
            # 获取数据
            all_data = []
            for symbol in valid_symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d"
                )
                
                # 转换为标准格式
                df = hist.reset_index()
                df['symbol'] = symbol
                
                # 重命名列以匹配标准格式
                df = df.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # 标准化数据
                df = self.normalize_data(df)
                all_data.append(df)
            
            # 合并所有数据
            final_df = pd.concat(all_data, ignore_index=True)
            
            # 准备返回数据
            data = {
                'symbols': valid_symbols,
                'data': final_df.to_dict('records'),
                'metadata': {
                    'start_date': start_date.strftime("%Y-%m-%d"),
                    'end_date': end_date.strftime("%Y-%m-%d"),
                    'fields': fields or list(final_df.columns),
                    'provider': 'yfinance'
                }
            }
            
            # 记录性能指标
            self._record_fetch_metrics(start_time, data, "yfinance")
            
            # 验证数据质量
            self._validate_market_data(data)
            
            return data
            
        except Exception as e:
            self._handle_fetch_error(e, "yfinance")
    
    def get_latest_data(
        self,
        symbols: Union[str, List[str]],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """获取最新数据"""
        # 获取当前日期
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=5)  # 获取最近5天的数据以确保有最新数据
        
        # 复用历史数据获取逻辑
        data = self.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )
        
        # 只保留每个symbol的最新数据
        df = pd.DataFrame(data['data'])
        latest_data = df.sort_values('date').groupby('symbol').last().reset_index()
        
        data['data'] = latest_data.to_dict('records')
        data['metadata']['data_type'] = 'latest'
        
        return data
    
    def validate_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """验证股票代码是否有效"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        valid_symbols = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # 尝试获取基本信息来验证股票代码是否有效
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    valid_symbols.append(symbol)
            except Exception:
                continue
                
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
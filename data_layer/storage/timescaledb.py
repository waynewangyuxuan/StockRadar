import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import json

from .base import DataStorageBase
from data_layer.monitoring.alerts import AlertManager
from data_layer.monitoring.metrics import MetricsCollector
from data_layer.cache.redis_cache import RedisCache

class TimescaleDBStorage(DataStorageBase):
    """TimescaleDB storage implementation"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "stockradar",
        user: str = "postgres",
        password: str = "postgres",
        alert_manager: Optional[AlertManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        cache: Optional[RedisCache] = None
    ):
        """Initialize TimescaleDB storage
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            alert_manager: Optional alert manager for notifications
            metrics_collector: Optional metrics collector for performance tracking
            cache: Redis cache instance
        """
        super().__init__()
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.alert_manager = alert_manager or AlertManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.cache = cache or RedisCache(
            alert_manager=self.alert_manager,
            metrics_collector=self.metrics_collector
        )
        
        # SQLAlchemy engine for pandas integration
        self.engine = None
        
        # Table names
        self.market_data_table = "market_data"
        
    def connect(self) -> None:
        """Establish connection to TimescaleDB"""
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create tables if they don't exist
            self._create_tables()
            
            self.is_connected = True
            self.alert_manager.info(
                title="Database Connection",
                message="Successfully connected to TimescaleDB",
                source=self.__class__.__name__
            )
            
        except OperationalError as e:
            self.is_connected = False
            self.alert_manager.error(
                title="Database Connection Error",
                message=f"Failed to connect to TimescaleDB: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': type(e).__name__}
            )
            raise
    
    def disconnect(self) -> None:
        """Close connection to TimescaleDB"""
        if self.engine:
            self.engine.dispose()
            self.is_connected = False
            self.alert_manager.info(
                title="Database Connection",
                message="Disconnected from TimescaleDB",
                source=self.__class__.__name__
            )
    
    def is_available(self) -> bool:
        """Check if TimescaleDB is available"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist"""
        with self.engine.connect() as conn:
            # Create market_data table with TimescaleDB features
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.market_data_table} (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    metadata JSONB,
                    CONSTRAINT market_data_pkey PRIMARY KEY (time, symbol)
                );
                
                -- Convert to TimescaleDB hypertable
                SELECT create_hypertable('{self.market_data_table}', 'time', if_not_exists => TRUE);
                
                -- Create index on symbol for faster lookups
                CREATE INDEX IF NOT EXISTS idx_{self.market_data_table}_symbol 
                ON {self.market_data_table} (symbol, time DESC);
            """))
            conn.commit()
    
    def save_market_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save market data to TimescaleDB
        
        Args:
            data: Market data to save
            metadata: Optional metadata about the data
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Record start time
            start_time = datetime.now()
            
            # Ensure we have a DataFrame
            if not isinstance(data.get('data'), pd.DataFrame):
                raise ValueError("Data must contain a pandas DataFrame")
            
            df = data['data']
            
            # Ensure required columns exist
            required_columns = ['symbol', 'Date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Rename Date column to time for TimescaleDB
            df = df.rename(columns={'Date': 'time'})
            
            # Add metadata if provided
            if metadata:
                df['metadata'] = str(metadata)
            
            # Save to TimescaleDB
            df.to_sql(
                self.market_data_table,
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            # Cache the data
            cache_key = f"market_data:{df['symbol'].iloc[0]}:{df['time'].iloc[0].strftime('%Y-%m-%d')}"
            self.cache.set(cache_key, data, expire=3600)  # Cache for 1 hour
            
            # Record completion time
            save_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "timescaledb_data_saving",
                save_time
            )
            
            self.alert_manager.info(
                title="Data Saved",
                message=f"Successfully saved {len(df)} records to TimescaleDB",
                source=self.__class__.__name__,
                metadata={
                    'num_records': len(df),
                    'save_time_ms': save_time
                }
            )
            
            return True
            
        except Exception as e:
            self.alert_manager.error(
                title="Data Save Error",
                message=f"Failed to save data to TimescaleDB: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            return False
    
    def get_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get market data from TimescaleDB
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            fields: Optional list of fields to retrieve
            
        Returns:
            Dict containing the market data and metadata
        """
        try:
            # Record start time
            start_time = datetime.now()
            
            # Try to get from cache first
            cache_key = f"market_data:{symbols[0]}:{start_date.strftime('%Y-%m-%d')}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.alert_manager.info(
                    title="Cache Hit",
                    message=f"Retrieved market data for {symbols[0]} from cache",
                    source=self.__class__.__name__,
                    metadata={
                        'symbols': symbols,
                        'start_date': start_date,
                        'end_date': end_date,
                        'fields': fields
                    }
                )
                return cached_data
            
            # If not in cache, get from database
            query = f"""
                SELECT 
                    time as "Date",
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    metadata
                FROM {self.market_data_table}
                WHERE symbol = ANY(%s)
                AND time BETWEEN %s AND %s
                ORDER BY time, symbol
            """
            
            # Execute query
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params=(symbols, start_date, end_date)
                )
            
            # Filter fields if specified
            if fields:
                required_columns = ['symbol', 'Date']
                available_fields = [f for f in fields if f in df.columns]
                df = df[required_columns + available_fields]
            
            # Record completion time
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "timescaledb_data_querying",
                query_time
            )
            
            # Cache the result
            self.cache.set(cache_key, {
                'data': df.to_dict('records'),
                'metadata': {
                    'query_time_ms': query_time,
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'fields': fields
                }
            }, expire=3600)  # Cache for 1 hour
            
            self.alert_manager.info(
                title="Data Retrieved",
                message=f"Successfully retrieved market data for {symbols[0]}",
                source=self.__class__.__name__,
                metadata={
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'fields': fields
                }
            )
            
            return {
                'data': df,
                'metadata': {
                    'query_time_ms': query_time,
                    'symbols': symbols,
                    'start_date': start_date,
                    'end_date': end_date,
                    'fields': fields
                }
            }
            
        except Exception as e:
            self.alert_manager.error(
                title="Data Retrieval Error",
                message=f"Failed to retrieve market data: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            raise
    
    def get_latest_data(self, symbols: List[str], fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get latest market data from TimescaleDB
        
        Args:
            symbols: List of stock symbols
            fields: Optional list of fields to retrieve
            
        Returns:
            Dict containing the latest market data and metadata
        """
        try:
            # Record start time
            start_time = datetime.now()
            
            # Build query to get latest data for each symbol
            query = f"""
                WITH latest_times AS (
                    SELECT symbol, MAX(time) as max_time
                    FROM {self.market_data_table}
                    WHERE symbol = ANY(%s)
                    GROUP BY symbol
                )
                SELECT 
                    m.time as "Date",
                    m.symbol,
                    m.open,
                    m.high,
                    m.low,
                    m.close,
                    m.volume,
                    m.metadata
                FROM {self.market_data_table} m
                INNER JOIN latest_times l
                ON m.symbol = l.symbol AND m.time = l.max_time
                ORDER BY m.symbol
            """
            
            # Execute query
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=(symbols,))
            
            # Filter fields if specified
            if fields:
                required_columns = ['symbol', 'Date']
                available_fields = [f for f in fields if f in df.columns]
                df = df[required_columns + available_fields]
            
            # Record completion time
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "timescaledb_latest_data_querying",
                query_time
            )
            
            return {
                'data': df,
                'metadata': {
                    'query_time_ms': query_time,
                    'symbols': symbols,
                    'fields': fields
                }
            }
            
        except Exception as e:
            self.alert_manager.error(
                title="Latest Data Query Error",
                message=f"Failed to query latest data from TimescaleDB: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            raise
    
    def delete_market_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """Delete market data from TimescaleDB
        
        Args:
            symbols: Optional list of stock symbols to delete data for
            start_date: Optional start date for deletion range
            end_date: Optional end date for deletion range
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Record start time
            start_time = datetime.now()
            
            # Build query
            conditions = []
            params = []
            
            if symbols:
                conditions.append("symbol = ANY(%s)")
                params.append(symbols)
            
            if start_date:
                conditions.append("time >= %s")
                params.append(start_date)
            
            if end_date:
                conditions.append("time <= %s")
                params.append(end_date)
            
            if not conditions:
                raise ValueError("At least one condition must be specified for deletion")
            
            query = f"""
                DELETE FROM {self.market_data_table}
                WHERE {' AND '.join(conditions)}
            """
            
            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                conn.commit()
                num_deleted = result.rowcount
            
            # Record completion time
            delete_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.metrics_collector.record_latency(
                "timescaledb_data_deletion",
                delete_time
            )
            
            # Invalidate cache for affected symbols
            if symbols:
                for symbol in symbols:
                    cache_key = f"market_data:{symbol}:*"
                    self.cache.delete(cache_key)
            
            self.alert_manager.info(
                title="Data Deleted",
                message=f"Successfully deleted {num_deleted} records from TimescaleDB",
                source=self.__class__.__name__,
                metadata={
                    'num_deleted': num_deleted,
                    'delete_time_ms': delete_time
                }
            )
            
            return True
            
        except Exception as e:
            self.alert_manager.error(
                title="Data Deletion Error",
                message=f"Failed to delete data from TimescaleDB: {str(e)}",
                source=self.__class__.__name__,
                metadata={'error_type': e.__class__.__name__}
            )
            return False 
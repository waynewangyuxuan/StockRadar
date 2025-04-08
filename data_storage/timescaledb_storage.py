"""TimescaleDB storage implementation."""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from typing import Optional, List, Dict, Any
from .base import DataStorageBase
import logging

class TimescaleDBStorage(DataStorageBase):
    """TimescaleDB storage for market data.
    
    This implementation uses TimescaleDB for efficient time-series data storage.
    Data is stored in hypertables with the following schema:
    - market_data (ticker, date, open, high, low, close, volume)
    - dataset_versions (dataset_name, version, created_at)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TimescaleDB storage.
        
        Args:
            config: Dictionary with TimescaleDB configuration:
                - host: Database host (default: localhost)
                - port: Database port (default: 5432)
                - dbname: Database name (default: stockradar)
                - user: Database user
                - password: Database password
                - schema: Database schema (default: public)
        """
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up database connection
        self.conn_params = {
            'host': self.config.get('host', 'localhost'),
            'port': self.config.get('port', 5432),
            'dbname': self.config.get('dbname', 'stockradar'),
            'user': self.config.get('user'),
            'password': self.config.get('password'),
            'options': f"-c search_path={self.config.get('schema', 'public')}"
        }
        
        # Initialize database schema
        self._init_schema()
        
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.conn_params)
        
    def _init_schema(self):
        """Initialize database schema with hypertables."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Create schema if not exists
                    schema = self.config.get('schema', 'public')
                    cur.execute(f"""
                        CREATE SCHEMA IF NOT EXISTS {schema};
                    """)

                    # Set search path
                    cur.execute(f"""
                        SET search_path TO {schema}, public;
                    """)

                    # Create datasets table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {schema}.datasets (
                            name TEXT PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW()
                        );
                    """)

                    # Create versions table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {schema}.dataset_versions (
                            dataset_name TEXT REFERENCES {schema}.datasets(name),
                            version TEXT,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (dataset_name, version)
                        );
                    """)

                    # Create market data table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {schema}.market_data (
                            dataset_name TEXT,
                            version TEXT,
                            ticker TEXT,
                            date TIMESTAMPTZ,
                            open DOUBLE PRECISION,
                            high DOUBLE PRECISION,
                            low DOUBLE PRECISION,
                            close DOUBLE PRECISION,
                            volume BIGINT,
                            PRIMARY KEY (dataset_name, version, ticker, date)
                        );
                    """)

                    # Convert to hypertable if not already
                    cur.execute(f"""
                        SELECT create_hypertable('{schema}.market_data', 'date',
                            if_not_exists => TRUE,
                            migrate_data => TRUE
                        );
                    """)

                    conn.commit()
                    return True

        except Exception as e:
            self.logger.error(f"Error initializing schema: {str(e)}")
            return False
    
    def save_data(self,
                 data: pd.DataFrame,
                 dataset_name: str,
                 version: Optional[str] = None) -> bool:
        """Save market data to TimescaleDB.
        
        Args:
            data: DataFrame containing market data
            dataset_name: Name/identifier for the dataset
            version: Optional version identifier
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Validate data first
            self.validate_data(data)
            
            # Use 'latest' as default version
            version = version or 'latest'
            
            schema = self.config.get('schema', 'public')
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Insert dataset if not exists
                    cur.execute(f"""
                        INSERT INTO {schema}.datasets (name)
                        VALUES (%s)
                        ON CONFLICT (name) DO NOTHING;
                    """, (dataset_name,))
                    
                    # Insert version if not exists
                    cur.execute(f"""
                        INSERT INTO {schema}.dataset_versions (dataset_name, version)
                        VALUES (%s, %s)
                        ON CONFLICT (dataset_name, version) DO NOTHING;
                    """, (dataset_name, version))
                    
                    # Insert market data
                    for _, row in data.iterrows():
                        cur.execute(f"""
                            INSERT INTO {schema}.market_data (
                                dataset_name, version, ticker, date,
                                open, high, low, close, volume
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (dataset_name, version, ticker, date)
                            DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume;
                        """, (
                            dataset_name, version, row['ticker'], row['date'],
                            row['open'], row['high'], row['low'], row['close'], row['volume']
                        ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error saving data to TimescaleDB: {str(e)}")
            return False
    
    def load_data(self, dataset_name: str, version: Optional[str] = None) -> pd.DataFrame:
        """Load data from TimescaleDB.
        
        Args:
            dataset_name: Name of the dataset to load
            version: Optional version to load (defaults to 'latest')
            
        Returns:
            DataFrame with the loaded data
        """
        if version is None:
            version = 'latest'
            
        schema = self.config.get('schema', 'public')
            
        with self._get_connection() as conn:
            # Load data with proper ordering
            query = f"""
                SELECT t.ticker, t.date::timestamp without time zone as date,
                       t.open, t.high, t.low, t.close, t.volume
                FROM {schema}.market_data t
                WHERE t.dataset_name = %s AND t.version = %s
                ORDER BY t.ticker, t.date
            """
            
            df = pd.read_sql_query(query, conn, params=(dataset_name, version))
            return df.sort_values(['ticker', 'date']).reset_index(drop=True)
            
    def _get_dataset_id(self, cur, dataset_name: str) -> Optional[int]:
        """Get dataset ID from name.
        
        Args:
            cur: Database cursor
            dataset_name: Name of the dataset
            
        Returns:
            Dataset ID or None if not found
        """
        query = f"""
            SELECT id FROM {self.config.get('schema', 'public')}.datasets
            WHERE name = %s
        """
        cur.execute(query, (dataset_name,))
        result = cur.fetchone()
        return result[0] if result else None
        
    def _get_version_id(self, cur, dataset_id: int, version: str) -> Optional[int]:
        """Get version ID from dataset ID and version name.
        
        Args:
            cur: Database cursor
            dataset_id: ID of the dataset
            version: Version name
            
        Returns:
            Version ID or None if not found
        """
        query = f"""
            SELECT id FROM {self.config.get('schema', 'public')}.dataset_versions
            WHERE dataset_id = %s AND version = %s
        """
        cur.execute(query, (dataset_id, version))
        result = cur.fetchone()
        return result[0] if result else None
    
    def delete_data(self,
                   dataset_name: str,
                   version: Optional[str] = None) -> bool:
        """Delete market data from TimescaleDB.
        
        Args:
            dataset_name: Name/identifier for the dataset
            version: Optional version to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            schema = self.config.get('schema', 'public')
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if version:
                        # Delete specific version
                        cur.execute(f"""
                            DELETE FROM {schema}.market_data
                            WHERE dataset_name = %s AND version = %s;
                            
                            DELETE FROM {schema}.dataset_versions
                            WHERE dataset_name = %s AND version = %s;
                        """, (dataset_name, version, dataset_name, version))
                    else:
                        # Delete all versions
                        cur.execute(f"""
                            DELETE FROM {schema}.market_data
                            WHERE dataset_name = %s;
                            
                            DELETE FROM {schema}.dataset_versions
                            WHERE dataset_name = %s;
                            
                            DELETE FROM {schema}.datasets
                            WHERE name = %s;
                        """, (dataset_name, dataset_name, dataset_name))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error deleting data from TimescaleDB: {str(e)}")
            return False
    
    def list_datasets(self) -> List[str]:
        """List all available datasets.
        
        Returns:
            List of dataset names
        """
        try:
            schema = self.config.get('schema', 'public')
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT name FROM {schema}.datasets
                        ORDER BY name;
                    """)
                    
                    return [row[0] for row in cur.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error listing datasets: {str(e)}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of version identifiers
        """
        try:
            schema = self.config.get('schema', 'public')
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT version FROM {schema}.dataset_versions
                        WHERE dataset_name = %s
                        ORDER BY version;
                    """, (dataset_name,))
                    
                    return [row[0] for row in cur.fetchall()]
                    
        except Exception as e:
            self.logger.error(f"Error getting versions: {str(e)}")
            return []

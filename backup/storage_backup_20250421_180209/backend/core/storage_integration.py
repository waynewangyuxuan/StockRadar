#!/usr/bin/env python3
"""
Storage Integration Module

This module integrates the various storage backends with the trading system,
providing a unified interface for storing and retrieving trading data.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime

from data_storage import LocalStorage, RedisCache, TimescaleDBStorage, VersionControl

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages data storage across different backends for the trading system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage backends based on configuration.
        
        Args:
            config: Storage configuration dictionary
        """
        self.config = config
        self.storage_backends = {}
        self.version_control = None
        self._initialize_backends()
        
    def _initialize_backends(self) -> None:
        """Initialize storage backends based on configuration."""
        storage_config = self.config.get("storage", {})
        
        # Initialize LocalStorage
        if storage_config.get("local", {}).get("enabled", True):
            try:
                local_config = storage_config.get("local", {})
                base_dir = local_config.get("base_dir", "data")
                self.storage_backends["local"] = LocalStorage(base_dir=base_dir)
                logger.info(f"LocalStorage initialized with base_dir={base_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize LocalStorage: {str(e)}")
        
        # Initialize RedisCache
        if storage_config.get("redis", {}).get("enabled", False):
            try:
                redis_config = storage_config.get("redis", {})
                host = redis_config.get("host", "localhost")
                port = redis_config.get("port", 6379)
                db = redis_config.get("db", 0)
                self.storage_backends["redis"] = RedisCache(host=host, port=port, db=db)
                logger.info(f"RedisCache initialized with host={host}, port={port}")
            except Exception as e:
                logger.error(f"Failed to initialize RedisCache: {str(e)}")
        
        # Initialize TimescaleDBStorage
        if storage_config.get("timescaledb", {}).get("enabled", False):
            try:
                timescaledb_config = storage_config.get("timescaledb", {})
                connection_string = timescaledb_config.get("connection_string", 
                                                           "postgresql://postgres:postgres@localhost:5432/stockradar")
                self.storage_backends["timescaledb"] = TimescaleDBStorage(connection_string=connection_string)
                logger.info("TimescaleDBStorage initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TimescaleDBStorage: {str(e)}")
        
        # Initialize VersionControl with primary backend
        primary_backend = storage_config.get("primary_backend", "local")
        if primary_backend in self.storage_backends:
            try:
                self.version_control = VersionControl(self.storage_backends[primary_backend])
                logger.info(f"VersionControl initialized with {primary_backend} backend")
            except Exception as e:
                logger.error(f"Failed to initialize VersionControl: {str(e)}")
    
    def store_market_data(self, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> str:
        """Store market data with versioning.
        
        Args:
            data: Market data DataFrame
            metadata: Optional metadata about the data
            
        Returns:
            Version ID of the stored data
        """
        if self.version_control is None:
            logger.warning("VersionControl not initialized, using LocalStorage fallback")
            # Fallback to LocalStorage
            if "local" not in self.storage_backends:
                self.storage_backends["local"] = LocalStorage()
            
            self.version_control = VersionControl(self.storage_backends["local"])
        
        # Prepare metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Store data with version control
        version_id, success = self.version_control.create_version(
            data, 
            "market_data", 
            metadata
        )
        
        if success:
            logger.info(f"Stored market data with version_id={version_id}")
        else:
            logger.error("Failed to store market data")
            
        return version_id
    
    def store_factor_data(self, factor_data: Dict[str, pd.DataFrame], metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """Store factor data with versioning.
        
        Args:
            factor_data: Dictionary mapping factor names to DataFrames
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary mapping factor names to version IDs
        """
        version_ids = {}
        
        for factor_name, data in factor_data.items():
            # Prepare factor-specific metadata
            factor_metadata = metadata.copy() if metadata else {}
            factor_metadata["factor_name"] = factor_name
            factor_metadata["timestamp"] = datetime.now().isoformat()
            
            # Store factor data
            version_id = self.store_market_data(data, factor_metadata)
            version_ids[factor_name] = version_id
            
        return version_ids
    
    def store_backtest_results(self, results: Dict[str, Any], strategy_name: str, metadata: Dict[str, Any] = None) -> str:
        """Store backtest results with versioning.
        
        Args:
            results: Dictionary of backtest results
            strategy_name: Name of the strategy
            metadata: Optional metadata about the results
            
        Returns:
            Version ID of the stored results
        """
        # Prepare metadata
        result_metadata = metadata.copy() if metadata else {}
        result_metadata["strategy_name"] = strategy_name
        result_metadata["timestamp"] = datetime.now().isoformat()
        
        # Convert Dict results to DataFrame for storage
        # Equity curve is already a DataFrame
        equity_df = results.get("equity_curve", pd.DataFrame())
        
        # Store metrics as DataFrame
        metrics_df = pd.DataFrame([results.get("metrics", {})])
        
        # Store trades as DataFrame
        trades_df = pd.DataFrame(results.get("trades", []))
        
        # Store each component
        equity_id = self.store_market_data(equity_df, {**result_metadata, "component": "equity_curve"})
        metrics_id = self.store_market_data(metrics_df, {**result_metadata, "component": "metrics"})
        trades_id = self.store_market_data(trades_df, {**result_metadata, "component": "trades"})
        
        # Store reference to all components
        components = {
            "equity_curve": equity_id,
            "metrics": metrics_id,
            "trades": trades_id
        }
        
        # Create index entry for the complete backtest
        index_metadata = {**result_metadata, "components": components}
        if self.version_control:
            index_id, _ = self.version_control.create_version(
                pd.DataFrame([components]), 
                f"backtest_{strategy_name}",
                index_metadata
            )
            return index_id
        else:
            logger.error("Failed to store backtest index, VersionControl not initialized")
            return None
    
    def get_market_data(self, version_id: str = None, latest: bool = True) -> pd.DataFrame:
        """Retrieve market data by version ID or get latest.
        
        Args:
            version_id: Optional version ID to retrieve
            latest: Whether to get the latest version (if version_id not specified)
            
        Returns:
            DataFrame containing the requested market data
        """
        if self.version_control is None:
            logger.error("VersionControl not initialized")
            return pd.DataFrame()
            
        if version_id:
            # Get specific version
            data, metadata = self.version_control.get_version("market_data", version_id)
        elif latest:
            # Get latest version
            versions = self.version_control.list_versions("market_data")
            if not versions:
                logger.warning("No market data versions found")
                return pd.DataFrame()
                
            latest_version = versions[-1]
            data, metadata = self.version_control.get_version("market_data", latest_version)
        else:
            logger.error("Either version_id or latest=True must be specified")
            return pd.DataFrame()
            
        return data
    
    def get_backtest_results(self, strategy_name: str, version_id: str = None) -> Dict[str, Any]:
        """Retrieve backtest results for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            version_id: Optional version ID (if not specified, get latest)
            
        Returns:
            Dictionary containing the backtest results
        """
        if self.version_control is None:
            logger.error("VersionControl not initialized")
            return {}
            
        # Get backtest index
        if version_id:
            index_df, index_metadata = self.version_control.get_version(f"backtest_{strategy_name}", version_id)
        else:
            # Get latest version
            versions = self.version_control.list_versions(f"backtest_{strategy_name}")
            if not versions:
                logger.warning(f"No backtest results found for strategy {strategy_name}")
                return {}
                
            latest_version = versions[-1]
            index_df, index_metadata = self.version_control.get_version(f"backtest_{strategy_name}", latest_version)
        
        # Extract component IDs
        components = index_metadata.get("components", {})
        
        # Retrieve each component
        results = {}
        
        if "equity_curve" in components:
            equity_df, _ = self.version_control.get_version("market_data", components["equity_curve"])
            results["equity_curve"] = equity_df
            
        if "metrics" in components:
            metrics_df, _ = self.version_control.get_version("market_data", components["metrics"])
            # Convert metrics DataFrame to dict
            results["metrics"] = metrics_df.iloc[0].to_dict() if not metrics_df.empty else {}
            
        if "trades" in components:
            trades_df, _ = self.version_control.get_version("market_data", components["trades"])
            # Convert trades DataFrame to list of dicts
            results["trades"] = trades_df.to_dict(orient="records")
            
        return results 
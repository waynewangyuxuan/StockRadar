from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

class DataProvider(Enum):
    YFINANCE = "yfinance_provider"
    SIMULATOR = "simulator_provider"
    HIVE = "hive_provider"

class CacheProvider(Enum):
    REDIS = "redis"
    LOCAL = "local"

@dataclass
class DataSourceConfig:
    provider: DataProvider
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    interval: str

@dataclass
class FactorConfig:
    class_path: str
    params: Dict[str, Any]

@dataclass
class StrategyConfig:
    class_path: str
    params: Dict[str, Any]

@dataclass
class OutputConfig:
    path: str
    format: str
    include_metadata: bool

@dataclass
class LoggingConfig:
    level: str
    file: str
    max_size: int
    backup_count: int

@dataclass
class CacheConfig:
    enabled: bool
    provider: CacheProvider
    params: Dict[str, Any]

@dataclass
class MonitoringConfig:
    enabled: bool
    metrics: List[str]
    alert_thresholds: Dict[str, float]

@dataclass
class StockRadarConfig:
    data_source: DataSourceConfig
    factors: List[FactorConfig]
    strategies: List[StrategyConfig]
    output: OutputConfig
    logging: LoggingConfig
    cache: Optional[CacheConfig] = None
    monitoring: Optional[MonitoringConfig] = None

class ConfigValidator:
    """Validates the configuration structure and values."""
    
    @staticmethod
    def validate_data_source(config: Dict[str, Any]) -> None:
        """Validate data source configuration."""
        if 'provider' not in config:
            raise ValueError("data_source.provider is required")
            
        try:
            DataProvider(config['provider'])
        except ValueError:
            raise ValueError(f"Invalid data provider: {config['provider']}")
            
        if 'params' not in config:
            raise ValueError("data_source.params is required")
            
        params = config['params']
        required_params = ['symbols', 'start_date', 'end_date', 'interval']
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required data source parameters: {missing}")
            
        if not isinstance(params['symbols'], list):
            raise ValueError("data_source.params.symbols must be a list")
            
        try:
            datetime.strptime(params['start_date'], '%Y-%m-%d')
            datetime.strptime(params['end_date'], '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
            
    @staticmethod
    def validate_factors(config: List[Dict[str, Any]]) -> None:
        """Validate factor configurations."""
        if not config:
            raise ValueError("At least one factor is required")
            
        for factor in config:
            if 'class' not in factor:
                raise ValueError("Each factor must specify a class")
            if 'params' not in factor:
                raise ValueError("Each factor must specify params")
                
    @staticmethod
    def validate_strategies(config: List[Dict[str, Any]]) -> None:
        """Validate strategy configurations."""
        if not config:
            raise ValueError("At least one strategy is required")
            
        for strategy in config:
            if 'class' not in strategy:
                raise ValueError("Each strategy must specify a class")
            if 'params' not in strategy:
                raise ValueError("Each strategy must specify params")
                
    @staticmethod
    def validate_output(config: Dict[str, Any]) -> None:
        """Validate output configuration."""
        required = ['path', 'format', 'include_metadata']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required output parameters: {missing}")
            
        if not isinstance(config['include_metadata'], bool):
            raise ValueError("output.include_metadata must be a boolean")
            
    @staticmethod
    def validate_logging(config: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        required = ['level', 'file', 'max_size', 'backup_count']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required logging parameters: {missing}")
            
        if not isinstance(config['max_size'], int):
            raise ValueError("logging.max_size must be an integer")
        if not isinstance(config['backup_count'], int):
            raise ValueError("logging.backup_count must be an integer")
            
    @staticmethod
    def validate_cache(config: Optional[Dict[str, Any]]) -> None:
        """Validate cache configuration if present."""
        if config is None:
            return
            
        if 'enabled' not in config:
            raise ValueError("cache.enabled is required")
        if 'provider' not in config:
            raise ValueError("cache.provider is required")
            
        try:
            CacheProvider(config['provider'])
        except ValueError:
            raise ValueError(f"Invalid cache provider: {config['provider']}")
            
    @staticmethod
    def validate_monitoring(config: Optional[Dict[str, Any]]) -> None:
        """Validate monitoring configuration if present."""
        if config is None:
            return
            
        if 'enabled' not in config:
            raise ValueError("monitoring.enabled is required")
        if 'metrics' not in config:
            raise ValueError("monitoring.metrics is required")
        if 'alert_thresholds' not in config:
            raise ValueError("monitoring.alert_thresholds is required")
            
        if not isinstance(config['metrics'], list):
            raise ValueError("monitoring.metrics must be a list")
        if not isinstance(config['alert_thresholds'], dict):
            raise ValueError("monitoring.alert_thresholds must be a dictionary")

class ConfigLoader:
    """Loads and parses the configuration file."""
    
    def __init__(self, config_path: str):
        """Initialize the config loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        self.validator = ConfigValidator()
        
    def load(self) -> StockRadarConfig:
        """Load and validate the configuration.
        
        Returns:
            StockRadarConfig object containing the validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Validate all sections
        self.validator.validate_data_source(config_dict['data_source'])
        self.validator.validate_factors(config_dict['factors'])
        self.validator.validate_strategies(config_dict['strategies'])
        self.validator.validate_output(config_dict['output'])
        self.validator.validate_logging(config_dict['logging'])
        self.validator.validate_cache(config_dict.get('cache'))
        self.validator.validate_monitoring(config_dict.get('monitoring'))
        
        # Parse data source config
        data_source = DataSourceConfig(
            provider=DataProvider(config_dict['data_source']['provider']),
            symbols=config_dict['data_source']['params']['symbols'],
            start_date=datetime.strptime(config_dict['data_source']['params']['start_date'], '%Y-%m-%d'),
            end_date=datetime.strptime(config_dict['data_source']['params']['end_date'], '%Y-%m-%d'),
            interval=config_dict['data_source']['params']['interval']
        )
        
        # Parse factor configs
        factors = [
            FactorConfig(
                class_path=f['class'],
                params=f['params']
            )
            for f in config_dict['factors']
        ]
        
        # Parse strategy configs
        strategies = [
            StrategyConfig(
                class_path=s['class'],
                params=s['params']
            )
            for s in config_dict['strategies']
        ]
        
        # Parse output config
        output = OutputConfig(
            path=config_dict['output']['path'],
            format=config_dict['output']['format'],
            include_metadata=config_dict['output']['include_metadata']
        )
        
        # Parse logging config
        logging_config = LoggingConfig(
            level=config_dict['logging']['level'],
            file=config_dict['logging']['file'],
            max_size=config_dict['logging']['max_size'],
            backup_count=config_dict['logging']['backup_count']
        )
        
        # Parse optional cache config
        cache_config = None
        if 'cache' in config_dict:
            cache_config = CacheConfig(
                enabled=config_dict['cache']['enabled'],
                provider=CacheProvider(config_dict['cache']['provider']),
                params=config_dict['cache']['params']
            )
            
        # Parse optional monitoring config
        monitoring_config = None
        if 'monitoring' in config_dict:
            monitoring_config = MonitoringConfig(
                enabled=config_dict['monitoring']['enabled'],
                metrics=config_dict['monitoring']['metrics'],
                alert_thresholds=config_dict['monitoring']['alert_thresholds']
            )
            
        return StockRadarConfig(
            data_source=data_source,
            factors=factors,
            strategies=strategies,
            output=output,
            logging=logging_config,
            cache=cache_config,
            monitoring=monitoring_config
        ) 
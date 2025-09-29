"""
Storage Manager for trading data
Provides a unified interface to store and retrieve data from different storage backends
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Add necessary imports for storage backends
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcp_storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

from .config_loader import load_storage_config

logger = logging.getLogger(__name__)

class StorageBackendBase:
    """Base class for all storage backends"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the storage backend with config"""
        self.config = config
    
    def save_data(self, data: pd.DataFrame, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save data to storage"""
        raise NotImplementedError("Subclasses must implement save_data")
    
    def get_data(self, dataset_name: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from storage"""
        raise NotImplementedError("Subclasses must implement get_data")
    
    def list_datasets(self) -> List[str]:
        """List available datasets"""
        raise NotImplementedError("Subclasses must implement list_datasets")
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset"""
        raise NotImplementedError("Subclasses must implement get_versions")

class LocalStorageBackend(StorageBackendBase):
    """Storage backend that uses local filesystem"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the local storage backend"""
        super().__init__(config)
        self.base_path = Path(config.get('base_path', 'data/storage'))
        
        # Create the directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        logger.info(f"Initialized local storage at {self.base_path}")
    
    def save_data(self, data: pd.DataFrame, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save data to local storage"""
        try:
            # Create dataset directory if it doesn't exist
            dataset_dir = self.base_path / dataset_name
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Generate timestamp-based version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the data
            file_path = dataset_dir / f"{version}.parquet"
            data.to_parquet(file_path)
            
            # Save metadata if provided
            if metadata:
                import json
                meta_path = dataset_dir / f"{version}_metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            
            logger.info(f"Saved dataset {dataset_name} version {version} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to local storage: {str(e)}")
            return False
    
    def get_data(self, dataset_name: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from local storage"""
        try:
            dataset_dir = self.base_path / dataset_name
            
            if not dataset_dir.exists():
                logger.warning(f"Dataset {dataset_name} not found")
                return None
            
            # If version is not specified, get the latest version
            if not version:
                versions = self.get_versions(dataset_name)
                if not versions:
                    logger.warning(f"No versions found for dataset {dataset_name}")
                    return None
                version = versions[-1]  # Get latest version
            
            # Construct file path
            file_path = dataset_dir / f"{version}.parquet"
            if not file_path.exists():
                logger.warning(f"Version {version} not found for dataset {dataset_name}")
                return None
            
            # Load the data
            data = pd.read_parquet(file_path)
            logger.info(f"Loaded dataset {dataset_name} version {version} from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data from local storage: {str(e)}")
            return None
    
    def list_datasets(self) -> List[str]:
        """List available datasets in local storage"""
        try:
            if not self.base_path.exists():
                logger.warning(f"Storage directory {self.base_path} does not exist")
                return []
            
            # Get all directories in the base path
            datasets = [d.name for d in self.base_path.iterdir() if d.is_dir()]
            return sorted(datasets)
            
        except Exception as e:
            logger.error(f"Error listing datasets in local storage: {str(e)}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset in local storage"""
        try:
            dataset_dir = self.base_path / dataset_name
            
            if not dataset_dir.exists():
                logger.warning(f"Dataset {dataset_name} not found")
                return []
            
            # Get all parquet files in the dataset directory
            files = [f.name for f in dataset_dir.iterdir() if f.is_file() and f.suffix == '.parquet']
            
            # Extract versions from filenames
            versions = sorted([f.split('.')[0] for f in files])
            return versions
            
        except Exception as e:
            logger.error(f"Error getting versions for dataset {dataset_name}: {str(e)}")
            return []

class S3StorageBackend(StorageBackendBase):
    """Storage backend that uses Amazon S3"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the S3 storage backend"""
        super().__init__(config)
        
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage backend")
        
        self.bucket_name = config.get('bucket_name')
        if not self.bucket_name:
            raise ValueError("bucket_name is required for S3 storage backend")
        
        self.prefix = config.get('prefix', '')
        
        # Create S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key'),
            region_name=config.get('region_name', 'us-east-1')
        )
        
        logger.info(f"Initialized S3 storage with bucket {self.bucket_name}")
    
    def save_data(self, data: pd.DataFrame, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save data to S3 storage"""
        try:
            # Generate timestamp-based version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the S3 key
            base_key = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            data_key = f"{base_key}{version}.parquet"
            
            # Convert DataFrame to bytes
            import io
            buffer = io.BytesIO()
            data.to_parquet(buffer)
            buffer.seek(0)
            
            # Upload to S3
            self.s3_client.upload_fileobj(buffer, self.bucket_name, data_key)
            
            # Save metadata if provided
            if metadata:
                import json
                meta_key = f"{base_key}{version}_metadata.json"
                meta_bytes = json.dumps(metadata).encode('utf-8')
                meta_buffer = io.BytesIO(meta_bytes)
                self.s3_client.upload_fileobj(meta_buffer, self.bucket_name, meta_key)
            
            logger.info(f"Saved dataset {dataset_name} version {version} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to S3: {str(e)}")
            return False
    
    def get_data(self, dataset_name: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from S3 storage"""
        try:
            # Create the base S3 key
            base_key = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            
            # If version is not specified, get the latest version
            if not version:
                versions = self.get_versions(dataset_name)
                if not versions:
                    logger.warning(f"No versions found for dataset {dataset_name}")
                    return None
                version = versions[-1]  # Get latest version
            
            # Create the full S3 key
            data_key = f"{base_key}{version}.parquet"
            
            # Download from S3
            import io
            buffer = io.BytesIO()
            self.s3_client.download_fileobj(self.bucket_name, data_key, buffer)
            buffer.seek(0)
            
            # Load the data
            data = pd.read_parquet(buffer)
            logger.info(f"Loaded dataset {dataset_name} version {version} from S3")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data from S3: {str(e)}")
            return None
    
    def list_datasets(self) -> List[str]:
        """List available datasets in S3 storage"""
        try:
            # Create the prefix for listing
            prefix = f"{self.prefix}/" if self.prefix else ""
            
            # List objects with the given prefix
            result = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, Delimiter='/')
            
            # Extract dataset names from common prefixes
            datasets = []
            for cp in result.get('CommonPrefixes', []):
                prefix_path = cp.get('Prefix')
                if prefix_path:
                    # Extract the dataset name from the prefix
                    if self.prefix:
                        # Remove the base prefix and trailing slash
                        dataset = prefix_path[len(prefix):].rstrip('/')
                    else:
                        dataset = prefix_path.rstrip('/')
                    datasets.append(dataset)
            
            return sorted(datasets)
            
        except Exception as e:
            logger.error(f"Error listing datasets in S3: {str(e)}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset in S3 storage"""
        try:
            # Create the prefix for listing
            prefix = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            
            # List objects with the given prefix
            result = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            
            # Extract versions from object keys
            versions = []
            for obj in result.get('Contents', []):
                key = obj.get('Key')
                if key and key.endswith('.parquet'):
                    # Extract the version from the key
                    # Key format is {prefix}/{dataset_name}/{version}.parquet
                    filename = key.split('/')[-1]
                    version = filename.split('.')[0]
                    versions.append(version)
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"Error getting versions for dataset {dataset_name} in S3: {str(e)}")
            return []

class GCSStorageBackend(StorageBackendBase):
    """Storage backend that uses Google Cloud Storage"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the GCS storage backend"""
        super().__init__(config)
        
        if not GCP_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS storage backend")
        
        self.bucket_name = config.get('bucket_name')
        if not self.bucket_name:
            raise ValueError("bucket_name is required for GCS storage backend")
        
        self.prefix = config.get('prefix', '')
        
        # Create GCS client
        self.gcs_client = gcp_storage.Client()
        self.bucket = self.gcs_client.bucket(self.bucket_name)
        
        logger.info(f"Initialized GCS storage with bucket {self.bucket_name}")
    
    def save_data(self, data: pd.DataFrame, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save data to Google Cloud Storage"""
        try:
            # Generate timestamp-based version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the GCS blob name
            base_path = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            blob_name = f"{base_path}{version}.parquet"
            
            # Convert DataFrame to bytes
            import io
            buffer = io.BytesIO()
            data.to_parquet(buffer)
            buffer.seek(0)
            
            # Upload to GCS
            blob = self.bucket.blob(blob_name)
            blob.upload_from_file(buffer, content_type='application/octet-stream')
            
            # Save metadata if provided
            if metadata:
                import json
                meta_blob_name = f"{base_path}{version}_metadata.json"
                meta_blob = self.bucket.blob(meta_blob_name)
                meta_blob.upload_from_string(
                    json.dumps(metadata),
                    content_type='application/json'
                )
            
            logger.info(f"Saved dataset {dataset_name} version {version} to GCS")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to GCS: {str(e)}")
            return False
    
    def get_data(self, dataset_name: str, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve data from Google Cloud Storage"""
        try:
            # Create the base blob path
            base_path = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            
            # If version is not specified, get the latest version
            if not version:
                versions = self.get_versions(dataset_name)
                if not versions:
                    logger.warning(f"No versions found for dataset {dataset_name}")
                    return None
                version = versions[-1]  # Get latest version
            
            # Create the full blob name
            blob_name = f"{base_path}{version}.parquet"
            
            # Download from GCS
            import io
            blob = self.bucket.blob(blob_name)
            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            
            # Load the data
            data = pd.read_parquet(buffer)
            logger.info(f"Loaded dataset {dataset_name} version {version} from GCS")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data from GCS: {str(e)}")
            return None
    
    def list_datasets(self) -> List[str]:
        """List available datasets in Google Cloud Storage"""
        try:
            # Create the prefix for listing
            prefix = f"{self.prefix}/" if self.prefix else ""
            
            # List blobs with the given prefix
            blobs = list(self.bucket.list_blobs(prefix=prefix, delimiter='/'))
            
            # Get the prefixes (which correspond to datasets)
            prefixes = self.bucket.list_blobs(prefix=prefix, delimiter='/')
            datasets = []
            
            for prefix in prefixes.prefixes:
                # Extract the dataset name from the prefix
                if self.prefix:
                    # Remove the base prefix and trailing slash
                    dataset = prefix[len(self.prefix)+1:].rstrip('/')
                else:
                    dataset = prefix.rstrip('/')
                datasets.append(dataset)
            
            return sorted(datasets)
            
        except Exception as e:
            logger.error(f"Error listing datasets in GCS: {str(e)}")
            return []
    
    def get_versions(self, dataset_name: str) -> List[str]:
        """Get available versions for a dataset in Google Cloud Storage"""
        try:
            # Create the prefix for listing
            prefix = f"{self.prefix}/{dataset_name}/" if self.prefix else f"{dataset_name}/"
            
            # List blobs with the given prefix
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            # Extract versions from blob names
            versions = []
            for blob in blobs:
                if blob.name.endswith('.parquet'):
                    # Extract the version from the blob name
                    # Blob name format is {prefix}/{dataset_name}/{version}.parquet
                    filename = blob.name.split('/')[-1]
                    version = filename.split('.')[0]
                    versions.append(version)
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"Error getting versions for dataset {dataset_name} in GCS: {str(e)}")
            return []

class StorageManager:
    """
    Manages data storage operations for different backends and data types.
    
    Supports:
    - Local file system storage
    - (Placeholder for S3, GCS, etc. implementations)
    
    Handles:
    - Saving and loading data
    - Data retention policies
    - Different data formats
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the storage manager with configuration.
        
        Args:
            config_path (str, optional): Path to storage config file.
                                        If None, uses default config.
        """
        self.config = load_storage_config(config_path)
        self.backend = self.config['default_backend']
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the storage backend based on configuration."""
        if self.backend == 'local':
            self.base_path = self.config['backends']['local']['base_path']
            # Convert to absolute path if relative
            if not os.path.isabs(self.base_path):
                # Use project root for relative paths
                from backend.storage_integration.config_loader import get_project_root
                project_root = get_project_root()
                self.base_path = os.path.join(project_root, self.base_path)
            
            os.makedirs(self.base_path, exist_ok=True)
            logger.info(f"Initialized local storage backend at {self.base_path}")
        elif self.backend == 's3':
            # Placeholder for S3 backend initialization
            logger.info("S3 backend initialization not yet implemented")
            raise NotImplementedError("S3 backend not implemented yet")
        elif self.backend == 'gcs':
            # Placeholder for GCS backend initialization
            logger.info("GCS backend initialization not yet implemented")
            raise NotImplementedError("GCS backend not implemented yet")
        else:
            raise ValueError(f"Unsupported storage backend: {self.backend}")
    
    def _get_data_type_config(self, data_type):
        """Get configuration for a specific data type."""
        if data_type not in self.config['data_types']:
            raise ValueError(f"Unknown data type: {data_type}")
        return self.config['data_types'][data_type]
    
    def save(self, data, data_type, identifier, metadata=None):
        """
        Save data to storage.
        
        Args:
            data (pd.DataFrame): Data to save
            data_type (str): Type of data (market_data, backtest_results, models)
            identifier (str): Unique identifier for the data
            metadata (dict, optional): Additional metadata to store
            
        Returns:
            str: Path where data was saved
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        dt_config = self._get_data_type_config(data_type)
        format = dt_config['format']
        compression = dt_config.get('compression', None)
        
        # Generate path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{identifier}_{timestamp}"
        
        if format == 'parquet':
            filename = f"{filename}.parquet"
            file_path = self._get_storage_path(data_type, filename)
            
            # Save metadata as DataFrame attributes
            if metadata:
                for key, value in metadata.items():
                    data.attrs[key] = value
            
            # Save to parquet
            data.to_parquet(file_path, compression=compression)
            logger.info(f"Saved {data_type} data to {file_path}")
            
            # Apply retention policy
            self._apply_retention_policy(data_type)
            
            return file_path
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load(self, data_type, identifier=None, date_range=None, latest=False):
        """
        Load data from storage.
        
        Args:
            data_type (str): Type of data to load
            identifier (str, optional): Specific identifier to load
            date_range (tuple, optional): (start_date, end_date) to filter by
            latest (bool): Whether to return only the latest version
            
        Returns:
            pd.DataFrame or dict: Loaded data
        """
        dt_config = self._get_data_type_config(data_type)
        format = dt_config['format']
        
        # Get matching files
        data_dir = os.path.join(self.base_path, data_type)
        if not os.path.exists(data_dir):
            logger.warning(f"No data directory for {data_type}")
            return pd.DataFrame() if not latest else None
        
        files = os.listdir(data_dir)
        
        # Filter by identifier
        if identifier:
            files = [f for f in files if f.startswith(f"{identifier}_")]
        
        if not files:
            logger.warning(f"No files found for {data_type}" + 
                          (f" with identifier {identifier}" if identifier else ""))
            return pd.DataFrame() if not latest else None
        
        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        
        if latest:
            # Just return latest file
            file_path = os.path.join(data_dir, files[0])
            if format == 'parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Load all matching files
            result = {}
            for file in files:
                file_path = os.path.join(data_dir, file)
                if format == 'parquet':
                    result[file] = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            return result
    
    def _get_storage_path(self, data_type, filename):
        """Generate a storage path for a file."""
        data_dir = os.path.join(self.base_path, data_type)
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, filename)
    
    def _apply_retention_policy(self, data_type):
        """Delete files older than retention period."""
        dt_config = self._get_data_type_config(data_type)
        retention_days = dt_config.get('retention_days', 0)
        
        if retention_days <= 0:
            # No retention policy
            return
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Get files in the data type directory
        data_dir = os.path.join(self.base_path, data_type)
        if not os.path.exists(data_dir):
            return
        
        files = os.listdir(data_dir)
        for file in files:
            file_path = os.path.join(data_dir, file)
            
            # Get file modification time
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Check if file is older than retention period
            if file_time < cutoff_date:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted {file_path} due to retention policy")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
    
    def list_data(self, data_type, identifier=None):
        """
        List available data files.
        
        Args:
            data_type (str): Type of data to list
            identifier (str, optional): Filter by identifier
            
        Returns:
            list: Available data files
        """
        data_dir = os.path.join(self.base_path, data_type)
        if not os.path.exists(data_dir):
            return []
        
        files = os.listdir(data_dir)
        
        # Filter by identifier
        if identifier:
            files = [f for f in files if f.startswith(f"{identifier}_")]
        
        # Sort by timestamp (newest first)
        files.sort(reverse=True)
        
        return files 
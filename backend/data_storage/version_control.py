"""Version control system for market data."""

import pandas as pd
import os
from datetime import datetime
import hashlib
import json
from typing import Optional, List, Dict, Any, Tuple
from . import DataStorageBase
import logging

class VersionControl:
    """Version control system for market data.
    
    This class manages data versioning and snapshots, tracking changes
    and metadata for each version. It works with any storage backend
    that implements the DataStorageBase interface.
    """
    
    def __init__(self, storage: DataStorageBase):
        """Initialize version control.
        
        Args:
            storage: Storage backend instance
        """
        self.storage = storage
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metadata storage
        self.metadata_cache = {}
        
    def _generate_version_id(self, data: pd.DataFrame, metadata: Dict) -> str:
        """Generate a unique version ID based on data content and metadata.
        
        Args:
            data: DataFrame to version
            metadata: Version metadata
            
        Returns:
            Version identifier string
        """
        # Create a hash of the data content
        data_hash = hashlib.sha256()
        data_hash.update(pd.util.hash_pandas_object(data).values.tobytes())
        
        # Add metadata to the hash
        metadata_str = json.dumps(metadata, sort_keys=True)
        data_hash.update(metadata_str.encode())
        
        # Use first 8 characters of hash as version ID
        return data_hash.hexdigest()[:8]
        
    def create_version(self,
                      data: pd.DataFrame,
                      dataset_name: str,
                      metadata: Optional[Dict] = None) -> Tuple[str, bool]:
        """Create a new version of a dataset.
        
        Args:
            data: DataFrame to version
            dataset_name: Name/identifier for the dataset
            metadata: Optional metadata for the version
            
        Returns:
            Tuple of (version_id, success)
        """
        try:
            # Basic validation that the data is not empty
            if data is None:
                self.logger.error(f"Cannot create version for {dataset_name}: Data is None")
                raise ValueError("Data cannot be None")
                
            if data.empty:
                # Provide more debug information for empty DataFrames
                self.logger.error(f"Cannot create version for {dataset_name}: DataFrame is empty. Columns: {data.columns.tolist()}")
                raise ValueError("Data cannot be empty")
                
            # Prepare metadata
            metadata = metadata or {}
            
            # Add standard metadata
            extra_metadata = self._extract_metadata(data, dataset_name)
            metadata.update(extra_metadata)
            
            # Generate version ID
            version_id = self._generate_version_id(data, metadata)
            
            # Store the metadata separately 
            self._store_metadata(dataset_name, version_id, metadata)
            
            # We'll use a simpler version name now
            version_name = f"{version_id}"
            
            # Save to storage
            success = self.storage.save_data(data, dataset_name, version_name)
            
            return version_id, success
            
        except Exception as e:
            self.logger.error(f"Error creating version for {dataset_name}: {str(e)}")
            return None, False
    
    def _store_metadata(self, dataset_name: str, version_id: str, metadata: Dict) -> None:
        """Store metadata for a version.
        
        Args:
            dataset_name: Name of the dataset
            version_id: Version identifier
            metadata: Metadata to store
        """
        # Cache the metadata in memory
        if dataset_name not in self.metadata_cache:
            self.metadata_cache[dataset_name] = {}
        
        self.metadata_cache[dataset_name][version_id] = metadata
        
        # Also store in a file for persistence
        try:
            # Create dataset directory if it doesn't exist
            if hasattr(self.storage, "_get_dataset_path"):
                dataset_dir = self.storage._get_dataset_path(dataset_name)
                metadata_dir = os.path.join(dataset_dir, 'metadata')
                os.makedirs(metadata_dir, exist_ok=True)
                
                # Write metadata to JSON file
                metadata_path = os.path.join(metadata_dir, f"{version_id}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
        except Exception as e:
            self.logger.warning(f"Could not persist metadata to file: {str(e)}")
    
    def _get_metadata(self, dataset_name: str, version_id: str) -> Dict:
        """Retrieve metadata for a version.
        
        Args:
            dataset_name: Name of the dataset
            version_id: Version identifier
            
        Returns:
            Version metadata dictionary
        """
        # First check in-memory cache
        if dataset_name in self.metadata_cache and version_id in self.metadata_cache[dataset_name]:
            return self.metadata_cache[dataset_name][version_id]
        
        # If not in cache, try to load from file
        try:
            if hasattr(self.storage, "_get_dataset_path"):
                dataset_dir = self.storage._get_dataset_path(dataset_name)
                metadata_path = os.path.join(dataset_dir, 'metadata', f"{version_id}.json")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # Update cache
                    if dataset_name not in self.metadata_cache:
                        self.metadata_cache[dataset_name] = {}
                    self.metadata_cache[dataset_name][version_id] = metadata
                    
                    return metadata
        except Exception as e:
            self.logger.warning(f"Could not load metadata from file: {str(e)}")
        
        return {}
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data structure based on the requirements.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError if invalid
        """
        if data is None or data.empty:
            raise ValueError("Data cannot be empty")
         
        # We no longer do strict validation here
        # Each storage backend should handle its own validation   
        return True
    
    def _extract_metadata(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Extract metadata from the data based on the dataset type.
        
        Args:
            data: DataFrame to extract metadata from
            dataset_name: Type of dataset
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'created_at': datetime.now().isoformat(),
            'num_records': len(data)
        }
        
        # Add dataset-specific metadata
        if dataset_name == "market_data" and all(col in data.columns for col in ['ticker', 'date']):
            # Market data specific metadata
            metadata.update({
                'tickers': sorted(data['ticker'].unique().tolist()),
                'date_range': [
                    data['date'].min().isoformat(),
                    data['date'].max().isoformat()
                ]
            })
        elif 'date' in data.columns:
            # Generic time series metadata
            metadata.update({
                'date_range': [
                    data['date'].min().isoformat(),
                    data['date'].max().isoformat()
                ]
            })
        
        return metadata
            
    def get_version(self,
                   dataset_name: str,
                   version_id: str,
                   tickers: Optional[List[str]] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, Dict]:
        """Get a specific version of a dataset.
        
        Args:
            dataset_name: Name/identifier for the dataset
            version_id: Version identifier
            tickers: Optional list of ticker symbols to load
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            Tuple of (data, metadata)
        """
        try:
            # Find all versions to match with version_id prefix
            versions = self.storage.get_versions(dataset_name)
            matching_versions = [v for v in versions if v.startswith(version_id)]
            
            if not matching_versions:
                raise ValueError(f"Version not found: {version_id}")
            
            # Use the first matching version
            version_name = matching_versions[0]
            
            # Get metadata from our dedicated storage
            metadata = self._get_metadata(dataset_name, version_id)
            
            # Load data
            data = self.storage.load_data(
                dataset_name,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                version=version_name
            )
            
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"Error getting version: {str(e)}")
            return pd.DataFrame(), {}
            
    def list_versions(self, dataset_name: str) -> List[Dict]:
        """List all versions of a dataset with their metadata.
        
        Args:
            dataset_name: Name/identifier for the dataset
            
        Returns:
            List of version metadata dictionaries
        """
        try:
            versions = []
            for version_id in self.storage.get_versions(dataset_name):
                # Get metadata for this version
                metadata = self._get_metadata(dataset_name, version_id)
                if metadata:
                    metadata['version_id'] = version_id
                    versions.append(metadata)
                
            return sorted(versions, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error listing versions: {str(e)}")
            return []
            
    def delete_version(self, dataset_name: str, version_id: str) -> bool:
        """Delete a specific version of a dataset.
        
        Args:
            dataset_name: Name/identifier for the dataset
            version_id: Version identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Find the version that starts with this ID
            versions = self.storage.get_versions(dataset_name)
            matching_versions = [v for v in versions if v.startswith(version_id)]
            
            if not matching_versions:
                raise ValueError(f"Version not found: {version_id}")
            
            # Delete data
            success = self.storage.delete_data(dataset_name, matching_versions[0])
            
            # Delete metadata
            try:
                if hasattr(self.storage, "_get_dataset_path"):
                    metadata_path = os.path.join(
                        self.storage._get_dataset_path(dataset_name),
                        'metadata',
                        f"{version_id}.json"
                    )
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                
                # Remove from cache
                if dataset_name in self.metadata_cache and version_id in self.metadata_cache[dataset_name]:
                    del self.metadata_cache[dataset_name][version_id]
            except Exception as e:
                self.logger.warning(f"Could not delete metadata file: {str(e)}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting version: {str(e)}")
            return False
            
    def compare_versions(self,
                        dataset_name: str,
                        version_id1: str,
                        version_id2: str) -> Dict:
        """Compare two versions of a dataset.
        
        Args:
            dataset_name: Name/identifier for the dataset
            version_id1: First version identifier
            version_id2: Second version identifier
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load both versions
            data1, meta1 = self.get_version(dataset_name, version_id1)
            data2, meta2 = self.get_version(dataset_name, version_id2)
            
            if data1.empty or data2.empty:
                raise ValueError("Could not load one or both versions")
                
            # Compare basic statistics
            comparison = {
                'version1': {
                    'id': version_id1,
                    'metadata': meta1,
                    'num_records': len(data1)
                },
                'version2': {
                    'id': version_id2,
                    'metadata': meta2,
                    'num_records': len(data2)
                },
                'differences': {
                    'records_diff': len(data2) - len(data1)
                }
            }
            
            # Add dataset-specific comparisons
            if dataset_name == "market_data" and 'ticker' in data1.columns and 'ticker' in data2.columns:
                comparison['differences'].update({
                    'tickers_added': list(set(data2['ticker'].unique()) - set(data1['ticker'].unique())),
                    'tickers_removed': list(set(data1['ticker'].unique()) - set(data2['ticker'].unique())),
                })
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing versions: {str(e)}")
            return {}

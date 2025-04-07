"""Version control system for market data."""

import pandas as pd
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
            # Validate data first
            self.storage.validate_data(data)
            
            # Prepare metadata
            metadata = metadata or {}
            metadata.update({
                'created_at': datetime.now().isoformat(),
                'num_records': len(data),
                'tickers': sorted(data['ticker'].unique().tolist()),
                'date_range': [
                    data['date'].min().isoformat(),
                    data['date'].max().isoformat()
                ]
            })
            
            # Generate version ID
            version_id = self._generate_version_id(data, metadata)
            
            # Save version metadata as part of the version name
            version_name = f"{version_id}_{json.dumps(metadata)}"
            
            # Save to storage
            success = self.storage.save_data(data, dataset_name, version_name)
            
            return version_id, success
            
        except Exception as e:
            self.logger.error(f"Error creating version: {str(e)}")
            return None, False
            
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
            # Find the full version name that starts with this ID
            versions = self.storage.get_versions(dataset_name)
            version_name = next(
                (v for v in versions if v.startswith(version_id)),
                None
            )
            
            if not version_name:
                raise ValueError(f"Version not found: {version_id}")
                
            # Extract metadata from version name
            metadata = json.loads(version_name[9:])  # Skip ID and underscore
            
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
            for version_name in self.storage.get_versions(dataset_name):
                try:
                    version_id = version_name[:8]
                    metadata = json.loads(version_name[9:])
                    metadata['version_id'] = version_id
                    versions.append(metadata)
                except:
                    continue
            return sorted(versions, key=lambda x: x['created_at'], reverse=True)
            
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
            # Find the full version name that starts with this ID
            versions = self.storage.get_versions(dataset_name)
            version_name = next(
                (v for v in versions if v.startswith(version_id)),
                None
            )
            
            if not version_name:
                raise ValueError(f"Version not found: {version_id}")
                
            return self.storage.delete_data(dataset_name, version_name)
            
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
                    'num_records': len(data1),
                    'tickers': sorted(data1['ticker'].unique().tolist()),
                    'date_range': [
                        data1['date'].min().isoformat(),
                        data1['date'].max().isoformat()
                    ]
                },
                'version2': {
                    'id': version_id2,
                    'metadata': meta2,
                    'num_records': len(data2),
                    'tickers': sorted(data2['ticker'].unique().tolist()),
                    'date_range': [
                        data2['date'].min().isoformat(),
                        data2['date'].max().isoformat()
                    ]
                },
                'differences': {
                    'records_diff': len(data2) - len(data1),
                    'tickers_added': list(set(data2['ticker'].unique()) - set(data1['ticker'].unique())),
                    'tickers_removed': list(set(data1['ticker'].unique()) - set(data2['ticker'].unique())),
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing versions: {str(e)}")
            return {}

#!/usr/bin/env python
"""
Test script for the fixed storage implementation.
This script tests storing data without the required market data columns.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_storage import StorageManager

def generate_non_standard_data():
    """Generate dataframes that don't match the market_data schema."""
    
    # Test case 1: DataFrame with MultiIndex including 'ticker' but not as a column
    idx = pd.MultiIndex.from_product(
        [['AAPL', 'MSFT', 'GOOGL'], pd.date_range('2023-01-01', periods=5)],
        names=['ticker', 'date']
    )
    df1 = pd.DataFrame(
        np.random.randn(15, 4),
        index=idx,
        columns=['value1', 'value2', 'value3', 'value4']
    )
    
    # Test case 2: DataFrame without any market data columns
    df2 = pd.DataFrame({
        'value1': np.random.randn(10),
        'value2': np.random.randn(10),
        'value3': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    })
    
    # Test case 3: Portfolio snapshot with only value and timestamp
    df3 = pd.DataFrame({
        'timestamp': [datetime.now()],
        'cash': [100000.0],
        'portfolio_value': [105000.0],
        'positions': ['{"AAPL": 100, "MSFT": 50}']
    })
    
    return {
        'multiindex_data': df1,
        'generic_data': df2,
        'portfolio_data': df3
    }

def test_storage():
    """Test storing non-standard data with our fixed implementation."""
    
    # Initialize storage manager
    storage_mgr = StorageManager()
    
    # Generate test data
    test_data = generate_non_standard_data()
    
    # Test storing each type
    results = {}
    
    print("\n1. Testing MultiIndex data storage:")
    multiindex_data = test_data['multiindex_data']
    print(f"Original data shape: {multiindex_data.shape}")
    print(f"Original data index: {multiindex_data.index.names}")
    print(multiindex_data.head())
    
    try:
        version_id = storage_mgr.store_market_data(
            multiindex_data, 
            metadata={'source': 'test', 'type': 'multiindex'}
        )
        print(f"✅ Stored MultiIndex data with version_id: {version_id}")
        results['multiindex'] = version_id
    except Exception as e:
        print(f"❌ Failed to store MultiIndex data: {e}")
    
    print("\n2. Testing generic data storage:")
    generic_data = test_data['generic_data']
    print(f"Original data shape: {generic_data.shape}")
    print(f"Original data columns: {generic_data.columns.tolist()}")
    print(generic_data.head())
    
    try:
        version_id = storage_mgr.store_market_data(
            generic_data, 
            metadata={'source': 'test', 'type': 'generic'}
        )
        print(f"✅ Stored generic data with version_id: {version_id}")
        results['generic'] = version_id
    except Exception as e:
        print(f"❌ Failed to store generic data: {e}")
    
    print("\n3. Testing portfolio data storage:")
    portfolio_data = test_data['portfolio_data']
    print(f"Original data shape: {portfolio_data.shape}")
    print(f"Original data columns: {portfolio_data.columns.tolist()}")
    print(portfolio_data.head())
    
    try:
        version_id = storage_mgr.store_market_data(
            portfolio_data, 
            metadata={'source': 'test', 'type': 'portfolio'}
        )
        print(f"✅ Stored portfolio data with version_id: {version_id}")
        results['portfolio'] = version_id
    except Exception as e:
        print(f"❌ Failed to store portfolio data: {e}")
    
    # Try retrieving the data
    print("\nRetrieving stored data:")
    
    for data_type, version_id in results.items():
        if version_id:
            try:
                stored_data = storage_mgr.get_market_data(version_id)
                print(f"\nRetrieved {data_type} data (version_id: {version_id}):")
                print(f"Shape: {stored_data.shape}")
                print(f"Columns: {stored_data.columns.tolist()}")
                print(stored_data.head())
            except Exception as e:
                print(f"❌ Failed to retrieve {data_type} data: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_storage() 
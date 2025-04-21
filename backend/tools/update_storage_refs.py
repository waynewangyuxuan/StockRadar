#!/usr/bin/env python
"""
Script to update storage implementation references in the codebase.

This script:
1. Finds all Python files that reference the old storage implementations
2. Updates them to use the consolidated StorageManager from data_storage
"""

import os
import re
import sys
from pathlib import Path

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent

def find_py_files_with_storage_imports(root_dir):
    """Find Python files that import storage modules."""
    result = []
    storage_import_patterns = [
        r'from\s+storage_integration\s+import',
        r'from\s+backend\.storage_integration\s+import',
        r'from\s+core\.storage_integration\s+import',
        r'from\s+backend\.core\.storage_integration\s+import',
        r'import\s+storage_integration',
        r'import\s+backend\.storage_integration',
    ]
    
    for path in Path(root_dir).rglob('*.py'):
        # Skip the data_storage directory itself
        if 'data_storage' in str(path):
            continue
            
        try:
            with open(path, 'r') as f:
                content = f.read()
                
            for pattern in storage_import_patterns:
                if re.search(pattern, content):
                    result.append(path)
                    break
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return result

def update_imports(file_path):
    """Update storage imports in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace imports
    replacements = [
        (r'from\s+storage_integration\s+import\s+StorageManager', 'from data_storage import StorageManager'),
        (r'from\s+backend\.storage_integration\s+import\s+StorageManager', 'from backend.data_storage import StorageManager'),
        (r'from\s+core\.storage_integration\s+import\s+StorageManager', 'from backend.data_storage import StorageManager'),
        (r'from\s+backend\.core\.storage_integration\s+import\s+StorageManager', 'from backend.data_storage import StorageManager'),
        (r'import\s+storage_integration', 'from backend.data_storage import StorageManager'),
        (r'import\s+backend\.storage_integration', 'from backend.data_storage import StorageManager'),
    ]
    
    updated_content = content
    changes_made = False
    
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, updated_content)
        if new_content != updated_content:
            changes_made = True
            updated_content = new_content
    
    if changes_made:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        print(f"Updated imports in {file_path}")
    else:
        print(f"No changes needed in {file_path}")

def main():
    root_dir = get_project_root()
    print(f"Looking for files with storage imports in {root_dir}")
    
    files = find_py_files_with_storage_imports(root_dir)
    print(f"Found {len(files)} files with storage imports")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        update_imports(file_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 
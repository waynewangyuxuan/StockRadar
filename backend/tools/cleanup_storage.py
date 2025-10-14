#!/usr/bin/env python
"""
Script to clean up redundant storage implementations.

This script:
1. Backs up the redundant files to a backup directory
2. Removes the redundant files
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent.parent

def backup_and_remove_files():
    """Backup and remove redundant storage implementation files."""
    project_root = get_project_root()
    
    # Define backup directory
    backup_dir = project_root / 'backup' / f'storage_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(backup_dir, exist_ok=True)
    
    # Define files to remove
    files_to_remove = [
        project_root / 'backend' / 'core' / 'storage_integration.py',
        project_root / 'backend' / 'storage_integration' / 'storage_manager.py',
        project_root / 'backend' / 'storage_integration' / 'config_loader.py',
    ]
    
    for file_path in files_to_remove:
        if file_path.exists():
            # Create backup directory structure
            rel_path = file_path.relative_to(project_root)
            backup_file_path = backup_dir / rel_path
            os.makedirs(backup_file_path.parent, exist_ok=True)
            
            # Backup the file
            print(f"Backing up {rel_path} to {backup_file_path}")
            shutil.copy2(file_path, backup_file_path)
            
            # Remove the file
            print(f"Removing {rel_path}")
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print(f"Backup completed to {backup_dir}")
    
    # Check if storage_integration directory is empty and remove it
    storage_integration_dir = project_root / 'backend' / 'storage_integration'
    if storage_integration_dir.exists():
        if list(storage_integration_dir.glob('*')):
            print(f"Directory {storage_integration_dir} still contains files, not removing")
        else:
            print(f"Removing empty directory {storage_integration_dir}")
            os.rmdir(storage_integration_dir)

def main():
    print("This script will back up and remove redundant storage implementation files.")
    response = input("Do you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        backup_and_remove_files()
        print("Cleanup completed successfully!")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main() 
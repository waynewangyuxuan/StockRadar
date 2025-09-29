#!/usr/bin/env python3
"""
StockRadar Cleanup Script

This script removes unnecessary files and directories to clean up the repository.
Run with: python cleanup.py [--dry-run]
"""

import os
import shutil
import argparse
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def find_files_to_clean(project_root):
    """Find files and directories that should be cleaned up."""
    cleanup_targets = {
        'log_files': [],
        'cache_dirs': [],
        'temp_files': [],
        'old_dirs': [],
        'backup_dirs': []
    }

    # Find log files (keep only the most recent 5)
    output_dir = project_root / 'output'
    backend_output_dir = project_root / 'backend' / 'output'

    for output_path in [output_dir, backend_output_dir]:
        if output_path.exists():
            log_files = list(output_path.glob('*.log'))
            log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            # Keep the 5 most recent logs, mark others for cleanup
            cleanup_targets['log_files'].extend(log_files[5:])

    # Find __pycache__ directories (excluding venv)
    for cache_dir in project_root.rglob('__pycache__'):
        if 'venv' not in str(cache_dir):
            cleanup_targets['cache_dirs'].append(cache_dir)

    # Find .pyc files (excluding venv)
    for pyc_file in project_root.rglob('*.pyc'):
        if 'venv' not in str(pyc_file):
            cleanup_targets['temp_files'].append(pyc_file)

    # Old/unnecessary directories
    old_dirs = [
        'backup',
        'stockradar.egg-info',
        'logs',  # Empty logs directory
        'live_trading_results',  # Old results
        'results',  # Old results
        'backtest_results',  # Old results (if not in backend)
    ]

    for dir_name in old_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            cleanup_targets['old_dirs'].append(dir_path)

    # Look for backup directories
    for backup_dir in project_root.rglob('*backup*'):
        if backup_dir.is_dir() and 'venv' not in str(backup_dir):
            cleanup_targets['backup_dirs'].append(backup_dir)

    return cleanup_targets

def format_size(size_bytes):
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def calculate_cleanup_size(cleanup_targets):
    """Calculate total size of files to be cleaned up."""
    total_size = 0

    for category, items in cleanup_targets.items():
        for item in items:
            if item.exists():
                if item.is_file():
                    total_size += item.stat().st_size
                elif item.is_dir():
                    for file_path in item.rglob('*'):
                        if file_path.is_file():
                            try:
                                total_size += file_path.stat().st_size
                            except (OSError, PermissionError):
                                pass

    return total_size

def perform_cleanup(cleanup_targets, dry_run=False):
    """Perform the actual cleanup."""
    total_cleaned = 0
    action = "Would delete" if dry_run else "Deleting"

    print(f"\n{'='*60}")
    print(f"{'DRY RUN - ' if dry_run else ''}CLEANUP SUMMARY")
    print(f"{'='*60}")

    for category, items in cleanup_targets.items():
        if not items:
            continue

        category_name = category.replace('_', ' ').title()
        print(f"\n📁 {category_name}:")

        for item in items:
            if item.exists():
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  {action}: {item.relative_to(get_project_root())} ({format_size(size)})")
                    if not dry_run:
                        item.unlink()
                    total_cleaned += size
                elif item.is_dir():
                    # Calculate directory size
                    dir_size = 0
                    file_count = 0
                    for file_path in item.rglob('*'):
                        if file_path.is_file():
                            try:
                                dir_size += file_path.stat().st_size
                                file_count += 1
                            except (OSError, PermissionError):
                                pass

                    print(f"  {action}: {item.relative_to(get_project_root())}/ ({file_count} files, {format_size(dir_size)})")
                    if not dry_run:
                        shutil.rmtree(item, ignore_errors=True)
                    total_cleaned += dir_size

    print(f"\n{'='*60}")
    total_action = "Would free up" if dry_run else "Freed up"
    print(f"🎉 {total_action}: {format_size(total_cleaned)}")
    print(f"{'='*60}")

    return total_cleaned

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up StockRadar project files")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    args = parser.parse_args()

    project_root = get_project_root()
    print(f"🧹 StockRadar Cleanup Script")
    print(f"📂 Project root: {project_root}")

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")

    # Find cleanup targets
    print("\n📋 Scanning for cleanup targets...")
    cleanup_targets = find_files_to_clean(project_root)

    # Calculate total size
    total_size = calculate_cleanup_size(cleanup_targets)

    if total_size == 0:
        print("✨ Nothing to clean up! Project is already clean.")
        return

    print(f"🗂️  Found {format_size(total_size)} of cleanup targets")

    # Show summary
    for category, items in cleanup_targets.items():
        if items:
            category_name = category.replace('_', ' ').title()
            print(f"  • {category_name}: {len(items)} items")

    if not args.dry_run:
        response = input(f"\n❓ Proceed with cleanup? This will free up {format_size(total_size)} (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cleanup cancelled.")
            return

    # Perform cleanup
    perform_cleanup(cleanup_targets, dry_run=args.dry_run)

    if args.dry_run:
        print("\n💡 To actually perform the cleanup, run: python cleanup.py")
    else:
        print("\n✅ Cleanup completed successfully!")

if __name__ == '__main__':
    main()
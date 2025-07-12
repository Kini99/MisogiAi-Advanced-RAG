#!/usr/bin/env python3
"""
Test runner for Strategic Decision Engine.
This script provides various options for running tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Strategic Decision Engine")
    parser.add_argument(
        '--type', 
        choices=['unit', 'integration', 'api', 'all'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Run with verbose output'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run fast tests only (skip slow tests)'
    )
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install test dependencies first'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean test artifacts before running'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Install dependencies if requested
    if args.install:
        if not run_command([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], "Installing dependencies"):
            return 1
    
    # Clean test artifacts if requested
    if args.clean:
        artifacts = ['htmlcov', 'coverage.xml', '.coverage', '.pytest_cache', '__pycache__']
        for artifact in artifacts:
            if os.path.exists(artifact):
                if os.path.isdir(artifact):
                    subprocess.run(['rm', '-rf', artifact], check=True)
                else:
                    os.remove(artifact)
        print("Cleaned test artifacts")
    
    # Build pytest command
    cmd = [sys.executable, '-m', 'pytest']
    
    # Add test type filters
    if args.type == 'unit':
        cmd.extend(['-m', 'unit'])
    elif args.type == 'integration':
        cmd.extend(['-m', 'integration'])
    elif args.type == 'api':
        cmd.extend(['-m', 'api'])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            '--cov=backend',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml'
        ])
    
    # Add verbose output if requested
    if args.verbose:
        cmd.append('-v')
    
    # Skip slow tests if fast mode
    if args.fast:
        cmd.extend(['-m', 'not slow'])
    
    # Add test directory
    cmd.append('tests/')
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if success and args.coverage:
        print("\n" + "="*60)
        print("Coverage report generated:")
        print("- Terminal: (shown above)")
        print("- HTML: htmlcov/index.html")
        print("- XML: coverage.xml")
        print("="*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 
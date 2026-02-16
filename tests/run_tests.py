#!/usr/bin/env python3
"""
Main testing script for the gene PD pipeline.

Provides interactive menu for running different test scenarios:
1. Multi-model comparison on test set
2. Without summary comparison
3. Pipeline variation comparison
4. Option 3 multi-model comparison (recommended)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pd_pipeline.testing import (
    run_model_comparison_on_test_set,
    run_without_summary_comparison,
    run_pipeline_variation_comparison,
    run_option3_model_comparison,
)


def main():
    """Interactive test menu."""
    print("=" * 80)
    print("Gene PD Pipeline - Testing Framework")
    print("=" * 80)
    print("\nSelect test to run:")
    print("1. Multi-model comparison (original)")
    print("2. Without summary step (Sonnet 4.5 only)")
    print("3. Pipeline variation comparison")
    print("4. Option 3 multi-model comparison (WINNER!)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-4): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 80)
        print("Running multi-model comparison on test set...")
        print("=" * 80 + "\n")
        run_model_comparison_on_test_set()
        
    elif choice == "2":
        print("\n" + "=" * 80)
        print("Running without summary comparison...")
        print("=" * 80 + "\n")
        run_without_summary_comparison()
        
    elif choice == "3":
        print("\n" + "=" * 80)
        print("Running pipeline variation comparison...")
        print("=" * 80 + "\n")
        run_pipeline_variation_comparison()
        
    elif choice == "4":
        response = input("\nTest Option 3 across multiple models? (yes/no): ")
        if response.lower() in ["yes", "y"]:
            print("\n" + "=" * 80)
            print("Running Option 3 multi-model comparison...")
            print("=" * 80 + "\n")
            run_option3_model_comparison()
        else:
            print("Testing cancelled.")
            
    elif choice == "0":
        print("Exiting...")
        sys.exit(0)
        
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)


if __name__ == "__main__":
    main()

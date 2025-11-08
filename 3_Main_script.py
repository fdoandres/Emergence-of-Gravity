#!/usr/bin/env python3
"""
Main script for Dissipative Higgs Framework Analysis
Complete pipeline for PRD submission
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add scripts directory to path
sys.path.append('scripts')

# Import custom modules
from generate_figures import generate_all_figures
from generate_tables import generate_all_tables
from run_mcmc_analysis import run_complete_mcmc_analysis

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'figures', 'tables', 'latex', 'scripts']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Directory structure created")

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Dissipative Higgs Framework Analysis")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    try:
        # 1. Run MCMC analysis
        print("\n📊 Running MCMC analysis...")
        mcmc_results = run_complete_mcmc_analysis()
        
        # 2. Generate figures
        print("\n🎨 Generating figures...")
        figures = generate_all_figures(mcmc_results)
        
        # 3. Generate tables
        print("\n📋 Generating tables...")
        tables = generate_all_tables(mcmc_results)
        
        # 4. Summary
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("\n📊 Results Summary:")
        print(f"   • Figures generated: {len(figures)}")
        print(f"   • Tables generated: {len(tables)}")
        print(f"   • Optimal parameters: {len(mcmc_results.get('optimal_params', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
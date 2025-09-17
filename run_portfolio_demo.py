#!/usr/bin/env python3
"""
Fantasy Football Portfolio Demo Launcher

This script provides multiple ways to explore the portfolio depending on
available dependencies. It automatically detects what's available and 
runs the most comprehensive demo possible.
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check which demo dependencies are available."""
    deps = {
        'numpy': False,
        'pandas': False,
        'sklearn': False,
        'lightgbm': False,
        'ngboost': False
    }
    
    for dep in deps:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            pass
    
    return deps


def print_banner():
    """Print the portfolio banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      FANTASY FOOTBALL PORTFOLIO DEMO                         ║
║                                                                               ║
║  Advanced Draft Optimization & Machine Learning                              ║
║  Showcasing Data Science, Optimization, and Software Engineering             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_simple_demo():
    """Run the dependency-free demo."""
    print("🚀 Running Simple Demo (No Dependencies Required)")
    print("=" * 60)
    print("This demo showcases:")
    print("• Algorithm design and optimization concepts")
    print("• Statistical analysis and VORP calculations")
    print("• Monte Carlo simulation principles")
    print("• Clean, documented code structure")
    print()
    
    try:
        result = subprocess.run([sys.executable, 'simple_demo.py'], 
                              capture_output=False, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running simple demo: {e}")
        return False


def run_full_demo():
    """Run the full demo with all dependencies."""
    print("🚀 Running Full Portfolio Demo (All Dependencies Available)")
    print("=" * 60)
    print("This demo showcases:")
    print("• Complete ML pipeline with NGBoost and LightGBM")
    print("• Advanced optimization with MILP solver integration")
    print("• Comprehensive feature engineering")
    print("• Production-ready model validation")
    print()
    
    try:
        result = subprocess.run([sys.executable, 'demo_draft_optimization.py'], 
                              capture_output=False, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running full demo: {e}")
        return False


def run_basic_demo():
    """Run demo with basic dependencies (numpy/pandas only)."""
    print("🚀 Running Basic Demo (NumPy/Pandas Available)")
    print("=" * 60)
    print("This demo showcases:")
    print("• Data processing and analysis with pandas")
    print("• Numerical computations with numpy")
    print("• Statistical modeling concepts")
    print("• Vectorized algorithm implementations")
    print()
    
    # Import here to avoid dependency issues at module level
    try:
        import demo_draft_optimization
        demo_draft_optimization.main()
        return True
    except Exception as e:
        print(f"Error running basic demo: {e}")
        return False


def show_installation_guide():
    """Show installation instructions."""
    print()
    print("📦 INSTALLATION GUIDE")
    print("=" * 40)
    print("To see the full portfolio capabilities, install dependencies:")
    print()
    print("Option 1 - Full Installation:")
    print("  pip install -r requirements.txt")
    print()
    print("Option 2 - Basic Installation:")
    print("  pip install numpy pandas")
    print()
    print("Option 3 - Quick Demo (No Installation):")
    print("  python simple_demo.py")
    print()
    print("Dependencies:")
    print("• numpy, pandas: Core data processing")
    print("• scikit-learn: Machine learning infrastructure") 
    print("• lightgbm: Gradient boosting models")
    print("• ngboost: Probabilistic predictions")
    print("• optuna: Hyperparameter optimization")
    print("• scipy: Statistical functions and optimization")


def show_portfolio_summary():
    """Show a summary of what's demonstrated."""
    print()
    print("🎯 PORTFOLIO HIGHLIGHTS")
    print("=" * 40)
    
    highlights = [
        "Advanced Machine Learning: NGBoost, LightGBM, Ensemble Methods",
        "Mathematical Optimization: MILP, Monte Carlo, Constraint Satisfaction", 
        "Statistical Analysis: VORP, Age Curves, Correlation Modeling",
        "Feature Engineering: 50+ features, rolling stats, efficiency metrics",
        "Software Engineering: Modular design, type hints, comprehensive docs",
        "Algorithm Design: Value-based optimization, multi-objective decisions",
        "Performance Engineering: Vectorization, caching, memory optimization",
        "Data Science: Model validation, cross-validation, uncertainty quantification"
    ]
    
    for highlight in highlights:
        print(f"✓ {highlight}")
    
    print()
    print("📁 Key Files:")
    print("• README.md - Complete project overview") 
    print("• docs/PORTFOLIO_OVERVIEW.md - Technical deep dive")
    print("• src/draft/portfolio_optimizer.py - Main optimization engine")
    print("• src/models/draft_ml.py - ML model implementations")
    print("• src/data/synthetic_data.py - Demo data generation")
    print("• simple_demo.py - Dependency-free concept demonstration")


def main():
    """Main demo launcher."""
    print_banner()
    
    # Check available dependencies
    deps = check_dependencies()
    
    print("🔍 Dependency Check:")
    print("-" * 20)
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{status} {dep}")
    print()
    
    # Determine best demo to run
    if all(deps.values()):
        print("🎉 All dependencies available! Running full portfolio demo...")
        success = run_full_demo()
    elif deps['numpy'] and deps['pandas']:
        print("📊 Basic dependencies available! Running enhanced demo...")
        success = run_basic_demo()
    else:
        print("🎯 Running concept demo (no dependencies required)...")
        success = run_simple_demo()
    
    if success:
        show_portfolio_summary()
    else:
        print("\n⚠️  Demo encountered issues. Trying simple demo as fallback...")
        run_simple_demo()
        show_portfolio_summary()
    
    show_installation_guide()
    
    print("\n" + "="*80)
    print("Thank you for exploring the Fantasy Football Portfolio!")
    print("For questions or opportunities, please reach out through GitHub.")
    print("="*80)


if __name__ == "__main__":
    main()
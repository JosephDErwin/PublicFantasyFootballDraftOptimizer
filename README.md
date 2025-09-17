# Fantasy Football: A Technical Portfolio

This repository contains a **portfolio-friendly version** of a sophisticated fantasy football project. It is **not a functioning draft assistant** but a detailed demonstration of advanced data science, machine learning, and optimization techniques. The code showcases the methodologies used to build a robust system for player evaluation and draft strategy.

## 🎯 Technical Highlights

### Optimization Engine

- **Mixed-Integer Linear Programming (MILP):** Demonstrates the use of a MILP solver to find optimal draft strategies under various constraints.
    
- **Monte Carlo Simulations:** Illustrates running thousands of simulations to build a probabilistic framework for decision-making.
    
- **Value Over Replacement Player (VORP):** Showcases an advanced statistical method for evaluating a player’s worth.
    

### Machine Learning Models

- **NGBoost Regression:** A highlight of this project is the use of a probabilistic model to not only predict fantasy points but also quantify the uncertainty of those predictions.
    
- **Advanced Feature Engineering:** Over 50 features, including rolling statistics, efficiency metrics, and contextual data, are engineered to capture complex player performance trends.
    
- **Hyperparameter Optimization:** Utilizes `Optuna` for automated and efficient model tuning.
    

### Advanced Analytics

- **Correlation Modeling:** Demonstrates how to model and use player performance correlations to build a well-diversified fantasy team.
    
- **Time-Aware Validation:** The project uses a time-series cross-validation approach to prevent data leakage and ensure model robustness.
    

---

## 🏗️ Architecture & Structure

The repository is organized to showcase a professional software engineering approach.

- `src/`: Contains the core Python scripts for data generation, modeling, and optimization.
    
- `docs/`: Includes markdown files that provide a deeper technical dive into the project's methodologies.
    
- `requirements.txt`: Lists the necessary dependencies to run the demonstration.
    

---

## 🔬 Methodologies Demonstrated

### Data Science

- **Feature Engineering:** Shows the process of creating complex, predictive features from raw data.
    
- **Model Validation:** Highlights the importance of time-aware cross-validation and backtesting to build reliable models.
    
- **Ensemble Methods:** Demonstrates combining multiple model types to improve prediction accuracy.
    

### Optimization

- **Linear/Integer Programming:** Applies mathematical optimization to solve a real-world, constrained problem.
    
- **Stochastic Programming:** Illustrates handling uncertainty in optimization by integrating probabilistic forecasts.
    

---

## 📋 Portfolio Notes

This repository has been specifically adapted for portfolio presentation.

### ✅ What's Included

- A clean, modular code architecture.
    
- Core algorithms for optimization and machine learning.
    
- Synthetic data generation to make the project immediately runnable.
    
- Comprehensive documentation that explains the technical details.
    

### ❌ What's Excluded

- Personal or league-specific data and API credentials.
    
- All live data fetching capabilities from external APIs (e.g., ESPN, Sleeper).
    
- Any live or proprietary data used in the original project.
    

### **This project is a static representation of a complex system. Its purpose is to demonstrate technical capabilities, not to provide a functioning service.**

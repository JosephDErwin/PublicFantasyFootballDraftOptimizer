#!/usr/bin/env python3
"""
Demonstration script for Fantasy Football ML Model Improvements

This script showcases the enhanced features and capabilities added to the
Fantasy Football player prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample player data for demonstration."""
    np.random.seed(42)
    
    players = ['Josh Allen', 'Derrick Henry', 'DeAndre Hopkins', 'Travis Kelce', 'Aaron Rodgers']
    positions = ['QB', 'RB', 'WR', 'TE', 'QB']
    teams = ['BUF', 'TEN', 'ARI', 'KC', 'GB']
    
    # Create sample game data
    data = []
    for i, (player, pos, team) in enumerate(zip(players, positions, teams)):
        for week in range(1, 18):
            date = datetime(2023, 9, 1) + timedelta(weeks=week-1)
            
            # Generate realistic stats based on position
            if pos == 'QB':
                passing_yards = max(0, np.random.normal(250, 80))
                passing_tds = max(0, np.random.poisson(1.8))
                rushing_yards = max(0, np.random.normal(30, 15))
                stats = {
                    'passingYards': passing_yards,
                    'passingTouchdowns': passing_tds,
                    'passingAttempts': max(20, np.random.normal(35, 8)),
                    'rushingYards': rushing_yards,
                    'rushingTouchdowns': np.random.poisson(0.3)
                }
            elif pos == 'RB':
                rushing_yards = max(0, np.random.normal(80, 40))
                receiving_yards = max(0, np.random.normal(25, 20))
                stats = {
                    'rushingYards': rushing_yards,
                    'rushingAttempts': max(5, np.random.normal(18, 6)),
                    'rushingTouchdowns': np.random.poisson(0.8),
                    'receivingYards': receiving_yards,
                    'receivingTargets': max(0, np.random.normal(4, 3)),
                    'receivingReceptions': max(0, np.random.normal(3, 2))
                }
            elif pos in ['WR', 'TE']:
                receiving_yards = max(0, np.random.normal(65, 35))
                targets = max(0, np.random.normal(8, 4))
                stats = {
                    'receivingYards': receiving_yards,
                    'receivingTargets': targets,
                    'receivingReceptions': max(0, targets * np.random.uniform(0.5, 0.8)),
                    'receivingTouchdowns': np.random.poisson(0.6)
                }
            else:
                stats = {}
            
            # Add common fields
            row = {
                'player_id': i,
                'name': player,
                'position': pos,
                'team': team,
                'opponent': np.random.choice(['NE', 'MIA', 'NYJ', 'BAL', 'CIN']),
                'season': 2023,
                'week_num': week,
                'date': date,
                'age': 25 + i,
                'experience': 3 + i,
                'height': 72 + np.random.randint(-3, 4),
                'weight': 200 + np.random.randint(-20, 40),
                'gamesPlayed': 1,
                'draft_pick': 20 + i * 10,
                'draft_grade': 85 - i * 2,
                **stats
            }
            data.append(row)
    
    return pd.DataFrame(data)

def demonstrate_feature_engineering():
    """Demonstrate enhanced feature engineering capabilities."""
    print("=" * 60)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} player-games")
    print(f"Players: {df['name'].unique()}")
    print(f"Original features: {len(df.columns)}")
    
    try:
        from src.models.feature_engineering import (
            add_position_specific_features,
            add_advanced_rolling_stats,
            add_consistency_metrics,
            calculate_target_share
        )
        
        print("\n1. Adding position-specific features...")
        df_enhanced = add_position_specific_features(df)
        new_features = set(df_enhanced.columns) - set(df.columns)
        print(f"   Added {len(new_features)} position-specific features")
        print(f"   Examples: {list(new_features)[:5]}")
        
        print("\n2. Adding advanced rolling statistics...")
        stat_cols = ['receivingYards', 'rushingYards', 'passingYards']
        df_enhanced = add_advanced_rolling_stats(df_enhanced, stat_cols)
        new_features = set(df_enhanced.columns) - set(df.columns)
        print(f"   Added {len(new_features)} rolling features")
        
        print("\n3. Adding consistency metrics...")
        df_enhanced = add_consistency_metrics(df_enhanced, stat_cols)
        new_features = set(df_enhanced.columns) - set(df.columns)
        print(f"   Added {len(new_features)} consistency features")
        
        print("\n4. Calculating target share...")
        df_enhanced = calculate_target_share(df_enhanced)
        
        print(f"\nFinal dataset: {len(df_enhanced.columns)} total features")
        
        # Show sample of enhanced features for one player
        player_sample = df_enhanced[df_enhanced['player_id'] == 2].iloc[-1]
        enhanced_features = [col for col in df_enhanced.columns 
                           if col not in df.columns and not pd.isna(player_sample[col])]
        
        print(f"\nSample enhanced features for {player_sample['name']}:")
        for feature in enhanced_features[:8]:
            value = player_sample[feature]
            print(f"   {feature}: {value:.3f}")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Feature engineering modules not available in this environment")

def demonstrate_external_sources():
    """Demonstrate external data source integration."""
    print("\n" + "=" * 60)
    print("EXTERNAL DATA SOURCES DEMONSTRATION")
    print("=" * 60)
    
    try:
        from src.data.external_sources import (
            get_stadium_info,
            print_data_source_info,
            FREE_DATA_SOURCES
        )
        
        print("1. Stadium Information Database:")
        teams = ['GB', 'DEN', 'NO', 'SEA', 'LV']
        for team in teams:
            info = get_stadium_info(team)
            dome_status = "Dome" if info.get('dome') else "Outdoor"
            altitude = f", Altitude: {info.get('altitude', 0)}ft" if info.get('altitude') else ""
            print(f"   {team}: {info.get('name', 'Unknown')} ({dome_status}){altitude}")
        
        print("\n2. Available Free Data Sources:")
        for category, sources in FREE_DATA_SOURCES.items():
            print(f"\n   {category.upper()}:")
            for source, description in sources.items():
                print(f"     • {source}: {description}")
        
        print("\n3. Weather Impact Example:")
        print("   For outdoor stadiums in winter:")
        print("     • Temperature < 40°F affects passing accuracy")
        print("     • Wind > 15 mph impacts kicking and long passes")
        print("     • Precipitation reduces offensive efficiency")
        
        print("\n4. Betting Lines Example:")
        print("   Game totals and spreads predict:")
        print("     • High totals (>50) = more passing opportunities")
        print("     • Large spreads = potential garbage time stats")
        print("     • Implied team totals = expected offensive output")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("External data modules not available in this environment")

def demonstrate_enhanced_modeling():
    """Demonstrate enhanced modeling approaches."""
    print("\n" + "=" * 60)
    print("ENHANCED MODELING DEMONSTRATION")
    print("=" * 60)
    
    print("1. Position-Specific Modeling:")
    print("   • QB models focus on passing efficiency and weather")
    print("   • RB models emphasize usage share and game script")
    print("   • WR/TE models prioritize target share and catch rate")
    print("   • Each position gets optimized hyperparameters")
    
    print("\n2. Ensemble Methods:")
    print("   • Stacking: Meta-learner combines base model predictions")
    print("   • Voting: Weighted average of diverse model types")
    print("   • Base models: LightGBM, Ridge, ElasticNet, NGBoost")
    print("   • Automated weight optimization")
    
    print("\n3. Time-Aware Cross-Validation:")
    print("   • Season-based splits respect temporal structure")
    print("   • Player-grouped splits avoid data leakage")
    print("   • Forward-looking validation simulates real usage")
    
    print("\n4. Uncertainty Quantification:")
    print("   • Prediction intervals from NGBoost")
    print("   • Ensemble variance estimates")
    print("   • Position-specific risk profiles")

def demonstrate_improvements_summary():
    """Provide a summary of all improvements."""
    print("\n" + "=" * 60)
    print("IMPROVEMENTS SUMMARY")
    print("=" * 60)
    
    improvements = {
        "Enhanced Feature Engineering": [
            "Advanced rolling statistics (momentum, consistency, volatility)",
            "Position-specific efficiency metrics",
            "Game context features (rest, travel, season timing)",
            "Interaction terms and prime age indicators"
        ],
        "External Data Integration": [
            "Weather data for outdoor games",
            "Betting lines for game script prediction",
            "Stadium effects and altitude factors",
            "Advanced statistics framework"
        ],
        "Enhanced Modeling": [
            "Position-specific models with tailored features",
            "Ensemble methods (stacking and voting)",
            "Time-aware cross-validation",
            "Uncertainty quantification"
        ],
        "Free Data Sources": [
            "OpenWeatherMap API (1000 calls/day)",
            "The Odds API (500 requests/month)",
            "Pro Football Reference scraping",
            "ESPN and NFL.com statistics"
        ]
    }
    
    for category, items in improvements.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")
    
    print(f"\n{'='*60}")
    print("EXPECTED BENEFITS")
    print(f"{'='*60}")
    
    benefits = [
        "Improved prediction accuracy through better feature utilization",
        "Position-specific insights and risk assessment",
        "Weather and game script context integration",
        "Robust ensemble predictions with uncertainty bounds",
        "Identification of player trends and consistency patterns",
        "Better handling of external factors (rest, travel, weather)"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")

def main():
    """Run the complete demonstration."""
    print("Fantasy Football ML Model Improvements Demonstration")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    demonstrate_feature_engineering()
    demonstrate_external_sources()
    demonstrate_enhanced_modeling()
    demonstrate_improvements_summary()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Test enhanced features on historical data")
    print("2. Set up external API keys for weather and betting data")
    print("3. Compare model performance against baseline")
    print("4. Implement gradual rollout with A/B testing")
    print("5. Monitor feature importance and model drift")
    
    print("\nFor detailed documentation, see: docs/ML_MODEL_IMPROVEMENTS.md")

if __name__ == "__main__":
    main()
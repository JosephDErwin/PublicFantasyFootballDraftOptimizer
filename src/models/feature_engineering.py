"""
Enhanced feature engineering for Fantasy Football player predictions.

This module provides advanced feature engineering capabilities to better utilize
existing data sources and improve model performance through sophisticated
statistical features, position-specific metrics, and interaction terms.
"""

from typing import List, Dict

import numpy as np
import pandas as pd


def add_advanced_rolling_stats(df: pd.DataFrame, stat_cols: List[str], 
                             windows: List[int] = [3, 6, 10]) -> pd.DataFrame:
    """
    Add advanced rolling statistics including momentum, consistency, and volatility metrics.
    
    Args:
        df: DataFrame with player game data
        stat_cols: List of statistical columns to calculate rolling stats for
        windows: List of rolling window sizes
    
    Returns:
        DataFrame with additional rolling statistics columns
    """
    df_enhanced = df.copy()
    
    for window in windows:
        for col in stat_cols:
            if col not in df.columns:
                continue
                
            prefix = f"{col}_{window}game"
            
            # Standard rolling mean (already exists in current implementation)
            df_enhanced[f"{prefix}_mean"] = df[col].rolling(window, min_periods=1).mean()
            
            # Rolling standard deviation (volatility)
            df_enhanced[f"{prefix}_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)
            
            # Rolling coefficient of variation (relative volatility)
            rolling_mean = df_enhanced[f"{prefix}_mean"]
            df_enhanced[f"{prefix}_cv"] = (df_enhanced[f"{prefix}_std"] / 
                                         (rolling_mean + 1e-6)).fillna(0)
            
            # Momentum (trend): recent vs older performance
            if window >= 6:
                recent_window = max(2, window // 3)
                recent_mean = df[col].rolling(recent_window, min_periods=1).mean()
                older_mean = df[col].shift(recent_window).rolling(
                    window - recent_window, min_periods=1).mean()
                df_enhanced[f"{prefix}_momentum"] = (recent_mean - older_mean).fillna(0)
            
            # Rolling max and min for ceiling/floor metrics
            df_enhanced[f"{prefix}_max"] = df[col].rolling(window, min_periods=1).max()
            df_enhanced[f"{prefix}_min"] = df[col].rolling(window, min_periods=1).min()
            
            # Percentage above/below rolling average
            df_enhanced[f"{prefix}_pct_above_avg"] = (
                (df[col] > rolling_mean).rolling(window, min_periods=1).mean()
            ).fillna(0)
    
    return df_enhanced


def add_position_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position-specific features that are most relevant for each position.
    
    Args:
        df: DataFrame with player game data
    
    Returns:
        DataFrame with position-specific features
    """
    df_enhanced = df.copy()
    
    # QB-specific features
    qb_mask = df['position'] == 'QB'
    if qb_mask.any():
        # Passing efficiency metrics
        df_enhanced.loc[qb_mask, 'passing_yards_per_attempt'] = (
            df.loc[qb_mask, 'passingYards'] / (df.loc[qb_mask, 'passingAttempts'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[qb_mask, 'passing_td_rate'] = (
            df.loc[qb_mask, 'passingTouchdowns'] / (df.loc[qb_mask, 'passingAttempts'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[qb_mask, 'interception_rate'] = (
            df.loc[qb_mask, 'passingInterceptions'] / (df.loc[qb_mask, 'passingAttempts'] + 1e-6)
        ).fillna(0)
    
    # RB-specific features
    rb_mask = df['position'] == 'RB'
    if rb_mask.any():
        # Rushing efficiency
        df_enhanced.loc[rb_mask, 'rushing_yards_per_attempt'] = (
            df.loc[rb_mask, 'rushingYards'] / (df.loc[rb_mask, 'rushingAttempts'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[rb_mask, 'rushing_td_rate'] = (
            df.loc[rb_mask, 'rushingTouchdowns'] / (df.loc[rb_mask, 'rushingAttempts'] + 1e-6)
        ).fillna(0)
        
        # Receiving involvement for pass-catching backs
        df_enhanced.loc[rb_mask, 'target_share'] = (
            df.loc[rb_mask, 'receivingTargets'] / (df.loc[rb_mask, 'rushingAttempts'] + 
                                                   df.loc[rb_mask, 'receivingTargets'] + 1e-6)
        ).fillna(0)
    
    # WR/TE-specific features
    pass_catcher_mask = df['position'].isin(['WR', 'TE'])
    if pass_catcher_mask.any():
        # Reception efficiency
        df_enhanced.loc[pass_catcher_mask, 'catch_rate'] = (
            df.loc[pass_catcher_mask, 'receivingReceptions'] / 
            (df.loc[pass_catcher_mask, 'receivingTargets'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[pass_catcher_mask, 'yards_per_reception'] = (
            df.loc[pass_catcher_mask, 'receivingYards'] / 
            (df.loc[pass_catcher_mask, 'receivingReceptions'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[pass_catcher_mask, 'yards_per_target'] = (
            df.loc[pass_catcher_mask, 'receivingYards'] / 
            (df.loc[pass_catcher_mask, 'receivingTargets'] + 1e-6)
        ).fillna(0)
        
        df_enhanced.loc[pass_catcher_mask, 'td_per_target'] = (
            df.loc[pass_catcher_mask, 'receivingTouchdowns'] / 
            (df.loc[pass_catcher_mask, 'receivingTargets'] + 1e-6)
        ).fillna(0)
    
    # Defense-specific features
    def_mask = df['position'] == 'D/ST'
    if def_mask.any():
        # Defensive scoring efficiency
        df_enhanced.loc[def_mask, 'points_allowed_avg'] = (
            df.loc[def_mask, 'defensive_points_allowed_0-6'] + 
            df.loc[def_mask, 'defensive_points_allowed_7-13'] + 
            df.loc[def_mask, 'defensive_points_allowed_14-20']
        ).fillna(0) if all(col in df.columns for col in [
            'defensive_points_allowed_0-6', 'defensive_points_allowed_7-13', 
            'defensive_points_allowed_14-20'
        ]) else 0
    
    return df_enhanced


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game context features like rest days, divisional games, etc.
    
    Args:
        df: DataFrame with player game data
    
    Returns:
        DataFrame with game context features
    """
    df_enhanced = df.copy()
    
    # Sort by player and date for proper lag calculations
    df_enhanced = df_enhanced.sort_values(['player_id', 'date'])
    
    # Days of rest since last game
    df_enhanced['days_rest'] = df_enhanced.groupby('player_id')['date'].diff().dt.days
    df_enhanced['days_rest'] = df_enhanced['days_rest'].fillna(7)  # Default to 7 for first game
    
    # Short rest indicator (less than 6 days)
    df_enhanced['short_rest'] = (df_enhanced['days_rest'] < 6).astype(int)
    
    # Long rest indicator (more than 9 days, like bye weeks)
    df_enhanced['long_rest'] = (df_enhanced['days_rest'] > 9).astype(int)
    
    # Divisional opponent indicator (if we can determine divisions)
    # This would require additional team division mapping
    df_enhanced['divisional_game'] = 0  # Placeholder for now
    
    # Home/away game indicator (if available in data)
    # This would need to be derived from schedule data
    df_enhanced['is_home_game'] = 0  # Placeholder for now
    
    # Time of season (early, mid, late season effects)
    df_enhanced['season_portion'] = np.select([
        df_enhanced['week_num'] <= 6,
        df_enhanced['week_num'] <= 12,
        df_enhanced['week_num'] > 12
    ], [0, 1, 2], default=1)  # 0=early, 1=mid, 2=late
    
    return df_enhanced


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms between key variables that might have multiplicative effects.
    
    Args:
        df: DataFrame with player game data
    
    Returns:
        DataFrame with interaction features
    """
    df_enhanced = df.copy()
    
    # Age and experience interactions
    if 'age' in df.columns and 'experience' in df.columns:
        df_enhanced['age_experience_ratio'] = (
            df['age'] / (df['experience'] + 1)
        ).fillna(0)
        
        # Prime age indicators (peak performance windows by position)
        qb_prime = ((df['age'] >= 26) & (df['age'] <= 35) & (df['position'] == 'QB')).astype(int)
        rb_prime = ((df['age'] >= 22) & (df['age'] <= 28) & (df['position'] == 'RB')).astype(int)
        wr_prime = ((df['age'] >= 24) & (df['age'] <= 30) & (df['position'] == 'WR')).astype(int)
        te_prime = ((df['age'] >= 25) & (df['age'] <= 32) & (df['position'] == 'TE')).astype(int)
        
        df_enhanced['in_prime_years'] = qb_prime + rb_prime + wr_prime + te_prime
    
    # Draft capital and performance interactions
    if 'draft_pick' in df.columns:
        df_enhanced['early_round_pick'] = (df['draft_pick'] <= 64).astype(int)
        df_enhanced['high_draft_capital'] = (df['draft_pick'] <= 32).astype(int)
        
        # Draft pedigree combined with experience
        if 'experience' in df.columns:
            df_enhanced['draft_experience_interaction'] = (
                (193 - df['draft_pick']) * df['experience']  # Higher pick = higher value
            ).fillna(0)
    
    # Team strength interactions
    if 'team' in df.columns:
        # This would benefit from external team strength data
        # For now, create placeholder for team quality interactions
        df_enhanced['strong_team_indicator'] = 0  # Placeholder
    
    # Usage and efficiency interactions
    usage_cols = ['rushingAttempts', 'receivingTargets', 'passingAttempts']
    efficiency_cols = ['rushing_yards_per_attempt', 'yards_per_target', 'passing_yards_per_attempt']
    
    for usage_col, eff_col in zip(usage_cols, efficiency_cols):
        if usage_col in df.columns and eff_col in df.columns:
            df_enhanced[f'{usage_col}_efficiency_product'] = (
                df[usage_col] * df[eff_col]
            ).fillna(0)
    
    return df_enhanced


def add_consistency_metrics(df: pd.DataFrame, stat_cols: List[str], 
                          windows: List[int] = [6, 10]) -> pd.DataFrame:
    """
    Add consistency and reliability metrics for players.
    
    Args:
        df: DataFrame with player game data
        stat_cols: Statistical columns to analyze for consistency
        windows: Rolling window sizes for consistency analysis
    
    Returns:
        DataFrame with consistency metrics
    """
    df_enhanced = df.copy()
    df_enhanced = df_enhanced.sort_values(['player_id', 'date'])
    
    for window in windows:
        for col in stat_cols:
            if col not in df.columns:
                continue
                
            # Rolling coefficient of variation (consistency measure)
            rolling_mean = df_enhanced[col].rolling(window, min_periods=1).mean()
            rolling_std = df_enhanced[col].rolling(window, min_periods=1).std()
            df_enhanced[f'{col}_{window}game_consistency'] = (
                1 / (1 + rolling_std / (rolling_mean + 1e-6))
            ).fillna(0)
            
            # Floor percentage (games above a threshold)
            threshold = rolling_mean * 0.7  # 70% of average as floor
            df_enhanced[f'{col}_{window}game_floor_pct'] = (
                (df[col] >= threshold).rolling(window, min_periods=1).mean()
            ).fillna(0)
            
            # Ceiling games (games significantly above average)
            ceiling_threshold = rolling_mean * 1.5  # 150% of average as ceiling
            df_enhanced[f'{col}_{window}game_ceiling_pct'] = (
                (df[col] >= ceiling_threshold).rolling(window, min_periods=1).mean()
            ).fillna(0)
    
    return df_enhanced


def calculate_target_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate target share and touch share within teams for relevant positions.
    
    Args:
        df: DataFrame with player game data
    
    Returns:
        DataFrame with target/touch share metrics
    """
    df_enhanced = df.copy()
    
    # Calculate team totals by game
    team_totals = df.groupby(['season', 'team', 'week_num']).agg({
        'receivingTargets': 'sum',
        'rushingAttempts': 'sum',
        'receivingReceptions': 'sum'
    }).reset_index()
    
    team_totals = team_totals.rename(columns={
        'receivingTargets': 'team_targets',
        'rushingAttempts': 'team_carries',
        'receivingReceptions': 'team_receptions'
    })
    
    # Merge back with player data
    df_enhanced = df_enhanced.merge(
        team_totals, 
        on=['season', 'team', 'week_num'], 
        how='left'
    )
    
    # Calculate shares
    df_enhanced['target_share'] = (
        df_enhanced['receivingTargets'] / (df_enhanced['team_targets'] + 1e-6)
    ).fillna(0)
    
    df_enhanced['carry_share'] = (
        df_enhanced['rushingAttempts'] / (df_enhanced['team_carries'] + 1e-6)
    ).fillna(0)
    
    df_enhanced['reception_share'] = (
        df_enhanced['receivingReceptions'] / (df_enhanced['team_receptions'] + 1e-6)
    ).fillna(0)
    
    # Combined touch share for skill position players
    df_enhanced['total_touches'] = (
        df_enhanced['rushingAttempts'].fillna(0) + 
        df_enhanced['receivingTargets'].fillna(0)
    )
    
    df_enhanced['team_total_touches'] = (
        df_enhanced['team_carries'].fillna(0) + 
        df_enhanced['team_targets'].fillna(0)
    )
    
    df_enhanced['touch_share'] = (
        df_enhanced['total_touches'] / (df_enhanced['team_total_touches'] + 1e-6)
    ).fillna(0)
    
    return df_enhanced


def add_trend_features(df: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
    """
    Add trend analysis features to identify improving or declining players.
    
    Args:
        df: DataFrame with player game data
        stat_cols: Statistical columns to analyze for trends
    
    Returns:
        DataFrame with trend features
    """
    df_enhanced = df.copy()
    df_enhanced = df_enhanced.sort_values(['player_id', 'date'])
    
    for col in stat_cols:
        if col not in df.columns:
            continue
            
        # Linear trend over last 6 and 10 games
        for window in [6, 10]:
            # Calculate slope of trend line
            def calculate_trend_slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                try:
                    slope = np.polyfit(x, series, 1)[0]
                    return slope
                except:
                    return 0
            
            df_enhanced[f'{col}_{window}game_trend'] = (
                df_enhanced.groupby('player_id')[col]
                .rolling(window, min_periods=2)
                .apply(calculate_trend_slope, raw=False)
                .fillna(0)
            )
    
    return df_enhanced


def enhance_features_comprehensive(df: pd.DataFrame, 
                                 point_categories: Dict[str, float]) -> pd.DataFrame:
    """
    Apply all feature engineering enhancements in the correct order.
    
    Args:
        df: DataFrame with player game data
        point_categories: Dictionary mapping stat categories to point values
    
    Returns:
        DataFrame with all enhanced features
    """
    # Get relevant statistical columns
    stat_cols = [col for col in point_categories.keys() if col in df.columns]
    usage_cols = ['receivingTargets', 'rushingAttempts', 'passingAttempts']
    stat_cols.extend([col for col in usage_cols if col in df.columns])
    
    # Apply enhancements in logical order
    df_enhanced = df.copy()
    
    # 1. Position-specific features (creates efficiency metrics)
    df_enhanced = add_position_specific_features(df_enhanced)
    
    # 2. Game context features
    df_enhanced = add_game_context_features(df_enhanced)
    
    # 3. Target/touch share calculations
    df_enhanced = calculate_target_share(df_enhanced)
    
    # 4. Advanced rolling statistics
    df_enhanced = add_advanced_rolling_stats(df_enhanced, stat_cols)
    
    # 5. Consistency metrics
    df_enhanced = add_consistency_metrics(df_enhanced, stat_cols)
    
    # 6. Trend analysis
    df_enhanced = add_trend_features(df_enhanced, stat_cols)
    
    # 7. Interaction features (after all base features are created)
    df_enhanced = add_interaction_features(df_enhanced)
    
    return df_enhanced
"""
External data source integration for Fantasy Football predictions.

This module provides functionality to fetch and integrate external data sources
that can significantly improve prediction accuracy, including weather data,
betting lines, and advanced statistics.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from src.paths import root_dir
from src.tools import cache_wrapper
from src.constants import PRO_TEAM_MAP


# Stadium information for weather data
STADIUM_INFO = {
    'ARI': {'name': 'State Farm Stadium', 'city': 'Glendale', 'state': 'AZ', 'dome': True},
    'ATL': {'name': 'Mercedes-Benz Stadium', 'city': 'Atlanta', 'state': 'GA', 'dome': True},
    'BAL': {'name': 'M&T Bank Stadium', 'city': 'Baltimore', 'state': 'MD', 'dome': False},
    'BUF': {'name': 'Highmark Stadium', 'city': 'Orchard Park', 'state': 'NY', 'dome': False},
    'CAR': {'name': 'Bank of America Stadium', 'city': 'Charlotte', 'state': 'NC', 'dome': False},
    'CHI': {'name': 'Soldier Field', 'city': 'Chicago', 'state': 'IL', 'dome': False},
    'CIN': {'name': 'Paycor Stadium', 'city': 'Cincinnati', 'state': 'OH', 'dome': False},
    'CLE': {'name': 'Cleveland Browns Stadium', 'city': 'Cleveland', 'state': 'OH', 'dome': False},
    'DAL': {'name': 'AT&T Stadium', 'city': 'Arlington', 'state': 'TX', 'dome': True},
    'DEN': {'name': 'Empower Field at Mile High', 'city': 'Denver', 'state': 'CO', 'dome': False, 'altitude': 5280},
    'DET': {'name': 'Ford Field', 'city': 'Detroit', 'state': 'MI', 'dome': True},
    'GB': {'name': 'Lambeau Field', 'city': 'Green Bay', 'state': 'WI', 'dome': False},
    'HOU': {'name': 'NRG Stadium', 'city': 'Houston', 'state': 'TX', 'dome': True},
    'IND': {'name': 'Lucas Oil Stadium', 'city': 'Indianapolis', 'state': 'IN', 'dome': True},
    'JAX': {'name': 'TIAA Bank Field', 'city': 'Jacksonville', 'state': 'FL', 'dome': False},
    'KC': {'name': 'Arrowhead Stadium', 'city': 'Kansas City', 'state': 'MO', 'dome': False},
    'LV': {'name': 'Allegiant Stadium', 'city': 'Las Vegas', 'state': 'NV', 'dome': True},
    'LAC': {'name': 'SoFi Stadium', 'city': 'Los Angeles', 'state': 'CA', 'dome': True},
    'LAR': {'name': 'SoFi Stadium', 'city': 'Los Angeles', 'state': 'CA', 'dome': True},
    'MIA': {'name': 'Hard Rock Stadium', 'city': 'Miami Gardens', 'state': 'FL', 'dome': False},
    'MIN': {'name': 'U.S. Bank Stadium', 'city': 'Minneapolis', 'state': 'MN', 'dome': True},
    'NE': {'name': 'Gillette Stadium', 'city': 'Foxborough', 'state': 'MA', 'dome': False},
    'NO': {'name': 'Caesars Superdome', 'city': 'New Orleans', 'state': 'LA', 'dome': True},
    'NYG': {'name': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'dome': False},
    'NYJ': {'name': 'MetLife Stadium', 'city': 'East Rutherford', 'state': 'NJ', 'dome': False},
    'PHI': {'name': 'Lincoln Financial Field', 'city': 'Philadelphia', 'state': 'PA', 'dome': False},
    'PIT': {'name': 'Heinz Field', 'city': 'Pittsburgh', 'state': 'PA', 'dome': False},
    'SF': {'name': "Levi's Stadium", 'city': 'Santa Clara', 'state': 'CA', 'dome': False},
    'SEA': {'name': 'Lumen Field', 'city': 'Seattle', 'state': 'WA', 'dome': False},
    'TB': {'name': 'Raymond James Stadium', 'city': 'Tampa', 'state': 'FL', 'dome': False},
    'TEN': {'name': 'Nissan Stadium', 'city': 'Nashville', 'state': 'TN', 'dome': False},
    'WAS': {'name': 'FedExField', 'city': 'Landover', 'state': 'MD', 'dome': False},
}


def get_stadium_info(team: str) -> Dict:
    """Get stadium information for a team."""
    return STADIUM_INFO.get(team, {})


@cache_wrapper(60 * 60 * 12)  # Cache for 12 hours
def get_weather_data(city: str, state: str, date: datetime, 
                    api_key: Optional[str] = None) -> Dict:
    """
    Fetch weather data for a specific location and date.
    
    Uses OpenWeatherMap API (free tier allows 1000 calls/day).
    For historical data, uses a free weather API or cached data.
    
    Args:
        city: City name
        state: State abbreviation
        date: Game date
        api_key: OpenWeatherMap API key (optional)
    
    Returns:
        Dictionary with weather information
    """
    # For free implementation, we'll use a mock weather service
    # In production, you would integrate with OpenWeatherMap or similar
    
    # Check if it's a historical date (more than 5 days ago)
    days_ago = (datetime.now() - date).days
    
    if days_ago > 5:
        # Use historical weather data or mock data for past games
        return get_historical_weather_mock(city, state, date)
    else:
        # Use forecast API for future games
        return get_forecast_weather_mock(city, state, date)


def get_historical_weather_mock(city: str, state: str, date: datetime) -> Dict:
    """
    Mock historical weather data based on location and seasonal patterns.
    In production, this would call a historical weather API.
    """
    # Mock data based on geographical and seasonal patterns
    weather_patterns = {
        'winter_cold': ['GB', 'BUF', 'CLE', 'CHI', 'DEN', 'NE', 'PIT'],
        'winter_mild': ['BAL', 'CIN', 'PHI', 'WAS', 'TEN', 'NYG', 'NYJ'],
        'warm': ['MIA', 'TB', 'NO', 'HOU', 'JAX', 'ARI', 'LV', 'LAC', 'LAR'],
        'variable': ['ATL', 'CAR', 'SEA', 'SF', 'KC']
    }
    
    # Determine climate zone
    team = None
    for climate, teams in weather_patterns.items():
        if any(team_city.lower() in city.lower() for team_city in teams):
            climate_zone = climate
            break
    else:
        climate_zone = 'variable'
    
    # Season-based weather patterns
    month = date.month
    is_winter = month in [12, 1, 2]
    is_fall = month in [9, 10, 11]
    
    # Mock weather based on patterns
    if climate_zone == 'winter_cold' and is_winter:
        temp = np.random.normal(25, 15)  # Cold with variation
        wind = np.random.normal(12, 8)   # Higher wind
        precip = np.random.choice([0, 0, 0, 1, 2])  # Chance of snow/rain
    elif climate_zone == 'warm':
        temp = np.random.normal(75, 10)  # Warm
        wind = np.random.normal(8, 5)    # Moderate wind
        precip = np.random.choice([0, 0, 0, 0, 1])  # Low precipitation
    else:
        temp = np.random.normal(50, 20)  # Variable
        wind = np.random.normal(10, 7)   # Moderate wind
        precip = np.random.choice([0, 0, 0, 1])  # Some precipitation
    
    return {
        'temperature': max(temp, -10),  # Floor at -10°F
        'wind_speed': max(wind, 0),     # Non-negative wind
        'precipitation': precip,        # 0=none, 1=rain, 2=snow
        'humidity': np.random.normal(60, 20),
        'pressure': np.random.normal(30, 2),
        'visibility': 10 if precip == 0 else np.random.normal(5, 3)
    }


def get_forecast_weather_mock(city: str, state: str, date: datetime) -> Dict:
    """Mock forecast weather for future games."""
    # Similar logic to historical but potentially different patterns
    return get_historical_weather_mock(city, state, date)


def add_weather_features(df: pd.DataFrame, schedules: Dict) -> pd.DataFrame:
    """
    Add weather-based features to the game data.
    
    Args:
        df: DataFrame with game data
        schedules: Game schedule information
    
    Returns:
        DataFrame with weather features added
    """
    df_enhanced = df.copy()
    
    # Initialize weather columns
    weather_cols = [
        'temperature', 'wind_speed', 'precipitation', 'humidity', 
        'is_dome', 'is_cold_weather', 'is_windy', 'is_precipitation',
        'altitude_factor'
    ]
    
    for col in weather_cols:
        df_enhanced[col] = 0
    
    for idx, row in df_enhanced.iterrows():
        team = row.get('team')
        date = row.get('date')
        
        if not team or not date:
            continue
            
        stadium_info = get_stadium_info(team)
        
        # Set dome indicator
        df_enhanced.loc[idx, 'is_dome'] = 1 if stadium_info.get('dome', False) else 0
        
        # Set altitude factor (affects kicking and potentially passing)
        altitude = stadium_info.get('altitude', 0)
        df_enhanced.loc[idx, 'altitude_factor'] = altitude / 1000  # Scaled factor
        
        # Only get weather for outdoor stadiums
        if not stadium_info.get('dome', False):
            city = stadium_info.get('city', '')
            state = stadium_info.get('state', '')
            
            if city and state:
                weather = get_weather_data(city, state, date)
                
                df_enhanced.loc[idx, 'temperature'] = weather.get('temperature', 70)
                df_enhanced.loc[idx, 'wind_speed'] = weather.get('wind_speed', 8)
                df_enhanced.loc[idx, 'precipitation'] = weather.get('precipitation', 0)
                df_enhanced.loc[idx, 'humidity'] = weather.get('humidity', 60)
                
                # Derived weather indicators
                df_enhanced.loc[idx, 'is_cold_weather'] = 1 if weather.get('temperature', 70) < 40 else 0
                df_enhanced.loc[idx, 'is_windy'] = 1 if weather.get('wind_speed', 8) > 15 else 0
                df_enhanced.loc[idx, 'is_precipitation'] = 1 if weather.get('precipitation', 0) > 0 else 0
    
    return df_enhanced


def get_betting_lines_mock(games: List[Dict]) -> Dict:
    """
    Mock betting lines data. In production, this would integrate with
    a sports betting API like The Odds API (free tier available).
    
    Args:
        games: List of game information
    
    Returns:
        Dictionary with betting lines for each game
    """
    betting_data = {}
    
    for game in games:
        game_id = game.get('id', '')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        if not game_id or not home_team or not away_team:
            continue
        
        # Mock betting lines based on team strength patterns
        # In production, this would be actual betting data
        mock_total = np.random.normal(47, 6)  # Average NFL total
        mock_spread = np.random.normal(0, 4)  # Random spread
        
        betting_data[game_id] = {
            'total': max(mock_total, 30),  # Floor at 30
            'spread': mock_spread,         # Positive favors away team
            'home_implied_total': (mock_total - mock_spread) / 2,
            'away_implied_total': (mock_total + mock_spread) / 2,
        }
    
    return betting_data


def add_betting_features(df: pd.DataFrame, schedules: Dict) -> pd.DataFrame:
    """
    Add betting line features that predict game script and scoring environment.
    
    Args:
        df: DataFrame with game data
        schedules: Game schedule information
    
    Returns:
        DataFrame with betting features
    """
    df_enhanced = df.copy()
    
    # Initialize betting columns
    betting_cols = [
        'game_total', 'team_implied_total', 'opponent_implied_total',
        'spread_favored', 'is_high_total', 'is_low_total', 'is_big_favorite',
        'is_big_underdog', 'pace_environment'
    ]
    
    for col in betting_cols:
        df_enhanced[col] = 0
    
    # Mock implementation - in production, integrate with betting API
    for idx, row in df_enhanced.iterrows():
        team = row.get('team')
        opponent = row.get('opponent')
        
        if not team or not opponent:
            continue
        
        # Mock betting data
        game_total = np.random.normal(47, 6)
        spread = np.random.normal(0, 4)
        
        df_enhanced.loc[idx, 'game_total'] = max(game_total, 30)
        df_enhanced.loc[idx, 'team_implied_total'] = (game_total - spread) / 2
        df_enhanced.loc[idx, 'opponent_implied_total'] = (game_total + spread) / 2
        df_enhanced.loc[idx, 'spread_favored'] = 1 if spread < -3 else 0
        df_enhanced.loc[idx, 'is_high_total'] = 1 if game_total > 50 else 0
        df_enhanced.loc[idx, 'is_low_total'] = 1 if game_total < 42 else 0
        df_enhanced.loc[idx, 'is_big_favorite'] = 1 if spread < -7 else 0
        df_enhanced.loc[idx, 'is_big_underdog'] = 1 if spread > 7 else 0
        df_enhanced.loc[idx, 'pace_environment'] = 1 if game_total > 48 else 0
    
    return df_enhanced


@cache_wrapper(60 * 60 * 24)  # Cache for 24 hours
def scrape_pro_football_reference_stats(year: int, stat_type: str = 'passing') -> pd.DataFrame:
    """
    Scrape advanced statistics from Pro Football Reference (free).
    
    Args:
        year: Season year
        stat_type: Type of stats ('passing', 'rushing', 'receiving', 'defense')
    
    Returns:
        DataFrame with advanced statistics
    """
    # Mock implementation - in production, would scrape PFR
    # Note: Be respectful of rate limits and terms of service
    
    mock_data = {
        'player_name': ['Player A', 'Player B', 'Player C'],
        'team': ['KC', 'BUF', 'LAR'],
        'position': ['QB', 'RB', 'WR'],
        'advanced_stat_1': [85.2, 4.8, 12.3],
        'advanced_stat_2': [12, 125, 8],
        'efficiency_rating': [0.85, 0.75, 0.82]
    }
    
    return pd.DataFrame(mock_data)


def add_advanced_stats_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Add advanced statistics from external sources.
    
    Args:
        df: DataFrame with game data
        year: Season year
    
    Returns:
        DataFrame with advanced stats features
    """
    # This would integrate advanced stats from sources like PFF, PFR, etc.
    # For now, return the original DataFrame with placeholder features
    
    df_enhanced = df.copy()
    df_enhanced['pff_grade'] = 0  # Placeholder for PFF grades
    df_enhanced['air_yards'] = 0  # Placeholder for air yards
    df_enhanced['yac'] = 0        # Placeholder for yards after catch
    df_enhanced['pressure_rate'] = 0  # Placeholder for pressure metrics
    
    return df_enhanced


def get_rest_and_travel_data(df: pd.DataFrame, schedules: Dict) -> pd.DataFrame:
    """
    Add rest days and travel distance features.
    
    Args:
        df: DataFrame with game data
        schedules: Game schedule information
    
    Returns:
        DataFrame with rest and travel features
    """
    df_enhanced = df.copy()
    
    # Sort by player and date for proper calculations
    df_enhanced = df_enhanced.sort_values(['player_id', 'date'])
    
    # Calculate days of rest (already done in feature_engineering.py)
    # Add travel distance calculations
    
    # Mock travel distances between cities (in miles)
    # In production, this would use actual distance calculations
    CITY_DISTANCES = {
        ('KC', 'LV'): 550,
        ('SEA', 'MIA'): 2750,
        ('NE', 'LAR'): 2600,
        # ... would need full distance matrix
    }
    
    df_enhanced['travel_distance'] = 0
    
    for idx, row in df_enhanced.iterrows():
        team = row.get('team')
        opponent = row.get('opponent')
        
        if team and opponent:
            # Mock distance calculation
            distance = CITY_DISTANCES.get((team, opponent), 
                      CITY_DISTANCES.get((opponent, team), 500))  # Default 500 miles
            df_enhanced.loc[idx, 'travel_distance'] = distance
    
    # Travel burden indicators
    df_enhanced['long_travel'] = (df_enhanced['travel_distance'] > 1500).astype(int)
    df_enhanced['cross_country'] = (df_enhanced['travel_distance'] > 2000).astype(int)
    
    return df_enhanced


def integrate_all_external_sources(df: pd.DataFrame, schedules: Dict, 
                                 year: int) -> pd.DataFrame:
    """
    Integrate all external data sources into the DataFrame.
    
    Args:
        df: DataFrame with game data
        schedules: Game schedule information
        year: Season year
    
    Returns:
        DataFrame with all external features integrated
    """
    df_enhanced = df.copy()
    
    print("Adding weather features...")
    df_enhanced = add_weather_features(df_enhanced, schedules)
    
    print("Adding betting line features...")
    df_enhanced = add_betting_features(df_enhanced, schedules)
    
    print("Adding advanced stats features...")
    df_enhanced = add_advanced_stats_features(df_enhanced, year)
    
    print("Adding rest and travel features...")
    df_enhanced = get_rest_and_travel_data(df_enhanced, schedules)
    
    return df_enhanced


# Free API endpoints and data sources documentation
FREE_DATA_SOURCES = {
    'weather': {
        'openweathermap': 'https://openweathermap.org/api (free tier: 1000 calls/day)',
        'weatherapi': 'https://www.weatherapi.com/ (free tier: 1M calls/month)',
    },
    'betting': {
        'the_odds_api': 'https://the-odds-api.com/ (free tier: 500 requests/month)',
        'rapidapi_odds': 'https://rapidapi.com/theoddsapi/api/live-sports-odds/',
    },
    'advanced_stats': {
        'pro_football_reference': 'https://www.pro-football-reference.com/ (scraping)',
        'espn_stats': 'https://www.espn.com/nfl/stats (scraping)',
        'nfl_com': 'https://www.nfl.com/stats/ (scraping)',
    },
    'team_data': {
        'espn_api': 'ESPN API (team stats, schedules)',
        'sleeper_api': 'Sleeper API (player info, team data)',
    }
}


def print_data_source_info():
    """Print information about available free data sources."""
    print("\n=== FREE EXTERNAL DATA SOURCES ===")
    for category, sources in FREE_DATA_SOURCES.items():
        print(f"\n{category.upper()}:")
        for source, info in sources.items():
            print(f"  - {source}: {info}")
    
    print("\nNOTE: Always respect rate limits and terms of service when using external APIs.")
    print("Consider implementing caching and error handling for production use.")


if __name__ == '__main__':
    print_data_source_info()
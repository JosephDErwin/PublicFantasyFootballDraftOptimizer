"""
Synthetic Data Generation for Portfolio Demo

This module generates realistic fantasy football data for demonstration purposes,
replacing the need for external API calls or real league data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class SyntheticDataGenerator:
    """Generates realistic fantasy football data for demo purposes."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.positions = ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']
        self.nfl_teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
            'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB',
            'TEN', 'WAS'
        ]
        
        # Realistic distribution of positions in fantasy
        self.position_weights = {
            'QB': 0.08, 'RB': 0.25, 'WR': 0.35, 'TE': 0.12, 'K': 0.08, 'D/ST': 0.12
        }
        
        # Fantasy scoring settings (PPR)
        self.scoring = {
            'passing_yards': 0.04,
            'passing_tds': 4,
            'rushing_yards': 0.1,
            'rushing_tds': 6,
            'receiving_yards': 0.1,
            'receiving_tds': 6,
            'receptions': 1,  # PPR
            'fumbles': -2,
            'interceptions': -2
        }
    
    def generate_players(self, n_players: int = 400) -> List[Dict]:
        """Generate a realistic pool of fantasy football players."""
        players = []
        
        for i in range(n_players):
            position = np.random.choice(
                list(self.position_weights.keys()),
                p=list(self.position_weights.values())
            )
            
            player = self._generate_single_player(i, position)
            players.append(player)
        
        return players
    
    def _generate_single_player(self, player_id: int, position: str) -> Dict:
        """Generate a single realistic player."""
        first_names = ['Josh', 'Justin', 'Lamar', 'Aaron', 'Patrick', 'Derrick', 'Alvin', 'Christian',
                      'Davante', 'Tyreek', 'Cooper', 'Travis', 'George', 'Mark', 'Daniel']
        last_names = ['Allen', 'Herbert', 'Jackson', 'Rodgers', 'Mahomes', 'Henry', 'Kamara', 'McCaffrey',
                     'Adams', 'Hill', 'Kupp', 'Kelce', 'Kittle', 'Andrews', 'Jones']
        
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        if player_id > 0:  # Add ID to avoid duplicates in demo
            name += f" {player_id}"
        
        # Generate realistic stats based on position
        base_stats = self._get_position_base_stats(position)
        
        # Add some randomness and correlation
        age = np.random.randint(22, 35)
        experience = min(age - 21, np.random.randint(0, 13))
        
        # Age curve effects (peak around 26-28)
        age_factor = 1.0 - abs(age - 27) * 0.02
        age_factor = max(0.7, age_factor)
        
        # Apply age factor to stats
        for stat in base_stats:
            if stat in ['projected_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
                base_stats[stat] *= age_factor
        
        player = {
            'id': f'player_{player_id:03d}',
            'name': name,
            'position': position,
            'pro_position': position,
            'pro_team': np.random.choice(self.nfl_teams),
            'age': age,
            'experience': experience,
            'height': np.random.randint(66, 80),  # inches
            'weight': np.random.randint(170, 280),  # lbs
            'bye_week': np.random.randint(4, 15),
            'adp': 0,  # Will be calculated later
            'injury_risk': np.random.beta(2, 8),  # Most players low risk
            'consistency': np.random.beta(5, 2),  # Most players fairly consistent
            **base_stats
        }
        
        return player
    
    def _get_position_base_stats(self, position: str) -> Dict:
        """Generate position-appropriate base statistics."""
        if position == 'QB':
            return {
                'passing_yards': np.random.normal(4200, 800),
                'passing_tds': np.random.normal(28, 8),
                'interceptions': np.random.normal(12, 4),
                'rushing_yards': np.random.normal(250, 200),
                'rushing_tds': np.random.normal(3, 2),
                'fumbles': np.random.normal(5, 2),
                'projected_points': np.random.normal(280, 50),
                'games_played': 16
            }
        elif position == 'RB':
            return {
                'rushing_yards': np.random.normal(1000, 400),
                'rushing_tds': np.random.normal(8, 4),
                'receptions': np.random.normal(40, 20),
                'receiving_yards': np.random.normal(350, 200),
                'receiving_tds': np.random.normal(2, 2),
                'fumbles': np.random.normal(2, 1),
                'projected_points': np.random.normal(200, 60),
                'games_played': np.random.randint(12, 17)
            }
        elif position == 'WR':
            return {
                'receptions': np.random.normal(70, 25),
                'receiving_yards': np.random.normal(1000, 350),
                'receiving_tds': np.random.normal(7, 4),
                'rushing_yards': np.random.normal(20, 30),
                'rushing_tds': np.random.normal(0.5, 1),
                'fumbles': np.random.normal(1, 1),
                'projected_points': np.random.normal(180, 50),
                'games_played': np.random.randint(14, 17)
            }
        elif position == 'TE':
            return {
                'receptions': np.random.normal(50, 20),
                'receiving_yards': np.random.normal(600, 250),
                'receiving_tds': np.random.normal(5, 3),
                'fumbles': np.random.normal(1, 1),
                'projected_points': np.random.normal(140, 40),
                'games_played': np.random.randint(13, 17)
            }
        elif position == 'K':
            return {
                'field_goals': np.random.normal(25, 5),
                'extra_points': np.random.normal(35, 8),
                'projected_points': np.random.normal(120, 20),
                'games_played': 16
            }
        else:  # D/ST
            return {
                'sacks': np.random.normal(35, 10),
                'interceptions': np.random.normal(12, 5),
                'fumble_recoveries': np.random.normal(8, 3),
                'defensive_tds': np.random.normal(2, 2),
                'points_allowed': np.random.normal(22, 6),
                'projected_points': np.random.normal(130, 25),
                'games_played': 16
            }
    
    def calculate_adp(self, players: List[Dict]) -> List[Dict]:
        """Calculate Average Draft Position based on projected points."""
        # Sort by projected points with some noise
        players_df = pd.DataFrame(players)
        players_df['draft_score'] = (
            players_df['projected_points'] + 
            np.random.normal(0, 20, len(players_df))  # Add noise
        )
        
        players_df = players_df.sort_values('draft_score', ascending=False)
        players_df['adp'] = range(1, len(players_df) + 1)
        
        # Add some position-based adjustments (QBs drafted later in real drafts)
        qb_mask = players_df['position'] == 'QB'
        players_df.loc[qb_mask, 'adp'] += 50
        
        return players_df.to_dict('records')
    
    def generate_historical_data(self, players: List[Dict], n_seasons: int = 3) -> pd.DataFrame:
        """Generate historical performance data for modeling."""
        historical_data = []
        
        for season in range(2021, 2021 + n_seasons):
            for player in players:
                # Skip some players for realism (injuries, etc.)
                if np.random.random() < 0.15:
                    continue
                
                season_data = self._generate_season_data(player, season)
                historical_data.append(season_data)
        
        return pd.DataFrame(historical_data)
    
    def _generate_season_data(self, player: Dict, season: int) -> Dict:
        """Generate a single season of data for a player."""
        base_points = player['projected_points']
        
        # Add season-to-season variation
        season_factor = np.random.normal(1.0, 0.2)
        season_factor = max(0.3, min(1.8, season_factor))
        
        # Games played variation
        games_played = np.random.randint(8, 17)
        
        return {
            'player_id': player['id'],
            'season': season,
            'position': player['position'],
            'age': player['age'] - (2024 - season),
            'games_played': games_played,
            'total_points': base_points * season_factor * (games_played / 16),
            'points_per_game': base_points * season_factor / 16,
            'consistency': np.random.beta(5, 2),
            'injury_games_missed': 16 - games_played
        }
    
    def generate_league_settings(self) -> Dict:
        """Generate realistic league settings."""
        return {
            'teams': 12,
            'roster_slots': {
                'QB': 1,
                'RB': 2,
                'WR': 2,
                'TE': 1,
                'FLEX': 1,  # RB/WR/TE
                'K': 1,
                'D/ST': 1,
                'BE': 6  # Bench
            },
            'scoring': self.scoring,
            'draft_type': 'snake',
            'seasons': [2021, 2022, 2023, 2024]
        }


def create_demo_dataset(output_dir: str = 'data/demo'):
    """Create a complete demo dataset for the portfolio."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDataGenerator()
    
    # Generate players
    print("Generating player data...")
    players = generator.generate_players(400)
    players = generator.calculate_adp(players)
    
    # Save players data
    with open(output_path / 'players.json', 'w') as f:
        json.dump(players, f, indent=2)
    
    # Generate historical data
    print("Generating historical data...")
    historical_df = generator.generate_historical_data(players)
    historical_df.to_csv(output_path / 'historical_data.csv', index=False)
    
    # Generate league settings
    league_settings = generator.generate_league_settings()
    with open(output_path / 'league_settings.json', 'w') as f:
        json.dump(league_settings, f, indent=2)
    
    print(f"Demo dataset created in {output_path}")
    print(f"Generated {len(players)} players with {len(historical_df)} historical records")
    
    return players, historical_df, league_settings


if __name__ == '__main__':
    create_demo_dataset()
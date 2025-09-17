"""
Demo Data Utilities for Portfolio

This module provides portfolio-friendly data access functions that work with
synthetic data instead of requiring external API connections.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import namedtuple
import sys

# Add src to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import DEMO_MODE
from data.synthetic_data import SyntheticDataGenerator, create_demo_dataset

# Lightweight data structures for demo
DemoLeague = namedtuple('DemoLeague', ['teams', 'roster_slots', 'scoring', 'season'])
DemoTeam = namedtuple('DemoTeam', ['team_id', 'team_name', 'owner'])
DemoPlayer = namedtuple('DemoPlayer', [
    'id', 'name', 'position', 'pro_position', 'pro_team', 'age', 'bye_week',
    'projected_points', 'adp', 'vorp', 'injury_risk'
])


class DemoDataManager:
    """Manages synthetic data for portfolio demonstration."""
    
    def __init__(self, data_dir: str = 'data/demo'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_demo_data()
    
    def _initialize_demo_data(self):
        """Initialize or load demo data."""
        if not (self.data_dir / 'players.json').exists():
            print("Creating demo dataset...")
            create_demo_dataset(str(self.data_dir))
        
        # Load data
        with open(self.data_dir / 'players.json', 'r') as f:
            self.players_data = json.load(f)
        
        with open(self.data_dir / 'league_settings.json', 'r') as f:
            self.league_settings = json.load(f)
        
        self.historical_data = pd.read_csv(self.data_dir / 'historical_data.csv')
    
    def get_demo_league(self, league_name: str = 'demo_league', season: int = 2024) -> DemoLeague:
        """Create a demo league object."""
        return DemoLeague(
            teams=self.league_settings['teams'],
            roster_slots=self.league_settings['roster_slots'],
            scoring=self.league_settings['scoring'],
            season=season
        )
    
    def get_demo_players(self) -> List[DemoPlayer]:
        """Get demo player objects with calculated VORP."""
        players = []
        players_df = pd.DataFrame(self.players_data)
        
        # Calculate VORP
        players_df = self._calculate_vorp(players_df)
        
        for _, row in players_df.iterrows():
            player = DemoPlayer(
                id=row['id'],
                name=row['name'],
                position=row['position'],
                pro_position=row['pro_position'],
                pro_team=row['pro_team'],
                age=row['age'],
                bye_week=row['bye_week'],
                projected_points=row['projected_points'],
                adp=row['adp'],
                vorp=row.get('vorp', 0.0),
                injury_risk=row['injury_risk']
            )
            players.append(player)
        
        return players
    
    def _calculate_vorp(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Value Over Replacement Player."""
        players_df = players_df.copy()
        
        # Replacement level thresholds by position
        replacement_levels = {
            'QB': 12, 'RB': 24, 'WR': 36, 'TE': 12, 'K': 12, 'D/ST': 12
        }
        
        vorp_values = []
        for _, player in players_df.iterrows():
            position = player['position']
            pos_players = players_df[players_df['position'] == position].sort_values(
                'projected_points', ascending=False
            )
            
            replacement_idx = min(
                replacement_levels.get(position, 12) - 1,
                len(pos_players) - 1
            )
            
            if replacement_idx >= 0:
                replacement_points = pos_players.iloc[replacement_idx]['projected_points']
                vorp = max(0, player['projected_points'] - replacement_points)
            else:
                vorp = 0
            
            vorp_values.append(vorp)
        
        players_df['vorp'] = vorp_values
        return players_df
    
    def get_historical_data(self, player_id: Optional[str] = None) -> pd.DataFrame:
        """Get historical performance data."""
        if player_id:
            return self.historical_data[self.historical_data['player_id'] == player_id]
        return self.historical_data
    
    def get_demo_teams(self, league: DemoLeague) -> List[DemoTeam]:
        """Generate demo team objects."""
        team_names = [
            'The Champions', 'Draft Kings', 'Fantasy Gurus', 'Playoff Bound',
            'Title Contenders', 'The Dominators', 'Victory Squad', 'Elite Players',
            'Championship Chase', 'Fantasy Masters', 'The Competitors', 'Season Leaders'
        ]
        
        teams = []
        for i in range(league.teams):
            team = DemoTeam(
                team_id=i + 1,
                team_name=team_names[i] if i < len(team_names) else f'Team {i+1}',
                owner=f'Owner_{i+1}'
            )
            teams.append(team)
        
        return teams


# Global demo data manager
_demo_manager = None

def get_demo_manager() -> DemoDataManager:
    """Get the global demo data manager instance."""
    global _demo_manager
    if _demo_manager is None:
        _demo_manager = DemoDataManager()
    return _demo_manager


# Portfolio-friendly API functions
def get_league(league_name: str = 'demo_league', season: int = 2024, **kwargs) -> DemoLeague:
    """Get a demo league object for portfolio demonstration."""
    if DEMO_MODE:
        return get_demo_manager().get_demo_league(league_name, season)
    else:
        # In real mode, would connect to actual APIs
        raise NotImplementedError("Real API connections removed for portfolio version")


def get_all_players(league: DemoLeague, league_name: str = 'demo_league') -> List[DemoPlayer]:
    """Get all players with projections and VORP calculations."""
    if DEMO_MODE:
        return get_demo_manager().get_demo_players()
    else:
        raise NotImplementedError("Real API connections removed for portfolio version")


def generate_correlated_samples(correlation_matrix: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    """Generate correlated random samples for Monte Carlo simulation."""
    n_players = correlation_matrix.shape[0]
    
    # Ensure positive semi-definite matrix
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate correlated samples
    try:
        samples = np.random.multivariate_normal(
            mean=np.zeros(n_players),
            cov=correlation_matrix,
            size=n_samples
        )
    except Exception:
        # Fallback to independent samples if correlation matrix is problematic
        samples = np.random.normal(0, 1, (n_samples, n_players))
    
    return samples


def get_vorp_curve(players: List[DemoPlayer], season: int = 2024) -> callable:
    """Generate a VORP curve function for draft pick values."""
    # Extract VORP values and sort by draft position
    vorp_data = [(p.adp, p.vorp) for p in players if p.adp > 0]
    vorp_data.sort()
    
    adps = [x[0] for x in vorp_data]
    vorps = [x[1] for x in vorp_data]
    
    # Create interpolation function
    def vorp_curve(pick_numbers):
        if isinstance(pick_numbers, (int, float)):
            pick_numbers = [pick_numbers]
        
        result = []
        for pick in pick_numbers:
            if pick <= len(vorps):
                result.append(vorps[int(pick) - 1])
            else:
                # Extrapolate with decay
                result.append(max(0, vorps[-1] * (len(vorps) / pick)))
        
        return np.array(result) if len(result) > 1 else result[0]
    
    return vorp_curve


def get_age_curves(players: List[DemoPlayer]) -> Dict[str, callable]:
    """Generate age curve functions by position."""
    age_curves = {}
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']:
        pos_players = [p for p in players if p.position == position]
        
        if not pos_players:
            continue
        
        # Simple age curve model (peak around 27, decline after 30)
        def make_age_curve(pos_players=pos_players):
            def age_curve(age):
                if age < 24:
                    return 0.85 + (age - 22) * 0.075  # Improvement to peak
                elif age <= 28:
                    return 1.0  # Peak performance
                else:
                    return max(0.6, 1.0 - (age - 28) * 0.04)  # Decline
            return age_curve
        
        age_curves[position] = make_age_curve()
    
    return age_curves


def get_correlation_matrix(players: List[DemoPlayer]) -> np.ndarray:
    """Generate a realistic correlation matrix for player performances."""
    n_players = len(players)
    
    # Start with identity matrix
    correlation_matrix = np.eye(n_players)
    
    # Add position-based correlations
    positions = [p.position for p in players]
    teams = [p.pro_team for p in players]
    
    for i in range(n_players):
        for j in range(i + 1, n_players):
            correlation = 0.0
            
            # Same team correlation
            if teams[i] == teams[j]:
                correlation += 0.15
            
            # Same position correlation (negative for competing players)
            if positions[i] == positions[j]:
                correlation -= 0.05
            
            # Add small random correlation
            correlation += np.random.normal(0, 0.02)
            
            # Clamp correlation
            correlation = max(-0.3, min(0.3, correlation))
            
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    return correlation_matrix


# Compatibility functions for existing codebase
def get_schedule(league: DemoLeague) -> Dict:
    """Generate demo schedule data."""
    return {team: {} for team in ['Week 1', 'Week 2']}  # Simplified for demo


def get_sleeper():
    """Demo replacement for Sleeper API."""
    return {}


def get_bye_weeks(season: int = 2024) -> Dict[str, int]:
    """Generate demo bye week data."""
    teams = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
        'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
        'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB',
        'TEN', 'WAS'
    ]
    
    return {team: (hash(team) % 10) + 5 for team in teams}  # Bye weeks 5-14


if __name__ == '__main__':
    # Demo the functionality
    league = get_league()
    players = get_all_players(league)
    
    print(f"Generated {len(players)} demo players")
    print(f"League has {league.teams} teams")
    
    # Show top players by VORP
    top_players = sorted(players, key=lambda p: p.vorp, reverse=True)[:10]
    print("\nTop 10 Players by VORP:")
    for i, player in enumerate(top_players, 1):
        print(f"{i:2d}. {player.name:<25} ({player.position}) - VORP: {player.vorp:.2f}")
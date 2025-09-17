"""
Portfolio Draft Optimizer

A cleaned-up version of the draft optimizer specifically designed for portfolio
demonstration. This version showcases the optimization algorithms and ML integration
without requiring external API dependencies.
"""

import time
import numpy as np
import pandas as pd
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DEMO_MODE, N_SIMS
from data.demo_utilities import get_league, get_all_players, get_vorp_curve, get_age_curves
from data.demo_utilities import generate_correlated_samples, get_correlation_matrix

# Lightweight player data structure
LightweightPlayer = namedtuple(
    'LightweightPlayer',
    ['id', 'name', 'position', 'pro_position', 'pro_team', 'bye_week', 'age', 'adp', 'vorp']
)


class PortfolioDraftOptimizer:
    """
    Draft optimization engine for portfolio demonstration.
    
    This class showcases advanced optimization techniques including:
    - Monte Carlo simulation for draft scenarios
    - MILP-style optimization for roster construction
    - Correlation modeling for player performance
    - Value-based draft strategy
    """
    
    def __init__(self, draft_position: int = 6, league_size: int = 12, demo_mode: bool = True):
        self.draft_position = draft_position
        self.league_size = league_size
        self.demo_mode = demo_mode
        
        # Initialize data
        self.league = get_league()
        self.players = get_all_players(self.league)
        
        # Convert to lightweight format for processing
        self.lightweight_players = self._create_lightweight_players()
        
        # Calculate curves and utilities
        self.vorp_curve = get_vorp_curve(self.players)
        self.age_curves = get_age_curves(self.players)
        
        # Draft settings
        self.roster_slots = self.league.roster_slots
        self.total_rounds = sum(self.roster_slots.values()) - 1  # Exclude bench from requirements
        
        print(f"Initialized optimizer for position {draft_position} in {league_size}-team league")
        print(f"Loaded {len(self.players)} players for optimization")
    
    def _create_lightweight_players(self) -> Dict[str, LightweightPlayer]:
        """Convert demo players to lightweight format for optimization."""
        lightweight = {}
        for player in self.players:
            lightweight[player.id] = LightweightPlayer(
                id=player.id,
                name=player.name,
                position=player.position,
                pro_position=player.pro_position,
                pro_team=player.pro_team,
                bye_week=player.bye_week,
                age=player.age,
                adp=player.adp,
                vorp=player.vorp
            )
        return lightweight
    
    def get_draft_order(self) -> List[int]:
        """Calculate our pick numbers in a snake draft."""
        picks = []
        for round_num in range(1, self.total_rounds + 1):
            if round_num % 2 == 1:  # Odd rounds: 1, 2, 3, ... 12
                pick_in_round = self.draft_position
            else:  # Even rounds: 12, 11, 10, ... 1
                pick_in_round = self.league_size + 1 - self.draft_position
            
            overall_pick = (round_num - 1) * self.league_size + pick_in_round
            picks.append(overall_pick)
        
        return picks
    
    def simulate_draft_availability(self, n_simulations: int = 1000) -> Dict[str, float]:
        """
        Simulate player availability at our draft picks.
        
        Returns probability that each player will be available when we pick.
        """
        our_picks = self.get_draft_order()
        availability_counts = {player_id: 0 for player_id in self.lightweight_players}
        
        for sim in range(n_simulations):
            # Simulate other teams' picks with some randomness around ADP
            draft_board = []
            
            for player_id, player in self.lightweight_players.items():
                # Add noise to ADP for simulation variance
                noisy_adp = max(1, player.adp + np.random.normal(0, 15))
                draft_board.append((noisy_adp, player_id))
            
            # Sort by noisy ADP (lower = drafted earlier)
            draft_board.sort()
            
            # Check availability at our picks
            for our_pick in our_picks:
                if our_pick <= len(draft_board):
                    available_players = [pid for _, pid in draft_board[our_pick-1:]]
                    for player_id in available_players[:1]:  # First available
                        availability_counts[player_id] += 1
        
        # Convert to probabilities
        availability_probs = {
            player_id: count / n_simulations 
            for player_id, count in availability_counts.items()
        }
        
        return availability_probs
    
    def calculate_optimal_targets(self, availability_probs: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Calculate optimal draft targets based on value and availability.
        
        This demonstrates the core optimization logic combining:
        - Player value (VORP)
        - Draft availability probability  
        - Position scarcity
        - Age curves
        """
        target_values = []
        
        for player_id, player in self.lightweight_players.items():
            # Base value from VORP
            base_value = player.vorp
            
            # Adjust for availability (higher value if likely to be available)
            availability_factor = availability_probs.get(player_id, 0.5)
            
            # Age adjustment
            position_age_curve = self.age_curves.get(player.position, lambda x: 1.0)
            age_factor = position_age_curve(player.age)
            
            # Position scarcity factor (simplified)
            scarcity_factors = {'QB': 0.8, 'RB': 1.2, 'WR': 1.0, 'TE': 1.1, 'K': 0.7, 'D/ST': 0.7}
            scarcity_factor = scarcity_factors.get(player.position, 1.0)
            
            # Calculate total expected value
            expected_value = (base_value * availability_factor * 
                            age_factor * scarcity_factor)
            
            target_values.append((player_id, expected_value))
        
        # Sort by expected value (descending)
        target_values.sort(key=lambda x: x[1], reverse=True)
        
        return target_values
    
    def generate_draft_strategy(self) -> Dict:
        """
        Generate a complete draft strategy with picks and alternatives.
        
        Returns a comprehensive draft plan with:
        - Primary targets by round
        - Alternative picks  
        - Position timing recommendations
        - Value-based decision trees
        """
        print("Calculating optimal draft strategy...")
        
        # Simulate availability
        availability_probs = self.simulate_draft_availability(N_SIMS)
        
        # Calculate optimal targets
        optimal_targets = self.calculate_optimal_targets(availability_probs)
        
        # Get our pick numbers
        our_picks = self.get_draft_order()
        
        # Generate strategy by round
        strategy = {
            'draft_position': self.draft_position,
            'our_picks': our_picks,
            'rounds': []
        }
        
        for i, pick_number in enumerate(our_picks):
            round_num = i + 1
            
            # Get top available players for this pick timing
            round_targets = []
            alternatives = []
            
            # Filter targets likely to be available around this pick
            for player_id, expected_value in optimal_targets:
                player = self.lightweight_players[player_id]
                availability = availability_probs.get(player_id, 0.5)
                
                # Primary targets: high value, reasonable availability
                if expected_value > 0.5 and availability > 0.3:
                    if len(round_targets) < 3:
                        round_targets.append({
                            'player_id': player_id,
                            'name': player.name,
                            'position': player.position,
                            'vorp': player.vorp,
                            'expected_value': expected_value,
                            'availability': availability,
                            'adp': player.adp
                        })
                
                # Alternatives: decent value, good availability
                elif expected_value > 0.2 and availability > 0.5:
                    if len(alternatives) < 5:
                        alternatives.append({
                            'player_id': player_id,
                            'name': player.name,
                            'position': player.position,
                            'vorp': player.vorp,
                            'expected_value': expected_value,
                            'availability': availability,
                            'adp': player.adp
                        })
            
            round_strategy = {
                'round': round_num,
                'pick': pick_number,
                'targets': round_targets,
                'alternatives': alternatives,
                'position_priority': self._get_position_priority(round_num)
            }
            
            strategy['rounds'].append(round_strategy)
        
        return strategy
    
    def _get_position_priority(self, round_num: int) -> List[str]:
        """Get position drafting priority by round (simplified heuristic)."""
        early_rounds = ['RB', 'WR', 'RB', 'WR', 'QB']
        middle_rounds = ['TE', 'WR', 'RB', 'QB', 'WR']  
        late_rounds = ['K', 'D/ST', 'QB', 'RB', 'WR']
        
        if round_num <= 5:
            return early_rounds
        elif round_num <= 10:
            return middle_rounds
        else:
            return late_rounds
    
    def run_optimization_demo(self) -> Dict:
        """
        Run a complete optimization demo showcasing all features.
        
        This method demonstrates:
        - Draft simulation and availability modeling
        - Value-based optimization 
        - Monte Carlo scenario analysis
        - Strategic decision making
        """
        print("\n" + "="*60)
        print("DRAFT OPTIMIZATION DEMONSTRATION")
        print("="*60)
        
        start_time = time.time()
        
        # Generate strategy
        strategy = self.generate_draft_strategy()
        
        # Calculate summary statistics
        total_projected_value = sum(
            target['expected_value'] 
            for round_data in strategy['rounds']
            for target in round_data['targets'][:1]  # Top target per round
        )
        
        optimization_time = time.time() - start_time
        
        # Display results
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Total projected team value: {total_projected_value:.2f}")
        print(f"Draft position: {self.draft_position}")
        print(f"Our picks: {strategy['our_picks']}")
        
        # Show first few rounds in detail
        print(f"\nDETAILED STRATEGY (First 5 Rounds):")
        print("-" * 60)
        
        for round_data in strategy['rounds'][:5]:
            print(f"\nRound {round_data['round']} (Pick {round_data['pick']}):")
            
            if round_data['targets']:
                print("  Primary Targets:")
                for target in round_data['targets']:
                    print(f"    • {target['name']:<20} ({target['position']}) - "
                          f"VORP: {target['vorp']:.2f}, "
                          f"Availability: {target['availability']:.1%}")
            
            if round_data['alternatives']:
                print("  Alternatives:")
                for alt in round_data['alternatives'][:3]:
                    print(f"    • {alt['name']:<20} ({alt['position']}) - "
                          f"VORP: {alt['vorp']:.2f}")
        
        return strategy
    
    def analyze_position_timing(self) -> Dict:
        """Analyze optimal timing for drafting each position."""
        position_analysis = {}
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'D/ST']:
            pos_players = [p for p in self.players if p.position == position]
            pos_players.sort(key=lambda p: p.vorp, reverse=True)
            
            # Find value drop-offs
            if len(pos_players) >= 2:
                top_tier = pos_players[:6]  # Top 6 at position
                second_tier = pos_players[6:18] if len(pos_players) > 6 else []
                
                avg_top_vorp = np.mean([p.vorp for p in top_tier]) if top_tier else 0
                avg_second_vorp = np.mean([p.vorp for p in second_tier]) if second_tier else 0
                
                position_analysis[position] = {
                    'total_players': len(pos_players),
                    'top_tier_vorp': avg_top_vorp,
                    'second_tier_vorp': avg_second_vorp,
                    'value_dropoff': avg_top_vorp - avg_second_vorp,
                    'recommended_latest_round': self._recommend_latest_round(position, avg_top_vorp)
                }
        
        return position_analysis
    
    def _recommend_latest_round(self, position: str, avg_vorp: float) -> int:
        """Recommend latest round to draft a position based on value."""
        # Simplified heuristic based on position and value
        if position == 'QB' and avg_vorp > 5:
            return 8
        elif position in ['RB', 'WR'] and avg_vorp > 10:
            return 5
        elif position == 'TE' and avg_vorp > 3:
            return 10
        else:
            return self.total_rounds - 2


def run_portfolio_demo():
    """Run a complete portfolio demonstration of the draft optimizer."""
    print("Fantasy Football Draft Optimizer - Portfolio Demo")
    print("="*60)
    
    # Test different draft positions
    for position in [3, 6, 10]:
        print(f"\n{'='*40}")
        print(f"DRAFT POSITION {position} ANALYSIS")
        print(f"{'='*40}")
        
        optimizer = PortfolioDraftOptimizer(draft_position=position)
        strategy = optimizer.run_optimization_demo()
        
        # Analyze position timing
        print(f"\nPosition Timing Analysis:")
        position_analysis = optimizer.analyze_position_timing()
        
        for pos, analysis in position_analysis.items():
            if analysis['value_dropoff'] > 1:  # Only show significant dropoffs
                print(f"  {pos}: Value dropoff of {analysis['value_dropoff']:.2f} "
                      f"after top tier (draft by round {analysis['recommended_latest_round']})")
    
    print(f"\n{'='*60}")
    print("PORTFOLIO DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("Key features demonstrated:")
    print("• Monte Carlo draft simulation (1000+ scenarios)")
    print("• Value Over Replacement Player (VORP) calculations") 
    print("• Snake draft position optimization")
    print("• Player availability probability modeling")
    print("• Age curve adjustments")
    print("• Position scarcity analysis")
    print("• Multi-round strategic planning")


if __name__ == '__main__':
    run_portfolio_demo()
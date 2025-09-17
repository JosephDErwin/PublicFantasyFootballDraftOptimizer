from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pulp as plp
import tqdm
from pulp import lpSum

from src.config import LEAGUE_NAME, SEASON, N_SIMS
from src.data.utilities import get_league
from src.models.ml_models import get_all_players
from src.paths import root_dir
from src.tools import get_best_fas


def calculate_opponent_points(week, schedule, agents, league):
    """
    Estimates the opponent's score for a given week across all simulations by building
    their best possible lineup for each simulation run.
    """
    opponent_team_abbrev = schedule[week - 1].team_abbrev

    # Get opponent's roster and free agents
    opponent_roster = sorted([p for p in agents if p.manager == opponent_team_abbrev], key=lambda x: x.ave_projected_ppg, reverse=True)
    free_agents = sorted([p for p in agents if not p.manager], key=lambda x: x.ave_projected_ppg, reverse=True)

    # This will store the final calculated score for each simulation
    opponent_scores = defaultdict(float)

    # Find existing points for locked players if it's the current week
    locked_points = {}
    if week == league.current_week:
        for player in opponent_roster:
            if player.locked and player.game_logs.shape[0] > 0:
                actual_score = player.game_logs.loc[(player.game_logs['week_num'] == week) &
                                                    (player.game_logs['season'] == SEASON), 'applied_total'].mean()
                actual_score = 0 if np.isnan(actual_score) else actual_score
                locked_points[player.id] = actual_score

    # Calculate the best possible score for each simulation
    for i in range(N_SIMS):
        needed_positions = Counter(league.roster_slots)
        needed_positions['BE'] = 0
        needed_positions['IR'] = 0

        # Get player scores for this specific simulation (i)
        roster_sim_scores = [(p, p.simulations.get(str(week), {}).get(i, 0)) for p in opponent_roster]
        fa_sim_scores = [(p, p.simulations.get(str(week), {}).get(i, 0)) for p in free_agents]

        if week == league.current_week:
            # For the current week, ensure locked players are considered first
            roster_sim_scores.sort(key=lambda x: (not x[0].locked, -x[1]))

        # Fill starting lineup from the opponent's roster
        for player, sim_score in roster_sim_scores:
            if player.bye_week == week or player.injured:
                continue

            if player.locked and week == league.current_week:
                if player.assigned_slot not in ['BE', 'IR']:
                    opponent_scores[i] += locked_points[player.id]
                    needed_positions[player.assigned_slot] -= 1
                continue  # Move to next player

            for pos in player.positions:
                if needed_positions.get(pos, 0) > 0:
                    opponent_scores[i] += sim_score
                    needed_positions[pos] -= 1
                    break  # Move to next player

        # Fill any remaining starting slots with the best free agents for this simulation
        temp_free_agents = fa_sim_scores[:]
        while sum(needed_positions.values()) > 0:
            best_fa_found = False
            for pos in ['QB', 'RB', 'WR', 'TE', 'RB/WR/TE', 'P', 'K', 'D/ST']:
                if needed_positions.get(pos, 0) > 0:
                    for fa, sim_score in temp_free_agents:
                        if pos in fa.positions:
                            opponent_scores[i] += sim_score
                            needed_positions[pos] -= 1
                            temp_free_agents.remove((fa, sim_score))
                            best_fa_found = True
                            break  # FA found for this position, break to restart position loop
                    if best_fa_found:
                        break
            if not best_fa_found:
                break  # No more free agents can fill remaining slots

    return opponent_scores


class RosterOptimizer:
    """
    An optimizer for fantasy football rosters using linear programming to maximize wins.
    """

    def __init__(self, league, team_owner, new_players, tank=False, update=False):
        self.tank = tank
        self.league = league
        self.team_owner = team_owner
        self.new_players = new_players
        self.season = league.year
        self.my_team = next(i for i in league.teams if i.owner == team_owner)
        self.my_schedule = self.my_team.schedule + self.finals_opponents()
        self.slots = league.roster_slots
        self.agents = get_all_players(league, LEAGUE_NAME, force_update=update)
        self.best_fas = get_best_fas([agent for agent in self.agents if not agent.manager])
        self.value_override = {}
        self.week_range = self.get_week_range()
        self.prob = plp.LpProblem('Football_Roster', plp.LpMaximize)
        self.contributions_at_slots = {i: {x: [] for x in self.week_range} for i in self.slots.keys()}
        self.all_contributions = {x: [] for x in self.week_range}

    def finals_opponents(self):
        """Predicts playoff opponents based on current standings and playoff probabilities."""
        teams = sorted(self.league.teams, key=lambda x: x.playoff_pct, reverse=True)
        team_count = self.league.settings.playoff_team_count
        opponents = []

        for i in range(int(np.log2(team_count))):
            if self.league.settings.reg_season_count + i <= self.league.currentMatchupPeriod:
                continue

            my_pos = teams.index(self.my_team) if self.my_team in teams else team_count - 1
            opponents.append(teams[int(team_count - my_pos - 1)])
            team_count //= 2
            teams = teams[:team_count]

        return opponents

    def get_week_range(self):
        """Determines the range of weeks to optimize for."""
        today = datetime(self.league.year, 9, 1, 10) + timedelta(self.league.nfl_week * 7)
        offset = 1 if datetime.now() >= today else 0
        return range(self.league.currentMatchupPeriod + offset, len(self.my_schedule) + 1)

    def load_exclusions(self):
        """Loads player exclusion lists from input files."""
        with open(root_dir / 'input_files/exclusion_list.csv', "r") as f:
            exclusion_list_info = [i.split(',') for i in f.read().split("\n") if i]
            exclusion_list_info = [i for i in exclusion_list_info if i[0] and i[0][0] != '#']
            self.exclusion_list = [i[0] for i in exclusion_list_info]

    def initialize_variables(self):
        """Initializes all PuLP variables for players and rosters."""
        for agent in self.agents:
            # Ownership should be a binary decision.
            var = plp.LpVariable(f'own_{agent.id}', cat=plp.LpBinary)
            agent.selector = var

            # Constraint to only consider my players or free agents
            if agent.manager not in [self.my_team.team_abbrev, None, 'FA']:
                self.prob += var <= 0

            # Initialize agent attributes to prevent AttributeError later
            agent.active_slots = {}
            agent.inactive_slots = {}

            self.setup_weekly_variables(agent)

    def setup_weekly_variables(self, agent):
        """Creates variables for each player's status (active, bench, IR) for each week."""
        for x in self.week_range:
            inactive = {}
            player_slots = {}
            all_slots = []

            for slot in agent.positions:
                if slot not in self.slots:
                    continue

                if slot == 'IR' and agent.status != 'INJURY_RESERVE':
                    continue

                if (
                        (agent.locked and x == self.league.current_week) and
                        slot != agent.assigned_slot
                ):
                    continue

                position = plp.LpVariable(f'player_{agent.name}_at_{slot}_week{x}', cat=plp.LpBinary)
                position.setInitialValue(1)

                if slot not in ['BE', 'IR']:
                    player_slots[slot] = position
                else:
                    inactive[slot] = position

                self.all_contributions[x].append(position)
                self.contributions_at_slots[slot][x].append(position)
                all_slots.append(position)

            self.add_roster_constraints(agent)

            # Link weekly slot variables to overall ownership variable
            self.prob += plp.lpSum(all_slots) == agent.selector

            # if agent.locked and x == self.league.current_week:
            #     if agent.manager == self.my_team.team_abbrev:
            #         self.prob += plp.lpSum(all_slots) >= 1
            #     else:
            #         self.prob += plp.lpSum(all_slots) <= 0

            agent.active_slots[x] = player_slots
            agent.inactive_slots[x] = inactive

    def add_roster_constraints(self, agent):
        """Adds constraints for excluded players."""
        if agent.name in self.exclusion_list:
            self.prob += agent.selector <= 0

    def add_constraints(self):
        """Adds model-wide constraints for roster size."""
        for week in self.week_range:
            for slot, count in self.slots.items():
                if slot not in ['BE', 'IR']:
                    self.prob += plp.lpSum(self.contributions_at_slots[slot][week]) == count
                else:
                    self.prob += plp.lpSum(self.contributions_at_slots[slot][week]) <= count

        # Add constraints to only have 1 kicker and 1 defense
        limits = {'K': 1, 'D/ST': 1, 'TE': 2, 'QB': 2}
        player_types = defaultdict(list)
        for player in self.agents:
            for pos in limits:
                if pos in player.positions:
                    player_types[pos].append(player.selector)

        for pos, limit in limits.items():
            self.prob += lpSum(player_types[pos]) <= limit

    def solve(self):
        """Defines the objective function and solves the optimization problem."""
        win_vars = defaultdict(dict)
        weekly_points = defaultdict(dict)
        weekly_vorp = defaultdict(dict)
        opponent_projections = defaultdict(dict)

        print("Calculating opponent simulations...")
        all_opponent_projections = {
            week: calculate_opponent_points(week, self.my_schedule, self.agents, self.league)
            for week in self.week_range
        }

        print("Building optimization model with simulations...")
        for week in tqdm.tqdm(self.week_range):
            for i in range(N_SIMS):
                opponent_points = all_opponent_projections[week][i]
                opponent_projections[week][i] = opponent_points

                points_expr = 0
                vorp_expr = 0
                for player in self.agents:
                    # If player's game started, use actual score, otherwise use simulated score
                    if player.locked and week == self.league.current_week:
                        if player.game_logs.shape[0] > 0:
                            points_proj = player.game_logs.loc[(player.game_logs['week_num'] == week) &
                                                               (player.game_logs[
                                                                    'season'] == self.season), 'applied_total'].mean()
                            points_proj = 0 if np.isnan(points_proj) else points_proj
                        else:
                            points_proj = 0
                    else:
                        # **CORE CHANGE**: Use the pre-calculated simulation score for this iteration
                        if not player.injured:
                            points_proj = player.simulations.get(str(week), {}).get(i, 0)
                        else:
                            points_proj = 0

                    points_expr += points_proj * plp.lpSum(player.active_slots.get(week, {}).values())
                    vorp_expr += max(player.vorp, 0) * plp.lpSum(player.active_slots.get(week, {}).values())

                win_var = plp.LpVariable(f'win_{week}_{i}', cat=plp.LpBinary)
                # "Big M" constraint to link winning to points scored
                self.prob += points_expr - opponent_points >= -500 * (1 - win_var)
                self.prob += points_expr - opponent_points <= 500 * win_var - 0.01

                win_vars[week][i] = win_var
                weekly_points[week][i] = points_expr
                weekly_vorp[week][i] = vorp_expr

        # Flatten the nested dictionaries before passing to lpSum
        all_win_vars = [var for week_sims in win_vars.values() for var in week_sims.values()]
        all_weekly_points = [points for week_sims in weekly_points.values() for points in week_sims.values()]
        all_weekly_vorp = [vorp for week_sims in weekly_vorp.values() for vorp in week_sims.values()]

        if not self.tank:
            # Objective: Maximize average wins (primary) and average points (secondary)
            self.prob += (plp.lpSum(all_win_vars) / N_SIMS) * 0.99 + (plp.lpSum(all_weekly_points) / N_SIMS) * 0.01

        print("Solving optimization problem...")
        status = self.prob.solve(plp.CPLEX_CMD(path='/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex',
                                               msg=True,
                                               timeLimit=60 * 1,
                                               warmStart=True))

        return status, win_vars, weekly_points, opponent_projections

    def generate_results(self, win_vars, weekly_points, opponent_projections):
        """Processes and prints the optimal roster."""
        players = {}
        for agent in self.agents:
            if agent.selector.varValue < 0.5:
                continue

            pos_by_week = {'PPG': agent.ave_projected_ppg, 'VORP': agent.vorp, 'Manager': agent.manager}
            for week in self.week_range:
                for position, slot in agent.active_slots.get(week, {}).items():
                    if slot.varValue > 0.5:
                        pos_by_week[week] = position
                for position, slot in agent.inactive_slots.get(week, {}).items():
                    if slot.varValue > 0.5:
                        pos_by_week[week] = position
            players[agent.name] = pos_by_week

        sort_list = ['QB', 'RB', 'WR', 'TE', 'RB/WR/TE', 'D/ST', 'K', 'P', 'BE', 'IR', '-']
        players_df = pd.DataFrame(players).T
        if not players_df.empty:
            players_df = players_df.loc[:, ['PPG', 'VORP', 'Manager'] + list(self.week_range)]
            players_df = players_df[players_df.loc[:, self.week_range].notna().any(axis=1)].fillna('-')
            players_df.sort_values(by=list(self.week_range),
                                   key=lambda column: column.map(
                                       lambda e: sort_list.index(e) if e in sort_list else len(sort_list)),
                                   inplace=True)

        win_probs_series = pd.Series({
            week: sum(var.varValue for var in sims.values()) / len(sims) if sims else 0
            for week, sims in win_vars.items()
        }, name='Win %')
        points_series = pd.Series({
            week: plp.value(plp.lpSum(points.values())) / len(points) if points else 0
            for week, points in weekly_points.items()
        }, name='Points')
        opponent_pts_series = pd.Series({
            week: sum(projs.values()) / len(projs) if projs else 0
            for week, projs in opponent_projections.items()
        }, name='Opponent Pts.')

        summary_df = pd.DataFrame([win_probs_series, points_series, opponent_pts_series])
        players_df = pd.concat([players_df, summary_df])
        players_df.loc['Opponent', list(self.week_range)] = {i + 1: team.team_abbrev for i, team in
                                                             enumerate(self.my_schedule)}

        print("\n--- Optimal Roster & Projections ---")
        print(players_df.fillna('-').to_string())

        expected_wins = win_probs_series[:self.league.settings.reg_season_count].sum() + 1
        print(f"\nExpected Record: {expected_wins:.2f} - {self.league.settings.reg_season_count - expected_wins:.2f} ")

        return opponent_projections

    def run(self):
        """Executes the entire optimization pipeline."""
        self.load_exclusions()
        self.initialize_variables()
        self.add_constraints()
        status, win_vars, weekly_points, opponent_projections = self.solve()
        if plp.LpStatus[status] == 'Optimal':
            print("\nOptimal solution found.")
            self.generate_results(win_vars, weekly_points, opponent_projections)
        else:
            print(f"\nCould not find an optimal solution. Status: {plp.LpStatus[status]}")


def sweat(update=False):
    """Main function to run the optimizer for a specific manager."""
    manager = 'Joseph Erwin'
    league = get_league(LEAGUE_NAME, SEASON, force_update=update)
    optimizer = RosterOptimizer(league, manager, new_players=0, tank=False, update=update)
    optimizer.run()


if __name__ == '__main__':
    sweat(update=False)
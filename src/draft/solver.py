from collections import defaultdict

import pulp

# Assuming these are defined elsewhere
from src.constants import GAMES_IN_SEASON
from src.data.utilities import calculate_keeper_value


# --- Helper function for pickling ---
# This top-level function is used to create nested defaultdicts. Because it's a
# named function at the module level, it can be pickled by the multiprocessing
# library, unlike a lambda function defined inside a method.
def _nested_defaultdict_factory():
    return defaultdict(dict)


class DraftMilpSolver:
    """
    An object-oriented wrapper for a multi-round, lookahead draft MILP model,
    refactored to be compatible with Python's multiprocessing module.

    --- REFACTOR NOTES FOR MULTIPROCESSING ---
    1.  The __init__ method is now lightweight. It only stores the raw, picklable
        data (dictionaries, lists) needed to build the model. It does NOT
        create any PuLP objects.

    2.  A new `_build_model` method contains the original model construction logic.

    3.  The `solve` method now uses lazy initialization: it checks if the PuLP
        model has been built (`self.problem is None`) and calls `_build_model`
        if needed. This ensures the complex, unpicklable PuLP objects are
        created within each worker process, not during the initial creation.
    """

    def __init__(self, player_pool, roster_slots, my_players_ids):
        """
        Initializes the solver by storing the data required for the MILP.

        This method is lightweight and does not create the PuLP model itself,
        making the object picklable and safe to pass to multiprocessing workers.

        Args:
            player_pool (dict): {player_id: Player_object} for all players.
            roster_slots (dict): The full roster configuration for a team.
            my_players_ids (dict): A dict of player_ids already on our team.
        """
        # Store all the raw data needed to build the model
        self.player_pool = player_pool
        self.roster_slots = roster_slots
        self.my_players_ids = my_players_ids

        # Initialize attributes that will hold the PuLP model components.
        # These will be populated by `_build_model` within the worker process.
        self.problem = None
        self.player_vars = {}
        # FIX: Use a named function instead of a lambda for pickling compatibility.
        self.assignment_vars = defaultdict(_nested_defaultdict_factory)
        # self.player_starts_expressions = defaultdict(pulp.LpAffineExpression)
        self.player_starts_vars = defaultdict(lambda: defaultdict(list))

        self.pos_deviations = {}

    def _build_model(self):
        """
        Creates the PuLP problem, decision variables, and static constraints.
        This method is called internally by `solve` on its first run.
        """
        self.problem = pulp.LpProblem("FullSeasonDraftOptimization", pulp.LpMaximize)

        # --- Define Variables & Pre-calculate Start Expressions ---
        override_maxes = {'D/ST': 1, 'K': 1, 'QB': 2, 'TE': 2}
        override_lists = defaultdict(list)
        for p_id, p_ob in self.player_pool.items():
            self.player_vars[p_id] = pulp.LpVariable(f"player_{p_id}", cat='Binary')

            for pos in override_maxes:
                if pos in p_ob.positions:
                    override_lists[pos].append(self.player_vars[p_id])

            for week in range(1, GAMES_IN_SEASON + 1):
                for pos in p_ob.positions:
                    if pos not in self.roster_slots or pos == 'IR':
                        continue

                    if p_ob.bye_week == week and pos not in ['BE', 'IR']:
                        continue

                    var = pulp.LpVariable(f"assign_{p_id}_{pos.replace('/', '_')}_{week}", cat='Binary')
                    self.assignment_vars[p_id][pos][week] = var

                    if pos not in ['BE', 'IR']:
                        # self.player_starts_expressions[p_id] += var
                        self.player_starts_vars[p_id][week].append(var)

        for pos, max_desired in override_maxes.items():
            total_players = override_lists[pos]
            if total_players:
                self.pos_deviations[pos] = pulp.LpVariable(f"deviation_{pos}", lowBound=0)
                self.problem += pulp.lpSum(total_players) <= max_desired + self.pos_deviations[pos], f"Total_{pos.replace('/', '_')}"

        # --- Linking Constraints ---
        for p_id in self.player_pool:
            for week in range(1, GAMES_IN_SEASON + 1):
                weekly_assignments = [
                    var for pos, weekly_vars in self.assignment_vars[p_id].items()
                    for w, var in weekly_vars.items() if w == week
                ]
                if weekly_assignments:
                    self.problem += pulp.lpSum(weekly_assignments) == self.player_vars[p_id], \
                        f"WeeklyAssignmentLink_{p_id}_{week}"

        # --- Weekly Roster Slot Constraints ---
        for week in range(1, GAMES_IN_SEASON + 1):
            for position, count_needed in self.roster_slots.items():
                if position == 'BE': continue
                weekly_assignments_for_pos = [
                    self.assignment_vars[p_id][position].get(week)
                    for p_id in self.player_pool
                    if self.assignment_vars[p_id].get(position, {}).get(week) is not None
                ]
                if weekly_assignments_for_pos:
                    self.problem += pulp.lpSum(weekly_assignments_for_pos) <= count_needed, \
                        f"WeeklyRoster_{position.replace('/', '_')}_{week}"

        # --- Total Roster Size Constraint ---
        total_slots = sum(v for k, v in self.roster_slots.items() if k != 'IR')
        self.problem += pulp.lpSum(self.player_vars.values()) == total_slots, "TotalRosterSize"

        # --- Pre-Drafted Player Constraint ---
        for p_id in self.my_players_ids.values():
            if p_id in self.player_vars:
                self.problem += self.player_vars[p_id] == 1, f"PreDrafted_{p_id}"

    def solve(self, scenario_points, scenario_games_played, availability_probs, age_curve, pick_vals, max_time=20):
        """
        Updates and solves the model, limiting the lookahead horizon for speed.
        """
        if self.problem is None:
            self._build_model()

        temp_problem = self.problem.copy()

        player_at_pick = defaultdict(dict)

        # --- 1. Reduce the Lookahead Horizon ---
        # Define how many of your future picks you want to plan for.
        # A value of 4-5 is usually a good balance of foresight and speed.
        LOOKAHEAD_PICKS = 4
        KEEPER_DISCOUNT = 0.15
        KEEPER_OBJECTIVE_WEIGHT = 0.2
        TEAMS_IN_LEAGUE = 12  # Should be a parameter for the class or method

        # Get your future picks in chronological order and slice them
        future_picks_sorted = sorted([pick for pick in availability_probs.keys() if pick not in self.my_players_ids.keys()])
        picks_in_horizon = future_picks_sorted[:LOOKAHEAD_PICKS]

        # Create a new, smaller dictionary containing only the picks in the horizon
        filtered_availability_probs = {
            pick: availability_probs[pick] for pick in picks_in_horizon
        }

        player_starts_expressions = {}
        for player_id in self.player_pool:
            # Also impose games_played restriction
            player_starts_expressions[player_id] = pulp.lpSum(
                [pulp.lpSum(vars) * scenario_games_played[player_id][week_num - 1] for week_num, vars in
                 self.player_starts_vars[player_id].items()])

        # --- 2. Build the Model Using Only the Filtered Probs ---
        by_player = defaultdict(list)
        # Use the filtered dictionary to create pick variables
        for pick, player_probs in filtered_availability_probs.items():
            round_picks = []
            for p_id in player_probs.keys():
                if p_id in self.my_players_ids: continue
                pick_var = pulp.LpVariable(f"pick_{p_id}_at_{pick}", cat='Binary')
                temp_problem += pick_var <= self.player_vars[p_id], f"PickLink_{p_id}_at_{pick}"
                player_at_pick[pick][p_id] = pick_var
                by_player[p_id].append(pick_var)
                round_picks.append(pick_var)
            if round_picks:
                temp_problem += pulp.lpSum(round_picks) == 1, f"PickOne_{pick}"

        for player_id, pick_vars in by_player.items():
            temp_problem += pulp.lpSum(pick_vars) <= 1, f"PickOnce_{player_id}"

        objective = pulp.LpAffineExpression()

        keepers = []

        # Add value from players already on our team
        for pick_round, p_id in enumerate(self.my_players_ids.values()):
            if p_id in self.player_pool:
                points_per_start = scenario_points.get(p_id, 0)
                keeper_val = calculate_keeper_value(pick_round * TEAMS_IN_LEAGUE + TEAMS_IN_LEAGUE // 2,
                                                    self.player_pool[p_id],
                                                    points_per_start,
                                                    age_curve, pick_vals,
                                                    TEAMS_IN_LEAGUE,
                                                    KEEPER_DISCOUNT)

                keeper_var = pulp.LpVariable(f"keeper_{p_id}_at_{pick_round}", cat='Binary')
                keepers.append(keeper_var)

                objective += keeper_val * keeper_var * KEEPER_OBJECTIVE_WEIGHT
                if points_per_start > 0:
                    objective += player_starts_expressions[p_id] * points_per_start

        # Use the filtered dictionary to build the objective function
        for pick, player_probs in filtered_availability_probs.items():
            for p_id, prob_available in player_probs.items():
                if p_id in self.my_players_ids: continue

                p = self.player_pool[p_id]
                p_ave_points = scenario_points.get(p_id, 0)

                value_from_starts = player_starts_expressions[p_id] * p_ave_points
                pick_var = player_at_pick[pick][p_id]
                keeper_var = pulp.LpVariable(f"keeper_{p_id}_at_{pick}", cat='Binary')

                temp_problem += keeper_var <= pick_var, f"KeepLink_{p_id}_at_{pick}"
                keepers.append(keeper_var)

                if pick > min(player_at_pick.keys()):
                    v_pk = pulp.LpVariable(f"val_{p_id}_at_{pick}", lowBound=0)

                    # Use a tighter, player-specific M
                    M = max(p_ave_points, 2) * GAMES_IN_SEASON

                    temp_problem += v_pk <= M * pick_var, f"Lin1_{p_id}_{pick}"
                    temp_problem += v_pk <= value_from_starts, f"Lin2_{p_id}_{pick}"
                    temp_problem += v_pk >= value_from_starts - M * (1 - pick_var), f"Lin3_{p_id}_{pick}"

                    objective += v_pk * prob_available
                else:
                    objective += value_from_starts

                objective += - max(p.adp - pick, 0) ** 2 * 0.01 * pick_var

                if p.pro_position in age_curve:
                    keeper_value = calculate_keeper_value(pick, p, p_ave_points, age_curve, pick_vals,
                                                          teams_in_league=TEAMS_IN_LEAGUE,
                                                          keeper_discount=KEEPER_DISCOUNT)

                    objective += prob_available * keeper_value * keeper_var * KEEPER_OBJECTIVE_WEIGHT

        temp_problem += pulp.lpSum(keepers) <= 2, "KeepTwo"

        temp_problem.setObjective(objective + pulp.lpSum(self.pos_deviations.values()) * -1e5)

        # solver = pulp.CPLEX_CMD(path='/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex',
        #                         msg=False,
        #                         gapRel=0.05,
        #                         timeLimit=30,
        #                         warmStart=True
        #                         )

        solver = pulp.HiGHS(timeLimit=max_time, gapRel=0.075, msg=False, threads=8)
        temp_problem.solve(solver)

        if temp_problem.status == pulp.LpStatusOptimal:
            if not player_at_pick:
                return None
            next_pick_to_make = min(player_at_pick.keys())
            for p_id, var in player_at_pick[next_pick_to_make].items():
                if var.varValue is not None and var.varValue > 0.99:
                    solution_attributes = {
                        'solutionTime': temp_problem.solutionTime,
                        'objectiveValue': temp_problem.objective.value(),
                    }

                    return p_id, solution_attributes
            return None, {}
        else:
            return None, {}
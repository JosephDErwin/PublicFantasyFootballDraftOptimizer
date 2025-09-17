import time
from collections import namedtuple
from pathlib import Path

import pandas as pd
import tqdm

from src.config import SEASON, LEAGUE_NAME, TEAM_ID
from src.data.utilities import get_league, generate_correlated_samples
from src.draft.opponent_picks import PlayerAvailabilityModel
from src.draft.solver import DraftMilpSolver
from src.models.draft_ml import get_correlation_mat, construct_corr_matrix, get_all_players
from src.paths import root_dir
from src.tools import get_vorp_curve, get_age_curves

# A lightweight, picklable object to hold only the essential player data
LightweightPlayer = namedtuple(
    'LightweightPlayer',
    ['name', 'positions', 'pro_position', 'pro_team', 'bye_week', 'current_age', 'adp']
)


def clean_name(name):
    replacements = {'Sr.': ''}

    for k, v in replacements.items():
        name = name.replace(k, v)

    name = name.strip()

    return name


def get_id(name, player_pool):
    for id, p in player_pool.items():
        if clean_name(p.name) == clean_name(name):
            return id
    return None


class DraftOptimizer:
    """
    A process that runs draft simulations on-demand by reading from a shared state object.
    """

    def __init__(self, players, draft_settings, shared_draft_log, results_list, opimizer_trigger, shutdown_event,
                 vorp_curve, age_curve):
        self.players = {p.id: p for p in players}
        self.lightweight_players = self.get_lightweight_players()
        self.draft_settings = draft_settings
        self.shared_draft_log = shared_draft_log
        self.results_list = results_list
        self.opimizer_trigger = opimizer_trigger
        self.shutdown_event = shutdown_event
        self.player_pool = self.players.copy()
        self.my_team_ids = {}
        self.draft_log_ids = []
        self.last_log_size = -1
        self.on_the_clock = False
        self.state_version = 0
        self.total_picks = []
        self.total_sims_run = 0
        self.vorp_curve = vorp_curve
        self.age_curve = age_curve
        self.draft_map = self._get_draft_map()
        self.my_picks = self._get_my_pick_slots()
        self.player_corr_mat = self._get_correlation_matrix()
        self.availability_model = PlayerAvailabilityModel.load(root_dir / Path('data/player_availability_model.joblib'))

    def _get_correlation_matrix(self):
        """Constructs the player-level correlation matrix."""
        game_logs = []
        for player in self.players.values():
            if player.game_logs.empty: continue
            game_logs.append(player.game_logs[player.game_logs['season'] < SEASON])

        if not game_logs:
            return pd.DataFrame()

        game_logs = pd.concat(game_logs)
        pos_corr_mat = get_correlation_mat(game_logs)
        player_corr_mat = construct_corr_matrix(self.players, pos_corr_mat)
        return player_corr_mat

    def _get_draft_map(self):
        draft_list = []
        for i in range(self.draft_settings['rounds']):
            if (i + 1) % 2 == 1:
                draft_list += self.draft_settings['teams_config']
            else:
                draft_list += self.draft_settings['teams_config'][::-1]

        draft_map = {i + 1: draft_list[i] for i in range(len(draft_list))}

        return draft_map

    def _get_my_pick_slots(self):
        """Calculates all of our pick numbers in a snake draft."""
        picks = []
        num_teams = self.draft_settings['teams']
        my_pick_in_round = self.draft_settings['my_pick']
        for i in range(self.draft_settings['rounds']):
            if (i + 1) % 2 == 1:
                pick = i * num_teams + my_pick_in_round
            else:
                pick = i * num_teams + (num_teams - my_pick_in_round + 1)
            picks.append(pick)
        return picks

    def test_run(self, BATCH_SIZE=100):
        """
        Runs a series of test simulations to validate the optimizer's functionality.
        This is useful for debugging and ensuring the optimizer behaves as expected.
        """
        print("[Optimizer] Running test simulations...")
        sample_picks = range(1, self.draft_settings['total_picks'] + 1)

        milp_solver = DraftMilpSolver(self.lightweight_players,
                                      self.draft_settings['roster_slots'],
                                      self.my_team_ids)

        scenarios, games_played = self._generate_scenarios(BATCH_SIZE)
        availability_probs = self._calculate_availability_probs()

        PICK_VALUES = dict(zip(sample_picks, self.vorp_curve(sample_picks)))

        solutions = []
        for i in tqdm.trange(len(scenarios)):
            result_pid, solution_data = milp_solver.solve(scenarios.iloc[i],
                                                          games_played[i],
                                                          availability_probs,
                                                          self.age_curve,
                                                          PICK_VALUES,
                                                          max_time=100)

            solutions.append(solution_data)

        sols_df = pd.DataFrame(solutions)

        ave_duration = sols_df['solutionTime'].mean()
        ave_vorp = sols_df['objectiveValue'].mean()

        print(f"[Optimizer] Finished test simulations.\n"
              f"    Average solve time: {ave_duration:.2f} seconds.\n"
              f"    Average VORP: {ave_vorp:.2f}.")


    def run(self):
        """
        Runs the optimizer sequentially with intelligent state checking.
        """
        print("[Optimizer] Process started in SEQUENTIAL mode.")
        milp_solver = None
        BATCH_SIZE = 200

        sample_picks = range(1, self.draft_settings['total_picks'] + 1)
        PICK_VALUES = dict(zip(sample_picks, self.vorp_curve(sample_picks)))

        state_has_changed = True
        while not self.shutdown_event.is_set():

            if state_has_changed:
                self.total_picks.clear()
                self.total_sims_run = 0
                self.results_list[:] = []
                milp_solver = DraftMilpSolver(self.lightweight_players,
                                              self.draft_settings['roster_slots'],
                                              self.my_team_ids)

            if self.on_the_clock:
                scenarios, games_played = self._generate_scenarios(BATCH_SIZE)

                if len(scenarios) > 0:
                    my_future_picks = {p: [] for p in self.my_picks if p not in list(self.my_team_ids.keys())}
                    if my_future_picks:
                        availability_probs = self._calculate_availability_probs()
                        for i in range(len(scenarios)):

                            if self.shutdown_event.is_set():
                                break

                            state_has_changed = self._update_state_from_shared_log()
                            if state_has_changed:
                                print("[Optimizer] State changed mid-batch. Aborting and restarting analysis.")
                                break

                            try:
                                result_pid = milp_solver.solve(scenarios.iloc[i],
                                                               games_played[i],
                                                               availability_probs,
                                                               self.age_curve,
                                                               PICK_VALUES,
                                                               max_time=5)
                                if result_pid is not None:
                                    self.total_picks.append(result_pid)
                                self.total_sims_run += 1
                            except Exception as e:
                                print(f"[Optimizer] ERROR during sequential solve: {e}")

                            if i > 0 and i % 5 == 0:
                                self._report_results()

                        self._report_results()
                        print(f"[Optimizer] Batch for v{self.state_version} finished.")

                time.sleep(0.1)
            else:
                state_has_changed = self._update_state_from_shared_log()
                time.sleep(1)

        print("[Optimizer] Sequential run finished.")

    def _update_state_from_shared_log(self):
        """
        Reads the shared log to update its internal state. This is the single
        source of truth for state changes.
        """
        # --- FIX: Always check the on_the_clock trigger ---
        self.on_the_clock = (self.opimizer_trigger.value == 1)

        draft_log_copy = list(self.shared_draft_log)

        # --- FIX: A state change is defined by a change in the draft log length ---
        if len(draft_log_copy) != self.last_log_size:
            self.last_log_size = len(draft_log_copy)
            self.state_version += 1
            print(f"[Optimizer] New draft state detected (v{self.state_version}). Updating player pool.")

            self.player_pool = self.players.copy()
            self.my_team_ids = {}
            self.draft_log_ids = []

            for pick_info in draft_log_copy:
                p_id = get_id(pick_info['player_name'], self.players)
                pick_num = int(pick_info['pick_num'])
                self.draft_log_ids.append(p_id)
                if p_id in self.player_pool:
                    del self.player_pool[p_id]
                if pick_info['team_name'] == self.draft_settings['my_team_name']:
                    self.my_team_ids[pick_num] = p_id

            return True  # Signal that the state has changed

        return False  # No change

    def _report_results(self):
        """Writes the latest aggregated results to the shared dictionary."""
        if not self.total_picks: return

        new_results = [self.lightweight_players[pid].name for pid, sol in self.total_picks]

        self.results_list[:] = new_results

    def _generate_scenarios(self, n_scenarios):
        """
        Generates N correlated scenarios of player VORP estimations.
        """
        player_ids_in_pool = list(self.players.keys())
        if not player_ids_in_pool or self.player_corr_mat.empty:
            return pd.DataFrame(), []

        active_corr_matrix = self.player_corr_mat.loc[player_ids_in_pool, player_ids_in_pool]
        random_variates = generate_correlated_samples(active_corr_matrix, n_scenarios)
        if random_variates.empty:
            return pd.DataFrame(), []

        scenario_vorp = {}
        scenario_games_played_by_player = {}

        for pid, p in self.players.items():
            scenario_vorp[pid], scenario_games_played_by_player[pid] = p.sample_vorp(n_scenarios, random_variates[pid])

        pids = list(scenario_games_played_by_player.keys())
        games_arrays = scenario_games_played_by_player.values()
        games_by_scenario = zip(*games_arrays)
        structured_games_played = [
            dict(zip(pids, games_in_scenario))
            for games_in_scenario in games_by_scenario
        ]

        return pd.DataFrame(scenario_vorp), structured_games_played

    def get_lightweight_players(self):
        """
        Creates a dictionary of lightweight, picklable player objects that contain
        only the data needed by the worker processes.
        """
        player_dict = {}
        for pid, player in self.players.items():
            player_dict[pid] = LightweightPlayer(
                name=player.name,
                positions=player.positions,
                pro_position=player.pro_position,
                pro_team=player.pro_team,
                bye_week=player.bye_week,
                current_age=player.current_age,
                adp=player.adp.get(SEASON, 300)
            )
        return player_dict

    def _calculate_availability_probs(self):
        """
        Calculates the probability of each player being available at each of our
        future picks using the trained survival model.
        """
        player_ids_in_pool = list(self.player_pool.keys())
        my_future_picks = [p for p in self.my_picks if p > len(self.draft_log_ids)]

        if not player_ids_in_pool or not my_future_picks:
            return {}

        for pid in player_ids_in_pool:
            p = self.players[pid]

        player_features = []
        for pid, p in self.player_pool.items():
            p_dict = {'adp': p.adp.get(SEASON), 'player_id': p.id, 'position': p.pro_position, 'pro_team': p.pro_team}

            player_features.append(p_dict)

        feature_df = pd.DataFrame(player_features, index=player_ids_in_pool)

        surv_funcs = self.availability_model.predict_survival_functions(feature_df)

        prob_available_by_pick = {}
        for pick in my_future_picks:
            eval_time = pick - 1
            prob_available_by_pick[pick] = {
                pid: func(eval_time) for pid, func in zip(player_ids_in_pool, surv_funcs) if func(eval_time) > 0.05
            }

        return prob_available_by_pick


if __name__ == "__main__":
    # This is just a placeholder to allow the module to be run directly for testing purposes.
    print("DraftOptimizer module loaded. No direct execution code provided.")
    # You can instantiate and run the DraftOptimizer here if needed for testing.
    # --- 1. Pre-load all necessary data ---
    league = get_league(LEAGUE_NAME, SEASON)
    players = get_all_players(league, LEAGUE_NAME)
    my_team_obj = league.get_team_data(TEAM_ID)

    vorp_curve = get_vorp_curve(players, SEASON)
    age_curves = get_age_curves(players)

    num_rounds = sum(league.roster_slots.values()) - 1

    draft_settings = {
        "my_pick": 6,
        "my_team_name": my_team_obj.team_name,
        "rounds": num_rounds,
        "total_picks": num_rounds * len(league.teams),
        "roster_slots": league.roster_slots,
        "teams": len(league.teams),
        "teams_config": [{'owner_name': team.owner} for team in league.teams]
    }

    optimizer = DraftOptimizer(players, draft_settings, None, None, None, None, vorp_curve, age_curves)
    optimizer.test_run()
    # Note: Ensure that the necessary data structures (players, draft_settings, etc.) are
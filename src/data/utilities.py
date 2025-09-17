#### UTILITY & DATA FETCHING FUNCTIONS ####
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import pandas as pd
from espn_api_mod.football import League
from scipy.stats import norm
from sleeper_wrapper import Players as SleeperPlayers

from src.config import S2_CODE, SWID
from src.models.player import Player
from src.paths import data_dir
from src.tools import cache_wrapper


def game_by_id(schedules):
    """Creates a lookup dictionary for game matchups by their ID."""
    lookups = {}
    for season, schedule in schedules.items():
        for team in schedule.values():
            for day in team.values():
                for game in day:
                    lookups[str(game['id'])] = game
    return lookups


@cache_wrapper(60 * 60 * 24) # Cache for 24 hours
def get_schedule(league):
    """Fetches the pro schedule for the current season."""
    info = league.espn_request.get_pro_schedule()
    schedules = {x['abbrev'].upper(): x['proGamesByScoringPeriod'] for x in info['settings']['proTeams']}
    return schedules


@cache_wrapper(60 * 60 * 24) # Cache for 24 hours
def get_old_schedule(year, league_ob):
    """Fetches and caches historical schedules."""
    dest_dir = data_dir / Path('schedules')
    dest_dir.mkdir(exist_ok=True)
    file_name = dest_dir / f'old_schedule_{year}.sav'
    if os.path.exists(file_name):
        return pickle.load(open(file_name, 'rb'))
    else:
        schedule = get_schedule(league_ob)
        pickle.dump(schedule, open(file_name, 'wb'))
        return schedule


def get_extra_players(ids, years, league, league_obs):
    """Fetches and caches detailed player card info, including game logs."""
    dest = data_dir / Path('player_logs')
    dest.mkdir(exist_ok=True)
    all_records = []
    for year in years:
        filename = dest / f'player_logs{year}.json'
        if year == league.year:
            players = league.espn_request.get_player_card(ids, max_scoring_period=league.finalScoringPeriod)['players']
            json.dump(players, open(filename, 'w'), indent=4)
        elif os.path.exists(filename):
            players = json.load(open(filename, 'r'))
        else:
            current_league = league_obs[year]
            players = league.espn_request.get_player_card(ids, max_scoring_period=current_league.finalScoringPeriod)[
                'players']
            json.dump(players, open(filename, 'w'), indent=4)
        all_records += players

    game_logs, players_map = {}, {}
    for player in all_records:
        player_id = player['id']
        players_map[player_id] = player
        game_logs.setdefault(player_id, []).extend(player['player']['stats'])
    return game_logs, players_map


@cache_wrapper(60 * 60 * 24) # Cache for 24 hours
def get_sleeper():
    """Fetches all player data from the Sleeper API."""
    return SleeperPlayers().get_all_players("nfl")


@cache_wrapper(60 * 60 * 24) # Cache for 24 hours
def get_league(league_name, season):
    """Initializes the ESPN API League object."""
    league = League(league_id=os.environ.get('ESPN_LEAGUE_ID'), year=season, swid=SWID, espn_s2=S2_CODE)
    return league


def get_old_leagues(year_range, league_name):
    """Fetches historical league objects."""
    return {year: get_league(league_name, year) for year in year_range}


def get_bye_weeks(schedules):
    byes = {}
    for team, weeks in schedules.items():
        if weeks == {}:
            continue

        int_weeks = list(map(int, weeks.keys()))
        min_week = min(int_weeks)
        max_week = max(int_weeks)

        byes[team] = list(set(range(min_week, max_week + 1)) - set(int_weeks))[0]

    return byes




def calculate_keeper_value(
        pick: int,
        player: Player,
        player_ave_points: float,
        age_curve: Dict[str, Callable[[int], float]],
        pick_vals: Dict[int, float],
        teams_in_league: int = 12,
        keeper_discount: float = 0.15,
) -> float:
    """
    Calculates the future value of keeping a player in a fantasy league.

    Args:
        pick: The current draft pick number being considered.
        player: The player object, containing their age and position.
        player_ave_points: The player's average points per game.
        age_curve: A dictionary mapping player positions to functions that model
                   point loss due to aging. The function takes an age (int) and
                   returns an age factor (float).
        pick_vals: A dictionary mapping a draft pick number (int) to its
                   expected point value (float).
        teams_in_league: The number of teams in the league.
        keeper_discount: The discount rate applied to future value.

    Returns:
        The calculated keeper value as a float.
    """
    # Use a default age if the player's current age is not available.
    player_age = player.current_age or 26

    # If the player's position is not in the age curve data, their value is 0.
    if player.pro_position not in age_curve:
        return 0

    # Get the age factor for the player's current age and position.
    current_age_factor = age_curve[player.pro_position](player_age)

    keeper_value = 0.0  # Initialize as float for consistency
    # Calculate the value for the next two years.
    for year in range(1, 3):
        # Estimate the equivalent pick in future drafts.
        future_pick = pick - year * teams_in_league
        if future_pick < 0:
            break  # Stop if the future pick is no longer on the board.

        # Calculate the expected point loss due to the player aging.
        age_loss = current_age_factor - age_curve[player.pro_position](player_age + year)

        # Calculate the value for this specific future year.
        # It's the player's projected points minus the value of the equivalent draft pick,
        # discounted back to the present.
        yearly_value = (player_ave_points + age_loss - pick_vals.get(future_pick, 0.0)) / (
                    (1 + keeper_discount) ** year)

        # The value cannot be negative.
        keeper_value += max(yearly_value, 0)

    return keeper_value


def get_correlation_mat(df):
    """
    Calculates a nuanced positional correlation matrix by first finding detailed
    role-based correlations and then performing a sample-weighted aggregation
    back into a stable, dense positional matrix.

    Args:
        df (pd.DataFrame): A DataFrame containing game-level data for all players.
                           Must include columns: ['season', 'player_id', 'team',
                           'position', 'applied_total', 'week_num'].

    Returns:
        pd.DataFrame: A dense correlation matrix where columns are positions (e.g., 'QB',
                      'RB', 'WR') and values are the weighted average correlation coefficients.
    """
    print("Calculating nuanced role-based correlations for aggregation...")

    # --- Step 1: Calculate the detailed, role-based correlation matrix ---

    player_season_totals = df.groupby(
        ['season', 'player_id', 'team', 'position']
    )['applied_total'].sum().reset_index()

    player_season_totals['pos_rank'] = player_season_totals.groupby(
        ['season', 'team', 'position']
    )['applied_total'].rank(method='first', ascending=False)

    player_season_totals['role'] = (
            player_season_totals['position'] +
            player_season_totals['pos_rank'].astype(int).astype(str)
    )
    role_mapping = player_season_totals[['season', 'player_id', 'role']]

    df_with_roles = pd.merge(df, role_mapping, on=['season', 'player_id'], how='left')
    df_with_roles.dropna(subset=['role'], inplace=True)

    pivot_df = df_with_roles.pivot_table(
        index=['season', 'team', 'week_num'],
        columns='role',
        values='applied_total'
    )

    role_corr_mat = pivot_df.corr()

    # --- Step 2: Calculate the number of observations for each correlation pair ---
    # This matrix will be used as the weights for our weighted average.
    # A boolean DataFrame (True where data exists) is created first.
    # The matrix multiplication (df.T @ df) efficiently counts pairwise co-occurrences.
    observation_counts = pivot_df.notna().T @ pivot_df.notna()

    # --- Step 3: Collapse the detailed matrix using a sample-weighted average ---

    base_positions = df['position'].unique()
    collapsed_corr = pd.DataFrame(index=base_positions, columns=base_positions, dtype=float)

    for pos1 in base_positions:
        for pos2 in base_positions:
            roles1 = [r for r in role_corr_mat.columns if r.startswith(pos1)]
            roles2 = [r for r in role_corr_mat.columns if r.startswith(pos2)]

            if not roles1 or not roles2:
                continue

            # Select the sub-matrices for both the correlations and their weights
            sub_corr_matrix = role_corr_mat.loc[roles1, roles2]
            sub_count_matrix = observation_counts.loc[roles1, roles2]

            # For intra-positional correlation (e.g., WR-WR), we must exclude the
            # diagonal (corr(WR1,WR1)=1) from the average.
            if pos1 == pos2:
                np.fill_diagonal(sub_corr_matrix.values, np.nan)
                np.fill_diagonal(sub_count_matrix.values, 0)

            # Flatten the matrices to 1D arrays, removing NaNs from the correlations
            correlations = sub_corr_matrix.values.flatten()
            weights = sub_count_matrix.values.flatten()

            valid_indices = ~np.isnan(correlations)
            correlations = correlations[valid_indices]
            weights = weights[valid_indices]

            # Calculate the weighted average if there's anything to average
            if weights.sum() > 0:
                weighted_avg = np.average(correlations, weights=weights)
                collapsed_corr.loc[pos1, pos2] = weighted_avg
            else:
                collapsed_corr.loc[pos1, pos2] = 0.0  # Default to 0 if no valid pairs

    # Fill any NaNs and ensure the matrix is symmetric with a perfect diagonal
    collapsed_corr.fillna(0, inplace=True)
    collapsed_corr = (collapsed_corr + collapsed_corr.T) / 2

    print("Aggregated correlation matrix calculation complete.")
    return collapsed_corr


def construct_corr_matrix(players, cov_mat):
    """Constructs the covariance matrix for the data."""

    position_matrices = defaultdict(dict)
    for player in players.values():
        position_matrices[player.pro_team][player.id] = player.pro_position

    player_covariance_matrix = defaultdict(dict)
    for team, players in position_matrices.items():
        for id1, position1 in players.items():
            for id2, position2 in players.items():
                player_covariance_matrix[id1][id2] = cov_mat.loc[position1, position2]

    player_covariance_matrix = pd.DataFrame(player_covariance_matrix).fillna(0)
    np.fill_diagonal(player_covariance_matrix.values, 1.0)

    return player_covariance_matrix


def make_positive_definite(corr_matrix, epsilon=1e-6):
    """
    Adjusts a matrix to be strictly positive definite using eigenvalue decomposition.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Find the smallest eigenvalue
    min_eigenvalue = np.min(eigenvalues)

    # If the smallest eigenvalue is less than our epsilon, we need to adjust
    if min_eigenvalue < epsilon:
        # Add a small offset to all eigenvalues to ensure they are all positive.
        offset = epsilon - min_eigenvalue
        eigenvalues += offset

    # Reconstruct the matrix: A = V @ diag(lambda) @ V_T
    pd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Normalize the diagonal to 1 to maintain it as a correlation matrix
    diag_inv_sqrt = np.diag(1 / np.sqrt(np.diag(pd_matrix)))
    pd_matrix = diag_inv_sqrt @ pd_matrix @ diag_inv_sqrt

    return pd_matrix


def generate_correlated_samples_weekly(corr_matrix, num_scenarios, weeks):
    """
    Generates correlated random samples for each week and combines them.

    Args:
        corr_matrix (pd.DataFrame): The correlation matrix for a single week.
                                    Assumed to be (num_players x num_players).
        num_scenarios (int): The number of scenarios to generate.
        weeks (list): A list of week numbers to include in the simulation.

    Returns:
        pd.DataFrame: A DataFrame where rows are scenarios and columns are
                      a MultiIndex of (player_id, week_num).
    """

    # Pre-compute the Cholesky factor once for efficiency
    try:
        cholesky_factor = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Correlation matrix is not positive definite. Adjusting with eigenvalue decomposition.")
        corrected_matrix = make_positive_definite(corr_matrix)
        cholesky_factor = np.linalg.cholesky(corrected_matrix)

    player_ids = corr_matrix.columns
    num_players = len(player_ids)
    all_weekly_samples = []

    for week in weeks:
        # Generate uncorrelated noise for the current week
        uncorrelated_noise = np.random.normal(0, 1, size=(num_players, num_scenarios))

        # Perform matrix multiplication for this week
        correlated_standard_normal = cholesky_factor @ uncorrelated_noise

        # Apply the CDF to get correlated uniform samples
        correlated_uniform_samples = norm.cdf(correlated_standard_normal)

        # Convert to a DataFrame and add the week number
        weekly_samples_df = pd.DataFrame(correlated_uniform_samples.T, columns=player_ids)
        weekly_samples_df.columns = pd.MultiIndex.from_product([weekly_samples_df.columns, [week]])

        all_weekly_samples.append(weekly_samples_df)

    # Concatenate all weekly DataFrames along the columns axis
    final_samples = pd.concat(all_weekly_samples, axis=1)

    # Sort the MultiIndex for better organization
    final_samples.columns.names = ['player_id', 'week_num']

    return final_samples
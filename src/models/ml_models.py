import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tqdm
from lightgbm import LGBMClassifier
from ngboost import NGBRegressor, distns
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

from src.config import (LEAGUE_NAME, SEASON, N_SIMS,
                        CUMULATIVE_IMPORTANCE_INJURY, CUMULATIVE_IMPORTANCE_POINTS)
from src.constants import PRO_TEAM_MAP
from src.data.historical_adp import download_adp_data
from src.data.nfl_draft_data import get_nfl_draft_data
from src.data.utilities import get_league, get_schedule, get_sleeper, get_old_leagues, get_old_schedule, game_by_id, \
    get_extra_players, get_bye_weeks, generate_correlated_samples_weekly
from src.models.feature_selection import apply_feature_selection_to_pipeline
from src.models.player import Player
from src.paths import root_dir
from src.tools import cache_wrapper

#### MODELING & PROJECTION FUNCTIONS ####

# Global variables to store feature selectors and used features
injury_feature_selector = None
points_feature_selector = None
injury_features_used = None
points_features_used = None


def injury_pipeline(categorical_cols, numerical_cols, extra_numerical_cols, verbose=0):
    """A modular pipeline builder for the injury model."""
    onehot = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='Unknown', add_indicator=True)),
        ('enc', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])

    stats_cols = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0)),
        ('sc', StandardScaler())
    ])

    trans = ColumnTransformer([
        ('cat1', onehot, categorical_cols),
        ('num', numerical, numerical_cols),
        ('stats', stats_cols, extra_numerical_cols),
    ], verbose_feature_names_out=False)

    pipeline = Pipeline([
        ('prep', trans),
        ('clf', LGBMClassifier(verbose=verbose, random_state=42))
    ])

    return pipeline


def get_injury_model(df):
    """
    Trains and returns an injury prediction model with mandatory feature selection
    on rolling/efficiency stats.
    """
    global injury_feature_selector
    global injury_features_used

    param_file = root_dir / Path('data/injury_model_best_params.json')

    # --- 1. Define Feature Groups ---
    BASE_CATEGORICAL = ['position', 'team', 'opponent', 'college', 'is_rookie', 'is_defense', 'college_conference']
    BASE_NUMERICAL = ['height', 'weight', 'age_squared', 'experience', 'draft_pick', 'draft_grade',
                      'age_exp_interaction', 'week_num', 'season', 'adp',
                      'rushing_incumbent_value', 'receiving_incumbent_value', 'passing_incumbent_value']
    BASE_FEATURES = BASE_CATEGORICAL + BASE_NUMERICAL

    # Identify candidate features for selection (rolling avgs, etc.)
    extra_numeric_candidates = [
        col for col in df.columns if re.search(r'_(?:\d+game|lag\d+)(?:_std|_momentum)?$', col) or
                                     re.search(
                                         r'(?:catch_rate|yards_per_carry|yards_per_attempt|yards_per_reception|'
                                         r'completion_rate|target_share|consistency|upside|floor)_\d+game$', col
                                     )
    ]

    y = df['gamesPlayed']

    # --- 2. Hyperparameter Tuning & Feature Selection (if needed) ---
    if not param_file.exists():
        print("Hyperparameter file not found. Running Optuna study for injury model...")
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

        # Apply feature selection ONLY to the candidate columns
        print(f"Selecting best features from {len(extra_numeric_candidates)} candidates...")
        X_candidates = train_df[extra_numeric_candidates]
        y_temp = train_df['gamesPlayed']

        X_selected, feature_selector = apply_feature_selection_to_pipeline(
            X_candidates, y_temp, task_type='classification',
            cumulative_importance_threshold=CUMULATIVE_IMPORTANCE_INJURY
        )
        selected_extra_features = list(X_selected.columns)
        print(f"Selected {len(selected_extra_features)} features.")

        # The final feature set is the base features plus the selected extra features
        injury_features_used = BASE_FEATURES + selected_extra_features
        injury_feature_selector = feature_selector

        def objective(trial):
            param = {
                'verbose': -1, 'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 2.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 2.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            }

            pipeline = injury_pipeline(BASE_CATEGORICAL, BASE_NUMERICAL, selected_extra_features)
            pipeline.named_steps['clf'].set_params(**param)

            # Use the combined feature set for training
            train_X = train_df[injury_features_used]
            train_y = train_df['gamesPlayed']
            test_X = test_df[injury_features_used]
            test_y = test_df['gamesPlayed']

            pipeline.fit(train_X, train_y)
            preds = pipeline.predict_proba(test_X)
            nll = log_loss(test_y, preds)
            return nll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        best_params = study.best_trial.params

        print(f"Optuna study complete. Best params found: {best_params}")
        best_params['used_features'] = injury_features_used
        with param_file.open('w') as f:
            json.dump(best_params, f, indent=4)

        final_extra_numeric = selected_extra_features

    # --- 3. Load Existing Parameters ---
    else:
        print("Loading existing hyperparameters and features for injury model.")
        with param_file.open('r') as f:
            best_params = json.load(f)

        injury_features_used = best_params.pop('used_features')
        final_extra_numeric = list(set(injury_features_used) - set(BASE_FEATURES))

    # --- 4. Train Final Model ---
    print("Training final injury model with best parameters and selected features...")
    final_pipeline = injury_pipeline(BASE_CATEGORICAL, BASE_NUMERICAL, final_extra_numeric, verbose=-1)
    final_pipeline.named_steps['clf'].set_params(**best_params)

    # Retrain on the full dataset using only the selected features
    X_final = df[injury_features_used]
    final_pipeline.fit(X_final, y)

    return final_pipeline


def points_pipeline(params, categorical_cols, numerical_cols, extra_numerical_cols):
    """A modular pipeline builder that accepts feature lists as arguments."""
    oh = Pipeline([('imp', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                   ('enc', TargetEncoder())])
    base_num = Pipeline([('imp', SimpleImputer(add_indicator=True)),
                         ('sc', StandardScaler())])
    rolling_num = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value=0)),
                            ('sc', StandardScaler())])

    trans = ColumnTransformer([
        ('oh', oh, categorical_cols),
        ('num', base_num, numerical_cols),
        ('rolling_num', rolling_num, extra_numerical_cols),
    ], remainder='drop', verbose_feature_names_out=False)

    dist = getattr(distns, params['dist'])

    base_learner = DecisionTreeRegressor(
        max_depth=params['base_depth'],
        min_samples_split=params.get('min_samples_split', 2),
        min_samples_leaf=params.get('min_samples_leaf', 1)
    )

    pipeline = Pipeline([
        ('prep', trans),
        ('clf', NGBRegressor(Dist=dist,
                             Base=base_learner,
                             n_estimators=params['n_estimators'],
                             learning_rate=params['learning_rate'],
                             minibatch_frac=params['minibatch_frac'],
                             early_stopping_rounds=params['early_stopping_rounds'],
                             random_state=42,
                             validation_fraction=params['validation_fraction']))
    ])

    return pipeline


def add_usage_incumbent_value(df: pd.DataFrame, future: bool) -> pd.DataFrame:
    """
    Calculates incumbent opportunity value for rushing, receiving, and passing.

    This feature represents the total 10-game rolling usage for a team
    minus the individual player's usage, indicating the opportunity available
    from other players on the team.

    Args:
        df: The input DataFrame containing player game data.
        future: If True, uses future data for calculation (for projections).
                If False, uses only past data (for training).

    Returns:
        The DataFrame with three new incumbent value columns.
    """
    # Define the usage columns to base the feature on
    usage_cols = [
        'rushingAttempts_3game',
        'receivingTargets_3game',
        'passingAttempts_3game'
    ]

    # Fill NaNs with 0 for calculation; no data means no usage.
    df[usage_cols] = df[usage_cols].fillna(0)

    # Calculate the total usage for each team in each season
    team_totals = df[df['gamesPlayed'] > 0].groupby(['season', 'team', 'week_num'])[usage_cols].sum()
    team_totals.rename(columns={col: f'team_total_{col}' for col in usage_cols}, inplace=True)

    # Merge team totals back into the main DataFrame
    df = pd.merge(df, team_totals, on=['season', 'team', 'week_num'], how='left')

    # Calculate incumbent value by subtracting the player's usage from the team total
    df['rushing_incumbent_value'] = (df['team_total_rushingAttempts_3game'] - df['rushingAttempts_3game']).fillna(0)
    df['receiving_incumbent_value'] = (df['team_total_receivingTargets_3game'] - df['receivingTargets_3game']).fillna(0)
    df['passing_incumbent_value'] = (df['team_total_passingAttempts_3game'] - df['passingAttempts_3game']).fillna(0)

    # Clean up by dropping the intermediate total columns
    df.drop(columns=[f'team_total_{col}' for col in usage_cols], inplace=True)

    return df


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


def generate_correlated_samples(corr_matrix, num_scenarios):
    """
    Generates correlated random samples with a robust check for PSD matrices.
    """
    try:
        # First, we try to decompose the original matrix directly.
        cholesky_factor = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # If it fails, we know it's not positive definite and we must fix it.
        print("Warning: Correlation matrix is not positive definite. Adjusting with eigenvalue decomposition.")
        corrected_matrix = make_positive_definite(corr_matrix)
        cholesky_factor = np.linalg.cholesky(corrected_matrix)  # This should now succeed

    # The rest of the function is the same
    player_ids = corr_matrix.columns
    num_players = len(player_ids)
    uncorrelated_noise = np.random.normal(0, 1, size=(num_players, num_scenarios))
    correlated_standard_normal = cholesky_factor @ uncorrelated_noise
    correlated_uniform_samples = norm.cdf(correlated_standard_normal)

    return pd.DataFrame(correlated_uniform_samples.T, columns=player_ids)


def add_features(df, future=False):
    df['is_rookie'] = df['experience'] == 0
    df['is_defense'] = df['position'] == 'D/ST'
    df['age_squared'] = df['age'] ** 2

    # Interaction between age and experience
    df['age_exp_interaction'] = df['age'] * df['experience']

    df = add_usage_incumbent_value(df, future=future)

    return df


def get_points_model(df, point_values):
    """
    Trains and returns a fantasy points prediction model with mandatory feature selection
    on rolling/efficiency stats.
    """
    global points_feature_selector
    global points_features_used

    param_file = root_dir / Path('data/points_model_best_params.json')
    common_cols = set(df.columns) & set(point_values.keys())
    coefs = [point_values[k] for k in common_cols]
    df['applied_total'] = (df[list(common_cols)].fillna(0) @ coefs)

    # --- 1. Define Feature Groups ---
    BASE_CATEGORICAL = ['position', 'team', 'opponent', 'college', 'is_rookie', 'is_defense',
                        'college_conference']
    BASE_NUMERICAL = ['height', 'weight', 'experience', 'draft_pick', 'draft_grade', 'adp',
                      'pre_draft_ranking', 'age_squared', 'gamesPlayed', 'week_num', 'season',
                      'rushing_incumbent_value', 'receiving_incumbent_value', 'passing_incumbent_value']
    BASE_FEATURES = BASE_CATEGORICAL + BASE_NUMERICAL

    # Identify candidate features for selection (rolling avgs, etc.)
    extra_numeric_candidates = [
        col for col in df.columns if re.search(r'_\d+game(?:_std|_momentum)?$', col) or
                                     re.search(
                                         r'(?:catch_rate|yards_per_carry|yards_per_attempt|yards_per_reception|'
                                         r'completion_rate|target_share|consistency|upside|floor)_\d+game$', col
                                     )
    ]

    # --- 2. Hyperparameter Tuning & Feature Selection (if needed) ---
    if not param_file.exists():
        print("Hyperparameter file not found. Running Optuna study for points model...")
        complete_cases = df.dropna(subset=['applied_total'])
        train_df, test_df = train_test_split(complete_cases, shuffle=False, test_size=0.2)

        # Apply feature selection ONLY to the candidate columns
        print(f"Selecting best features from {len(extra_numeric_candidates)} candidates...")
        X_candidates = train_df[extra_numeric_candidates]
        y_temp = train_df['applied_total']

        X_selected, feature_selector = apply_feature_selection_to_pipeline(
            X_candidates, y_temp, task_type='regression',
            cumulative_importance_threshold=CUMULATIVE_IMPORTANCE_POINTS,
        )
        selected_extra_features = list(X_selected.columns)
        print(f"Selected {len(selected_extra_features)} features.")

        # The final feature set is the base features plus the selected extra features
        points_features_used = BASE_FEATURES + selected_extra_features
        points_feature_selector = feature_selector

        # Define the objective function for Optuna
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
                'minibatch_frac': trial.suggest_float('minibatch_frac', 0.25, 1),
                'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.5),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 0, 200),
                'dist': trial.suggest_categorical('dist', ['Gamma', 'LogNormal', 'Normal']),
                'base_depth': trial.suggest_int('base_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            }
            # The pipeline uses the combined feature set
            pipeline = points_pipeline(param, BASE_CATEGORICAL, BASE_NUMERICAL, selected_extra_features)

            # Use only the columns needed for this training run
            train_X = train_df[points_features_used]
            train_y = train_df['applied_total']
            test_X = test_df[points_features_used]
            test_y = test_df['applied_total']

            pipeline.fit(train_X, train_y.clip(1e-6))

            ngb_model = pipeline.named_steps['clf']

            preprocessing = pipeline.named_steps['prep']
            test_X_scaled = preprocessing.transform(test_X)

            pred_dist = ngb_model.pred_dist(test_X_scaled)
            nll = -pred_dist.logpdf(test_y.clip(1e-6)).mean()

            return nll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        best_params = study.best_trial.params

        # Save params and the full list of features used
        best_params['used_features'] = points_features_used
        with open(param_file, 'w') as f:
            json.dump(best_params, f, indent=4)

        final_extra_numeric = selected_extra_features

    # --- 3. Load Existing Parameters ---
    else:
        print("Loading existing hyperparameters and features for points model.")
        with open(param_file, 'r') as f:
            best_params = json.load(f)

        points_features_used = best_params.pop('used_features')
        # Determine which of the used features were the 'extra' ones
        final_extra_numeric = list(set(points_features_used) - set(BASE_FEATURES))

    # --- 4. Train Final Model ---
    print("Training final points model with best parameters and selected features...")
    model = points_pipeline(best_params, BASE_CATEGORICAL, BASE_NUMERICAL, final_extra_numeric)

    # Use the final combined feature set for training
    train_data = df.dropna(subset=['applied_total'])
    X_final = train_data[points_features_used]
    y_final = train_data['applied_total'].clip(1e-6)
    model.fit(X_final, y_final)

    global PLAYER_POINTS_DIST
    PLAYER_POINTS_DIST = best_params['dist']

    # --- 5. Evaluate Model ---
    r2_full = r2_score(df['applied_total'].dropna(),
                       model.predict(df[points_features_used]))
    r2_subset = r2_score(y_final, model.predict(X_final))
    print(f"R2 score (on non-NA targets): Full={r2_full:.4f}, Subset={r2_subset:.4f}")

    total_features = len(BASE_FEATURES) + len(extra_numeric_candidates)
    selected_count = len(points_features_used)
    print(f"Features selected: {selected_count} out of {total_features} total possible features.")
    print(f"Feature reduction: {((total_features - selected_count) / total_features * 100):.1f}%")

    return model


def project_player_dists(df, point_model):
    """Generates future point distributions for players."""
    # Use the globally stored feature list to slice the data
    df_to_predict = df[list(set(points_features_used) & set(df.columns))]

    X = point_model[0].transform(df_to_predict)

    projections = point_model[-1].pred_dist(X)
    new_params = {}
    if PLAYER_POINTS_DIST.lower() == 'gamma':
        new_params['scale'] = 1 / projections.params['beta']
        new_params['a'] = projections.params['alpha']
    elif PLAYER_POINTS_DIST.lower() == 'lognormal':
        pass
    elif PLAYER_POINTS_DIST.lower() == 'normal':
        new_params = projections.params
    else:
        raise ValueError(f"Unrecognized distribution: {PLAYER_POINTS_DIST}")
    # Add other distributions as needed
    df['params'] = pd.DataFrame(new_params, index=df.index).to_dict('records')
    return df.groupby('player_id')


def project_injury_rates(df, injury_model):
    """Generates future injury probabilities for players."""
    # Use the globally stored feature list to slice the data
    df_to_predict = df[list(set(injury_features_used) & set(df.columns))]

    df['injury_rate'] = injury_model.predict_proba(df_to_predict)[:, 1]
    return df.groupby('player_id')


def get_replacement_player_levels(players, league):
    """
    Calculates the replacement player scoring level for each position,
    including a separate calculation for flex spots.
    """
    positional_points = defaultdict(list)
    for player in players.values():
        if player.ave_projected_ppg is None: continue
        for position in player.positions:
            if position in ['BE', 'IR']: continue
            positional_points[position].append(player.ave_projected_ppg)

    vorp_levels = {}
    flex_candidates = []
    FLEX_ELIGIBLE_POSITIONS = ['RB', 'WR', 'TE']
    FLEX_KEY = 'RB/WR/TE'  # Standard Flex key, adjust if yours is different

    # Step 1: Calculate VORP for non-flex, primary positions
    for position, ppgs in positional_points.items():
        if position not in league.roster_slots or position == FLEX_KEY: continue

        top_ppgs = sorted(ppgs, reverse=True)
        expected_num_used = league.roster_slots[position] * league.settings.team_count

        if expected_num_used < len(top_ppgs):
            replacement_level = top_ppgs[expected_num_used]
            vorp_levels[position] = replacement_level
            if position in FLEX_ELIGIBLE_POSITIONS:
                flex_candidates.extend(top_ppgs[expected_num_used:])
        else:
            vorp_levels[position] = top_ppgs[-1] if top_ppgs else 0.0

    # Step 2: Calculate VORP for the Flex position
    if FLEX_KEY in league.roster_slots and flex_candidates:
        num_flex_starters = league.roster_slots[FLEX_KEY] * league.settings.team_count
        sorted_flex_candidates = sorted(flex_candidates, reverse=True)
        if num_flex_starters < len(sorted_flex_candidates):
            vorp_levels[FLEX_KEY] = sorted_flex_candidates[num_flex_starters]
        else:
            vorp_levels[FLEX_KEY] = sorted_flex_candidates[-1] if sorted_flex_candidates else 0.0

    return vorp_levels


def construct_schedule_matrix(schedules, league):
    """
    Constructs a schedule matrix for the each team in the league by season. Creates a DataFrame
    where each row is a team and each column indicates if the team played against that team in that season.
    """
    season_mats = defaultdict(dict)

    for season, schedule in schedules.items():
        # Process all matchups in one pass
        matchups = [(PRO_TEAM_MAP.get(matchup['awayProTeamId']), PRO_TEAM_MAP.get(matchup['homeProTeamId']), season)
                    for team_schedule in schedule.values()
                    for matchups in team_schedule.values()
                    for matchup in matchups
                    if 'awayProTeamId' in matchup and 'homeProTeamId' in matchup]

        # Update matrices for all matchups
        for away_id, home_id, game_season in matchups:
            season_mats[(game_season, away_id)][home_id] = 1
            season_mats[(game_season, home_id)][away_id] = 1

    season_dfs = pd.DataFrame(season_mats).T.fillna(0)

    return season_dfs


#### MAIN EXECUTION LOGIC ####
@cache_wrapper(60 * 60 * 24)  # Cache for 24 hours
def get_all_players(league, league_name):
    """Main function to orchestrate data fetching, modeling, and player object creation."""
    # Fetch all necessary data sources
    schedules = get_schedule(league)
    byes = get_bye_weeks(schedules)
    sleeper_players = get_sleeper()
    draft_data = get_nfl_draft_data(range(2005, league.year + 1))

    year_range = range(max(league.year - 3, 2022), league.year + 1)
    old_league_obs = get_old_leagues(year_range[:-1], league_name)
    old_schedules = {year: get_old_schedule(year, old_league_obs[year]) for year in year_range[:-1]}
    old_schedules[league.year] = schedules
    schedules_by_id = game_by_id(old_schedules)
    adps = download_adp_data(year_range, update=False)

    # Get all relevant players and their IDs
    players = league.all_agents(size=len(league.teams) * sum(league.roster_slots.values()) * 1.5)
    league.draft_dict = {player.playerId: player.round_num for player in league.draft}
    ids = [player.playerId for player in players]
    game_logs, extra_players = get_extra_players(ids, year_range, league, old_league_obs)

    rosters = {}
    assigned_slots = {}
    for team in league.teams:
        for player in team.roster:
            rosters[player.playerId] = team.team_abbrev
            assigned_slots[player.playerId] = player.lineupSlot

    # Create Player objects and collect data for modeling
    games_played, future_games, parsed_players = [], [], {}
    print("Parsing player data...")
    for player in tqdm.tqdm(players):
        if player.proTeam == 'None': continue
        player_ob = Player(player, league, extra_players, schedules, schedules_by_id, sleeper_players, draft_data,
                           game_logs, league.year, byes)
        player_ob.manager = rosters.get(player.playerId)
        player_ob.assigned_slot = assigned_slots.get(player.playerId)
        player_ob.add_adp(adps)

        parsed_players[player.playerId] = player_ob
        games_played.append(player_ob.games_played_mat(league.settings.category_points, league.currentMatchupPeriod))
        future_games += player_ob.get_future_matchups()

    # Train models
    print("Training models...")
    games_played_df = pd.concat(games_played).sort_values(by=['season', 'week_num']).dropna(subset=['player_id'])
    games_played_df = add_features(games_played_df)
    future_games_df = pd.DataFrame(future_games).dropna(subset=['player_id'])
    future_games_df = add_features(future_games_df, future=True)

    pos_corr_mat = get_correlation_mat(games_played_df)
    player_corr_mat = construct_corr_matrix(parsed_players, pos_corr_mat)

    weeks = list(map(str, range(1, league.finalScoringPeriod + 2)))
    samples = generate_correlated_samples_weekly(player_corr_mat, N_SIMS, weeks).T

    injury_model = get_injury_model(games_played_df)
    points_model = get_points_model(games_played_df, league.settings.category_points)

    injury_probs = project_injury_rates(future_games_df, injury_model)
    for player_id, data in injury_probs:
        if player_id in parsed_players:
            parsed_players[player_id].set_injury_probs(data)

    # Make projections and set them on player objects
    print("Generating player projections...")
    player_projections = project_player_dists(future_games_df, points_model)
    for player_id, dists in player_projections:
        if player_id in parsed_players:
            parsed_players[player_id].set_projections(dists, PLAYER_POINTS_DIST, samples.loc[player_id])

    # Calculate VORP
    print("Calculating VORP...")
    replacement_levels = get_replacement_player_levels(parsed_players, league)
    for player in parsed_players.values():
        player.set_vorp(replacement_levels)

    players_list = sorted(parsed_players.values(), key=lambda x: x.vorp, reverse=True)
    return players_list


if __name__ == '__main__':
    league = get_league(LEAGUE_NAME, SEASON, force_update=True)
    all_players = get_all_players(league, LEAGUE_NAME, force_update=True)
    print("\n--- Top 20 Players by VORP ---")
    for i, player in enumerate(all_players[:30]):
        print(f"{i + 1:2d}. {player.name:<25} ({player.pro_position}) - VORP: {player.vorp:.2f}")
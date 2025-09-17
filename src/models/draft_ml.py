import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tqdm
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor, distns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

from src.config import LEAGUE_NAME, SEASON
from src.constants import PRO_TEAM_MAP, GAMES_IN_SEASON
from src.data.historical_adp import download_adp_data
from src.data.nfl_draft_data import get_nfl_draft_data
from src.data.utilities import get_league, get_schedule, get_sleeper, get_old_leagues, get_old_schedule, game_by_id, \
    get_extra_players, get_bye_weeks
from src.models.player import DraftablePlayer
from src.paths import root_dir
from src.tools import cache_wrapper


#### MODELING & PROJECTION FUNCTIONS ####

def injury_pipeline(extra_numeric=None, extra_categorical=None,):
    cat = ['position', 'team', 'college', 'is_rookie', 'is_defense', 'college_conference']
    num = (['height', 'weight', 'age_squared', 'experience', 'draft_pick', 'draft_grade', 'adp', 'age_exp_interaction',
           'D/ST_incumbent', 'K_incumbent', 'QB_incumbent', 'RB_incumbent', 'WR_incumbent', 'TE_incumbent'])
    onehot = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='Unknown', add_indicator=True)),
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, min_frequency=2))
    ])

    numerical = Pipeline([
        ('imp', KNNImputer(add_indicator=True)),
        ('sc', StandardScaler())
    ])

    stats_cols = Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value=0)),
        ('sc', StandardScaler())
    ])

    teams_feats = Pipeline([('target', TargetEncoder())])

    trans = ColumnTransformer([
        ('cat1', onehot, ['player_id'] + cat),
        ('num', numerical, num),
        ('stats', stats_cols, extra_numeric or []),
        ('opponents', teams_feats, list(PRO_TEAM_MAP.values()))
    ], verbose_feature_names_out=False)

    pipeline = Pipeline([
        ('prep', trans),
        ('clf', LGBMRegressor(objective='poisson', verbose=-1, random_state=42))
    ])

    return pipeline


def get_injury_model(df):
    """
    Trains and returns an injury prediction model.

    It loads hyperparameters from a file if available, otherwise it runs
    an Optuna study to find and save them.
    """
    param_file = root_dir / Path('data/injury_model_best_params.json')
    extra_numeric = [col for col in df.columns if re.search(r'_lag\d+', col)]
    X = df.drop(columns=['gamesPlayed'])
    y = df['gamesPlayed']

    if not param_file.exists():
        print("Hyperparameter file not found. Running Optuna study for injury model...")

        def objective(trial):
            param = {
                'objective': 'poisson',
                'metric': 'rmse',
                'verbose': -1,
                'random_state': 42,
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

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)
            pipeline = injury_pipeline(extra_numeric)
            pipeline.named_steps['clf'].set_params(**param)
            pipeline.fit(train_X, train_y)
            preds = pipeline.predict(test_X)
            rmse = np.sqrt(mean_squared_error(test_y, preds))
            return rmse

        # Create and run the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)  # Adjust n_trials as needed
        best_params = study.best_trial.params

        print(f"Optuna study complete. Best params found: {best_params}")
        # Save the best parameters to a file
        with param_file.open('w') as f:
            json.dump(best_params, f, indent=4)
    else:
        print("Loading existing hyperparameters for injury model.")
        with param_file.open('r') as f:
            best_params = json.load(f)

    # Create the final pipeline and set the best (found or loaded) parameters
    final_pipeline = injury_pipeline(extra_numeric)
    final_pipeline.named_steps['clf'].set_params(**best_params)

    # Retrain on the full dataset
    print("Training final injury model with best parameters...")
    final_pipeline.fit(X, y)

    return final_pipeline


def points_pipeline(params, extra_numerical=None, extra_categorical=None):
    BASE_NUMERICAL = (['height', 'weight', 'experience', 'draft_pick', 'draft_grade',
                      'pre_draft_ranking', 'age_squared', 'adp', 'gamesPlayed', 'age_exp_interaction',
                      'D/ST_incumbent', 'K_incumbent', 'QB_incumbent', 'RB_incumbent', 'WR_incumbent', 'TE_incumbent'])

    BASE_CATEGORICAL = ['player_id', 'position', 'team', 'college', 'is_rookie', 'is_defense', 'college_conference']

    oh = Pipeline([('imp', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                   ('enc', OneHotEncoder(handle_unknown='ignore'))])
    base_num = Pipeline([('imp', KNNImputer(add_indicator=True)),
                    ('sc', StandardScaler())])
    rolling_num = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value=0)),
                            ('sc', StandardScaler())])
    teams_feats = Pipeline([('target', TargetEncoder())])

    trans = ColumnTransformer([
        ('oh', oh, BASE_CATEGORICAL),
        ('num', base_num, BASE_NUMERICAL),
        ('rolling_num', rolling_num, extra_numerical or []),
        ('opponents', teams_feats, list(PRO_TEAM_MAP.values()))
    ], remainder='drop', verbose_feature_names_out=False)

    dist = getattr(distns, params['dist'])

    # Define the base learner with tunable regularization parameters
    base_learner = DecisionTreeRegressor(
        max_depth=params['base_depth'],
        min_samples_split=params.get('min_samples_split', 2),  # Use .get for backward compatibility
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


def add_incumbant_value(df):
    """Adds the value of the value at each position for each team."""
    df['adp_value'] = (df['adp'].max() - df['adp']) / df['adp'].max()

    incumbant_values = df.groupby(['season', 'team', 'position'])['adp_value'].sum().unstack()

    df = df.merge(incumbant_values.reset_index(), on=['season', 'team'], how='left')
    mask = pd.get_dummies(df['position']).reindex(columns=incumbant_values.columns, fill_value=0)

    df[incumbant_values.columns] = df[incumbant_values.columns] - mask.multiply(df['adp_value'], axis=0)

    # Safer and more explicit way to rename
    rename_dict = {col: f'{col}_incumbent' for col in incumbant_values.columns}
    df = df.rename(columns=rename_dict)

    return df


def add_features(df, opponent_mats):
    df['is_rookie'] = df['experience'] == 0
    df['is_defense'] = df['position'] == 'D/ST'
    df['age_squared'] = df['age'] ** 2

    # Interaction between age and experience
    df['age_exp_interaction'] = df['age'] * df['experience']

    df = add_incumbant_value(df)

    df = df.merge(opponent_mats, left_on=['season', 'team'], right_index=True, how='left')

    return df


def get_points_model(df, point_values):
    """Trains and returns a fantasy points prediction model."""
    param_file = root_dir / Path('data/season_points_model_best_params.json')
    common_cols = set(df.columns) & set(point_values.keys())
    coefs = [point_values[k] for k in common_cols]
    df['applied_total'] = (df[list(common_cols)].fillna(0) @ coefs)

    extra_numeric = [col for col in df.columns if re.search(r'_lag\d+', col)]

    def objective(trial, train_X, train_y, test_X, test_y):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2, log=True),
            'minibatch_frac': trial.suggest_float('minibatch_frac', 0.25, 1),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.5), # Adjusted range
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 0, 200),
            'dist': trial.suggest_categorical('dist', ['Gamma', 'LogNormal', 'Normal', 'HalfNormal']),
            'base_depth': trial.suggest_int('base_depth', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
        pipeline = points_pipeline(param, extra_numeric)
        pipeline.fit(train_X, train_y.clip(1e-6))

        ngb_model = pipeline.named_steps['clf']
        preprocessing = pipeline.named_steps['prep']

        test_X_scaled = preprocessing.transform(test_X)
        pred_dist = ngb_model.pred_dist(test_X_scaled)

        nll = -pred_dist.logpdf(test_y).mean()

        return nll

    if not param_file.exists():
        # Split data once before the optimization loop
        train_df, test_df = train_test_split(df, shuffle=False, test_size=0.2)
        train_X, train_y = train_df, train_df['applied_total']
        test_X, test_y = test_df, test_df['applied_total']

        # Ensure training set is not empty
        if train_X.empty:
            raise ValueError("Training data is empty after splitting. Check the size of 'games_played_df'.")

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, train_X, train_y, test_X, test_y), n_trials=50)
        best_params = study.best_trial.params
        json.dump(best_params, open(param_file, 'w'), indent=4)
    else:
        best_params = json.load(open(param_file, 'r'))

    model = points_pipeline(best_params, extra_numeric)
    subset = df.dropna(subset=['applied_total'])
    model.fit(subset, subset['applied_total'].clip(1e-6))

    global PLAYER_POINTS_DIST
    PLAYER_POINTS_DIST = best_params['dist']

    r2_score(df['applied_total'], model.predict(df))
    print(
        f"R2 score: {r2_score(df['applied_total'], model.predict(df)):.4f} (best: {r2_score(subset['applied_total'], model.predict(subset)):.4f})"
    )
    print(
        f"Number of features: {len(df.columns) - len(extra_numeric) - 1} (best: {len(subset.columns) - len(extra_numeric) - 1})"
    )

    return model


def project_player_dists(df, point_model):
    """Generates future point distributions for players."""
    X = point_model[0].transform(df)
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
    df['injury_rate'] = injury_model.predict(df)
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
@cache_wrapper(60 * 60 * 24) # Cache for 24 hours
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
    adp_data = download_adp_data(year_range)
    opponent_mats = construct_schedule_matrix(old_schedules, league)

    # Get all relevant players and their IDs
    players = league.all_agents(size=len(league.teams) * sum(league.roster_slots.values()) * 1.2)
    league.draft_dict = {player.playerId: player.round_num for player in league.draft}
    ids = [player.playerId for player in players]
    game_logs, extra_players = get_extra_players(ids, year_range, league, old_league_obs)

    # Create Player objects and collect data for modeling
    games_played, future_games, parsed_players = [], [], {}
    print("Parsing player data...")
    for player in tqdm.tqdm(players):
        if player.proTeam == 'None': continue
        player_ob = DraftablePlayer(player, league, extra_players, schedules, schedules_by_id, sleeper_players, draft_data,
                                    game_logs, league.year, byes)
        player_ob.add_adp(adp_data)
        parsed_players[player.playerId] = player_ob
        games_played.append(player_ob.games_played_mat(league.settings.category_points))
        future_games.append(player_ob.get_future_matchups())

    # Train models
    print("Training models...")
    games_played_df = pd.concat(games_played).sort_values(by=['season'])
    games_played_df = add_features(games_played_df, opponent_mats)
    future_games_df = pd.DataFrame(future_games)
    future_games_df = add_features(future_games_df, opponent_mats)
    future_games_df['gamesPlayed'] = GAMES_IN_SEASON

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
            parsed_players[player_id].set_projections(dists, PLAYER_POINTS_DIST)

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
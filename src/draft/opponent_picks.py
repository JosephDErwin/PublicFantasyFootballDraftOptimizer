from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic.fields import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.config import SEASON, LEAGUE_NAME
from src.data.historical_adp import download_adp_data
from src.data.utilities import get_league
from src.paths import root_dir


def find_adp(player_name, adp_df):
    """Helper to find ADP for a player."""
    # ... (this can be moved outside the class to be reused)
    adp_row = adp_df[adp_df['Player Name'] == player_name]
    if adp_row.shape[0] == 1:
        return adp_row.to_dict(orient='records')[0]
    for part in player_name.split(' '):
        subset = adp_df[adp_df['Player Name'].str.contains(part, na=False)]
        if subset.shape[0] == 1:
            return subset.to_dict(orient='records')[0]
    return {}


def create_all_players_df(league, adp_df):
    """Creates a feature DataFrame for all draftable players in a given year."""
    players_map = {p.playerId: p for p in league.all_agents(size=500) if p.playerId in league.player_map}
    player_rows = []
    for player_id, player_info in players_map.items():
        adp_row = find_adp(player_info.name, adp_df)
        entry = {
            'player_id': player_id,
            'player_name': player_info.name,
            'adp': adp_row.get('AVG'),
            'position': player_info.position,
            'pro_team': player_info.proTeam,
        }
        player_rows.append(entry)
    return pd.DataFrame(player_rows)


class PlayerAvailabilityModel:
    """
    Models the probability that a player will be available at a given pick
    using a Random Survival Forest.

    This model is trained on historical draft data to learn how long each player
    "survives" on the draft board.
    """

    def __init__(self, n_estimators=100, min_samples_leaf=10, random_state=42):
        self.model_file = root_dir / Path('data/player_availability_model.joblib')

        # FIX: Reverted to OneHotEncoder and removed player_id as a feature.
        numerical_features = ['adp']
        categorical_features = ['position', 'pro_team']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('impute', SimpleImputer(strategy='median', add_indicator=True)),
                    ('scale', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                    # FIX: Use OneHotEncoder for non-ordinal features
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ], remainder='drop'
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', GradientBoostingSurvivalAnalysis(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            ))
        ])

    def _prepare_training_data(self, leagues, adps):
        """
        Transforms historical draft data into features (X) and a structured
        target array (y) suitable for survival analysis.
        """
        training_rows = []
        for league in leagues:
            print(f"Processing league: {league.year}...")

            draft_results = {p.playerId: i + 1 for i, p in enumerate(league.draft)}
            total_picks = len(league.draft)

            # Use ADP list as the universe of draftable players for that year
            adp_df = adps[league.year]
            players_map = {p.playerId: p for p in league.all_agents(size=500) if p.playerId in league.player_map}

            those_drafted = [p.playerId for p in league.draft]
            not_drafted = set(players_map.keys()) - set(those_drafted)

            augmented_draft = those_drafted + list(not_drafted)

            for pick_id, player_id in enumerate(augmented_draft):
                player = players_map.get(player_id)

                if not player: continue

                player_name = player.name

                adp_row = find_adp(player_name, adp_df)

                player_info = players_map.get(player_id)
                if not player_info: continue

                # Determine the event and time for the survival model
                was_drafted = player_id in draft_results
                pick_number = pick_id + 1 if was_drafted else total_picks

                entry = {
                    'player_id': player_id,
                    'adp': adp_row.get('AVG'),
                    'position': player_info.position,
                    'was_drafted': was_drafted,
                    'pick_number': pick_number,
                    'pro_team': player_info.proTeam,
                }

                training_rows.append(entry)

        df = pd.DataFrame(training_rows)

        # Create features X and target y
        X = df

        # The target must be a structured array with (event_indicator, event_time)
        y = np.array(
            list(zip(df['was_drafted'], df['pick_number'])),
            dtype=[('was_drafted', bool), ('pick_number', int)]
        )

        return X, y

    def train(self, leagues, adps):
        """Trains the Random Survival Forest model on the full dataset."""
        print("Preparing data for survival model...")
        X, y = self._prepare_training_data(leagues, adps)

        print(f"Training Random Survival Forest on {len(X)} samples...")
        self.pipeline.fit(X, y)
        print("Training complete.")

    def predict_survival_functions(self, features_df):
        """
        Predicts the BASELINE survival functions for a given set of players.
        This is the "in a vacuum" prediction based on static features.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be trained before it can be used for prediction.")
        return self.pipeline.predict_survival_function(features_df, return_array=False)

    def save(self):
        """Saves the trained pipeline to a file."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model.")

        self.model_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving model pipeline to {self.model_file}...")
        joblib.dump(self.pipeline, self.model_file)
        print("Save complete.")

    @staticmethod
    def load(filepath):
        """Loads a model pipeline from a file and reconstructs the class."""
        print(f"Loading model pipeline from {filepath}...")
        pipeline = joblib.load(filepath)

        model_instance = PlayerAvailabilityModel()
        model_instance.pipeline = pipeline

        print("Model loaded and reconstructed successfully.")
        return model_instance


# --- THE CORRECTED MAIN BLOCK ---
if __name__ == '__main__':
    # This block now serves as the main training script.
    print("--- Loading Historical Data ---")
    year_range = range(2022, SEASON)
    league_obs = [get_league(LEAGUE_NAME, season) for season in year_range]
    adps = download_adp_data(year_range)

    # 1. Instantiate the model
    availability_model = PlayerAvailabilityModel()

    # 2. Train the model
    print("\n--- Training Player Availability Model ---")
    availability_model.train(league_obs, adps)

    # 3. Save the trained model for later use
    availability_model.save()

    print(f"\nTraining complete. Model saved to {availability_model.model_file}")
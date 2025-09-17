"""
Enhanced modeling approaches for Fantasy Football predictions.

This module provides position-specific models, ensemble methods, and improved
cross-validation strategies for better prediction accuracy and uncertainty quantification.
"""

import json
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# ML imports
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, train_test_split
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor, distns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from src.paths import root_dir
from src.constants import PRO_TEAM_MAP


class PositionSpecificModeling:
    """
    Create and manage position-specific models with tailored features and hyperparameters.
    """
    
    def __init__(self):
        self.models = {}
        self.position_features = {}
        self.feature_importance = {}
        
        # Define position-specific feature priorities
        self._setup_position_features()
    
    def _setup_position_features(self):
        """Define which features are most important for each position."""
        
        self.position_features = {
            'QB': {
                'primary': [
                    'passingYards', 'passingTouchdowns', 'passingAttempts', 
                    'passingCompletions', 'passingInterceptions', 'rushingYards',
                    'passing_yards_per_attempt', 'passing_td_rate', 'interception_rate'
                ],
                'rolling_focus': ['passingYards', 'passingTouchdowns', 'passingAttempts'],
                'context': ['game_total', 'team_implied_total', 'is_dome', 'wind_speed', 'temperature']
            },
            'RB': {
                'primary': [
                    'rushingYards', 'rushingAttempts', 'rushingTouchdowns',
                    'receivingYards', 'receivingTargets', 'receivingReceptions',
                    'rushing_yards_per_attempt', 'target_share', 'carry_share'
                ],
                'rolling_focus': ['rushingAttempts', 'receivingTargets', 'rushingYards'],
                'context': ['game_total', 'spread_favored', 'pace_environment']
            },
            'WR': {
                'primary': [
                    'receivingYards', 'receivingTargets', 'receivingReceptions',
                    'receivingTouchdowns', 'catch_rate', 'yards_per_reception',
                    'yards_per_target', 'target_share'
                ],
                'rolling_focus': ['receivingTargets', 'receivingYards', 'receivingReceptions'],
                'context': ['game_total', 'team_implied_total', 'is_windy', 'is_dome']
            },
            'TE': {
                'primary': [
                    'receivingYards', 'receivingTargets', 'receivingReceptions',
                    'receivingTouchdowns', 'catch_rate', 'yards_per_reception',
                    'target_share', 'reception_share'
                ],
                'rolling_focus': ['receivingTargets', 'receivingYards', 'receivingReceptions'],
                'context': ['game_total', 'team_implied_total', 'pace_environment']
            },
            'K': {
                'primary': [
                    'kicking1-19', 'kicking20-29', 'kicking30-39', 'kicking40-49',
                    'kicking50+', 'kickingXPMade'
                ],
                'rolling_focus': ['kickingFGMade', 'kickingFGAtt'],
                'context': ['is_dome', 'wind_speed', 'altitude_factor', 'temperature']
            },
            'D/ST': {
                'primary': [
                    'defensiveSacks', 'defensiveInterceptions', 'defensiveFumbleRecoveries',
                    'defensiveBlockedKicks', 'defensivePointsAllowed', 'defensiveTouchdowns'
                ],
                'rolling_focus': ['defensiveSacks', 'defensiveInterceptions'],
                'context': ['opponent_implied_total', 'spread_favored', 'pace_environment']
            }
        }
    
    def get_position_features(self, position: str, df: pd.DataFrame) -> List[str]:
        """Get the most relevant features for a specific position."""
        pos_config = self.position_features.get(position, {})
        
        # Start with primary features
        features = []
        
        # Add primary features that exist in the data
        primary = pos_config.get('primary', [])
        features.extend([f for f in primary if f in df.columns])
        
        # Add rolling versions of key features
        rolling_focus = pos_config.get('rolling_focus', [])
        for stat in rolling_focus:
            for window in [3, 6, 10]:
                rolling_features = [
                    f'{stat}_{window}game_mean',
                    f'{stat}_{window}game_std',
                    f'{stat}_{window}game_momentum',
                    f'{stat}_{window}game_consistency'
                ]
                features.extend([f for f in rolling_features if f in df.columns])
        
        # Add contextual features
        context = pos_config.get('context', [])
        features.extend([f for f in context if f in df.columns])
        
        # Add general features that apply to all positions
        general_features = [
            'age', 'experience', 'gamesPlayed', 'height', 'weight',
            'is_rookie', 'draft_pick', 'draft_grade', 'days_rest',
            'short_rest', 'long_rest', 'travel_distance'
        ]
        features.extend([f for f in general_features if f in df.columns])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(features))
    
    def create_position_pipeline(self, position: str, params: Dict) -> Pipeline:
        """Create a model pipeline tailored for a specific position."""
        
        # Position-specific preprocessing
        if position in ['QB', 'RB', 'WR', 'TE']:
            # Skill positions benefit from more complex models
            base_model = LGBMRegressor(
                random_state=42,
                verbose=-1,
                **params.get('lgbm', {})
            )
        elif position == 'K':
            # Kickers are more predictable, simpler model
            base_model = ElasticNet(
                random_state=42,
                **params.get('elastic', {})
            )
        else:  # D/ST
            # Defense benefits from tree-based models
            base_model = LGBMRegressor(
                random_state=42,
                verbose=-1,
                **params.get('lgbm', {})
            )
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', self._create_preprocessor()),
            ('model', base_model)
        ])
        
        return pipeline
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline for position-specific models."""
        
        # Numerical preprocessing
        numeric_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # For now, assume all features are numeric
        # In practice, you'd identify categorical columns
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, slice(None))
        ])
        
        return preprocessor
    
    def optimize_position_model(self, position: str, X: pd.DataFrame, y: pd.Series,
                              n_trials: int = 50) -> Dict:
        """Optimize hyperparameters for a position-specific model."""
        
        def objective(trial):
            if position in ['QB', 'RB', 'WR', 'TE', 'D/ST']:
                params = {
                    'lgbm': {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 2.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 2.0, log=True),
                    }
                }
            else:  # Kicker
                params = {
                    'elastic': {
                        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                    }
                }
            
            pipeline = self.create_position_pipeline(position, params)
            
            # Time-aware cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_scores.append(rmse)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train_position_model(self, position: str, df: pd.DataFrame, 
                           target_col: str) -> Pipeline:
        """Train a model for a specific position."""
        
        # Filter to position
        pos_df = df[df['position'] == position].copy()
        if pos_df.empty:
            raise ValueError(f"No data found for position {position}")
        
        # Get position-relevant features
        features = self.get_position_features(position, pos_df)
        
        X = pos_df[features]
        y = pos_df[target_col]
        
        # Optimize hyperparameters
        print(f"Optimizing hyperparameters for {position}...")
        best_params = self.optimize_position_model(position, X, y)
        
        # Train final model
        pipeline = self.create_position_pipeline(position, best_params)
        pipeline.fit(X, y)
        
        # Store feature importance if available
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importance = pipeline.named_steps['model'].feature_importances_
            self.feature_importance[position] = dict(zip(feature_names, importance))
        
        self.models[position] = pipeline
        
        # Calculate and print performance metrics
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        print(f"{position} model performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Features used: {len(features)}")
        
        return pipeline
    
    def predict_by_position(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions using position-specific models."""
        predictions = np.zeros(len(df))
        
        for position, model in self.models.items():
            pos_mask = df['position'] == position
            if not pos_mask.any():
                continue
            
            pos_df = df[pos_mask]
            features = self.get_position_features(position, pos_df)
            X = pos_df[features]
            
            pos_predictions = model.predict(X)
            predictions[pos_mask] = pos_predictions
        
        return pd.Series(predictions, index=df.index)


class EnsembleModeling:
    """
    Ensemble methods combining multiple modeling approaches for robust predictions.
    """
    
    def __init__(self):
        self.ensemble_model = None
        self.base_models = {}
        self.model_weights = {}
    
    def create_base_models(self, params: Dict) -> Dict:
        """Create a diverse set of base models for ensembling."""
        
        base_models = {
            'lgbm': LGBMRegressor(
                random_state=42,
                verbose=-1,
                **params.get('lgbm', {})
            ),
            'ridge': Ridge(
                random_state=42,
                **params.get('ridge', {})
            ),
            'elastic': ElasticNet(
                random_state=42,
                **params.get('elastic', {})
            )
        }
        
        # Add NGBoost for uncertainty quantification
        if params.get('include_ngboost', False):
            base_models['ngboost'] = NGBRegressor(
                random_state=42,
                **params.get('ngboost', {})
            )
        
        return base_models
    
    def create_stacking_ensemble(self, base_models: Dict, 
                               meta_learner_params: Dict) -> StackingRegressor:
        """Create a stacking ensemble with a meta-learner."""
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        meta_learner = Ridge(**meta_learner_params.get('ridge', {}))
        
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # Use cross-validation for stacking
            n_jobs=-1
        )
        
        return ensemble
    
    def create_voting_ensemble(self, base_models: Dict, 
                             weights: Optional[List[float]] = None) -> VotingRegressor:
        """Create a voting ensemble."""
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=-1
        )
        
        return ensemble
    
    def optimize_ensemble_weights(self, models: Dict, X: pd.DataFrame, 
                                y: pd.Series, n_trials: int = 100) -> Dict[str, float]:
        """Optimize ensemble weights using Optuna."""
        
        def objective(trial):
            weights = {}
            for name in models.keys():
                weights[name] = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight == 0:
                return float('inf')
            
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate ensemble predictions
            predictions = np.zeros(len(X))
            for name, weight in normalized_weights.items():
                model_pred = models[name].predict(X)
                predictions += weight * model_pred
            
            rmse = np.sqrt(mean_squared_error(y, predictions))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Extract and normalize best weights
        best_weights = {}
        for name in models.keys():
            best_weights[name] = study.best_params[f'weight_{name}']
        
        total_weight = sum(best_weights.values())
        normalized_weights = {k: v/total_weight for k, v in best_weights.items()}
        
        return normalized_weights
    
    def train_ensemble(self, df: pd.DataFrame, target_col: str, 
                      features: List[str], ensemble_type: str = 'stacking') -> Any:
        """Train an ensemble model."""
        
        X = df[features]
        y = df[target_col]
        
        # Create and train base models
        base_params = {
            'lgbm': {'n_estimators': 500, 'learning_rate': 0.1},
            'ridge': {'alpha': 1.0},
            'elastic': {'alpha': 1.0, 'l1_ratio': 0.5}
        }
        
        base_models = self.create_base_models(base_params)
        
        print("Training base models...")
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            print(f"  {name} R²: {r2:.4f}")
        
        self.base_models = base_models
        
        # Create ensemble
        if ensemble_type == 'stacking':
            meta_params = {'ridge': {'alpha': 1.0}}
            self.ensemble_model = self.create_stacking_ensemble(base_models, meta_params)
        elif ensemble_type == 'voting_optimized':
            # Optimize weights
            print("Optimizing ensemble weights...")
            weights = self.optimize_ensemble_weights(base_models, X, y)
            self.model_weights = weights
            weight_list = [weights[name] for name in base_models.keys()]
            self.ensemble_model = self.create_voting_ensemble(base_models, weight_list)
        else:  # voting_equal
            self.ensemble_model = self.create_voting_ensemble(base_models)
        
        print(f"Training {ensemble_type} ensemble...")
        self.ensemble_model.fit(X, y)
        
        # Evaluate ensemble
        y_pred = self.ensemble_model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Ensemble performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        if hasattr(self, 'model_weights') and self.model_weights:
            print("  Model weights:")
            for name, weight in self.model_weights.items():
                print(f"    {name}: {weight:.3f}")
        
        return self.ensemble_model


class TimeAwareCrossValidation:
    """
    Improved cross-validation strategies that respect temporal structure.
    """
    
    @staticmethod
    def time_series_split_by_season(df: pd.DataFrame, n_splits: int = 3) -> List[Tuple]:
        """
        Create train/validation splits that respect season boundaries.
        """
        seasons = sorted(df['season'].unique())
        
        if len(seasons) < n_splits + 1:
            raise ValueError(f"Not enough seasons ({len(seasons)}) for {n_splits} splits")
        
        splits = []
        for i in range(n_splits):
            train_seasons = seasons[:-(n_splits-i)]
            val_season = seasons[-(n_splits-i)]
            
            train_idx = df[df['season'].isin(train_seasons)].index
            val_idx = df[df['season'] == val_season].index
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    @staticmethod
    def player_grouped_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple]:
        """
        Create splits that keep all games from the same player together.
        """
        gkf = GroupKFold(n_splits=n_splits)
        groups = df['player_id']
        
        splits = []
        for train_idx, val_idx in gkf.split(df, groups=groups):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def evaluate_model_with_time_cv(self, model, X: pd.DataFrame, y: pd.Series,
                                  df: pd.DataFrame, cv_type: str = 'time_series') -> Dict:
        """
        Evaluate a model using time-aware cross-validation.
        """
        if cv_type == 'time_series':
            splits = self.time_series_split_by_season(df)
        elif cv_type == 'player_grouped':
            splits = self.player_grouped_split(df)
        else:
            # Fallback to sklearn TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(X))
        
        cv_scores = {
            'r2': [],
            'rmse': [],
            'mae': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            
            cv_scores['r2'].append(r2)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            
            print(f"Fold {fold + 1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        # Calculate summary statistics
        summary = {}
        for metric in cv_scores:
            values = cv_scores[metric]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        
        return summary


def save_model_artifacts(models: Dict, feature_importance: Dict, 
                       performance_metrics: Dict, output_dir: Path):
    """Save model artifacts for later use."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Save feature importance
    with open(output_dir / 'feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    # Save performance metrics
    with open(output_dir / 'performance_metrics.json', 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    print(f"Model artifacts saved to {output_dir}")


def create_model_comparison_report(models: Dict, X: pd.DataFrame, 
                                 y: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Create a comparison report of different modeling approaches."""
    
    cv_evaluator = TimeAwareCrossValidation()
    results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics = cv_evaluator.evaluate_model_with_time_cv(model, X, y, df)
        
        result = {
            'model': model_name,
            **metrics
        }
        results.append(result)
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('r2_mean', ascending=False)
    
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    return comparison_df
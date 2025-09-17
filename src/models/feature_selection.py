"""
A streamlined feature selection module for Fantasy Football ML models, focusing on
tree-based feature importance.
"""
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class FeatureSelector:
    """
    Selects the most important features using a Random Forest model, based on a
    cumulative importance threshold.
    """

    def __init__(self,
                 max_features: Optional[int] = None,
                 cumulative_importance_threshold: float = 0.80):
        """
        Initialize the feature selector.

        Args:
            max_features: An optional hard limit on the maximum number of features to select.
            cumulative_importance_threshold: The target cumulative feature importance
                                           to reach (e.g., 0.80 for 80%).
        """
        if not 0 < cumulative_importance_threshold <= 1.0:
            raise ValueError("cumulative_importance_threshold must be between 0 and 1.")

        self.max_features = max_features
        self.cumulative_importance_threshold = cumulative_importance_threshold
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[pd.Series] = None
        self.numerical_imputer_ = None
        self.categorical_imputer_ = None

    def _impute_data(self, X: pd.DataFrame, fit_imputer: bool = True) -> pd.DataFrame:
        """
        Handle missing values using imputation strategies consistent with the main ML pipelines.
        Uses SimpleImputer
        """
        X_imputed = X.copy()
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if numerical_cols:
            if fit_imputer:
                self.numerical_imputer_ = SimpleImputer(strategy='median')
                X_imputed[numerical_cols] = self.numerical_imputer_.fit_transform(X[numerical_cols])
            elif self.numerical_imputer_:
                X_imputed[numerical_cols] = self.numerical_imputer_.transform(X[numerical_cols])

        if categorical_cols:
            if fit_imputer:
                self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
                X_imputed[categorical_cols] = self.categorical_imputer_.fit_transform(X[categorical_cols].astype(str))
            elif self.categorical_imputer_:
                X_imputed[categorical_cols] = self.categorical_imputer_.transform(X[categorical_cols].astype(str))

        return X_imputed

    def _tree_importance_selection(self, X: pd.DataFrame, y: pd.Series,
                                   task_type: str) -> List[str]:
        """Apply tree-based feature importance selection."""
        model = (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                 if task_type == 'regression' else
                 RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

        model.fit(X, y)
        importances = model.feature_importances_

        feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        self.feature_scores_ = feature_importance

        # Select top features based on cumulative importance
        cumulative_importance = feature_importance.cumsum()
        top_features = cumulative_importance[
            cumulative_importance <= self.cumulative_importance_threshold
        ].index.tolist()

        # If max_features is set, further reduce to that hard limit
        if self.max_features and len(top_features) > self.max_features:
            top_features = feature_importance.head(self.max_features).index.tolist()

        # Ensure at least one feature is selected
        if not top_features and not X.empty:
            return [feature_importance.index[0]]

        return top_features

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                      task_type: str = 'regression') -> pd.DataFrame:
        """
        Fit the feature selector on the data and return the transformed dataframe.

        Args:
            X: Feature matrix.
            y: Target variable.
            task_type: 'regression' or 'classification'.

        Returns:
            DataFrame with only the selected features.
        """
        # Drop columns with no variance before processing
        X_clean = X.loc[:, X.nunique(dropna=True) > 1]

        # Impute data temporarily to allow the Random Forest to run
        X_imputed = self._impute_data(X_clean, fit_imputer=True)

        # Temporarily encode categorical features for the selection model
        X_processed = X_imputed.copy()
        categorical_cols = X_imputed.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

        # Perform the feature selection
        selected_features = self._tree_importance_selection(X_processed, y, task_type)
        self.selected_features_ = selected_features

        # Return the original dataframe with the selected columns
        return X[self.selected_features_]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataframe using the already-fitted feature selection.
        This method's only job is to select the correct columns.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Feature selector must be fitted before transform can be called.")

        return X[self.selected_features_]

    def get_feature_info(self) -> Dict:
        """Get information about the selected features."""
        if self.selected_features_ is None:
            return {}

        return {
            'selected_features': self.selected_features_,
            'n_selected': len(self.selected_features_),
            'feature_scores': self.feature_scores_.to_dict()
        }


def apply_feature_selection_to_pipeline(X: pd.DataFrame, y: pd.Series,
                                        task_type: str = 'regression',
                                        max_features: Optional[int] = None,
                                        cumulative_importance_threshold: float = 0.80,
                                        ) -> Tuple[pd.DataFrame, FeatureSelector]:
    """
    Convenience function to apply feature selection.

    Args:
        X: Feature matrix.
        y: Target variable.
        task_type: 'regression' or 'classification'.
        max_features: Maximum number of features to select.
        cumulative_importance_threshold: The target cumulative feature importance.

    Returns:
        A tuple of (DataFrame with selected features, fitted FeatureSelector instance).
    """

    original_features = X.shape[1]
    selector = FeatureSelector(
        max_features=max_features,
        cumulative_importance_threshold=cumulative_importance_threshold
    )
    X_selected = selector.fit_transform(X, y, task_type)

    info = selector.get_feature_info()
    print("--- Feature Selection Results ---")
    print(f"  Original features: {original_features}")
    print(f"  Selected features: {info['n_selected']}")
    print(f"  Reduction: {((original_features - info['n_selected']) / original_features * 100):.1f}%")

    return X_selected, selector
"""
Unit tests for feature selection module.

This test file ensures the feature selection functionality works correctly
with different data types and selection methods.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.feature_selection import FeatureSelector, apply_feature_selection_to_pipeline


class TestFeatureSelection(unittest.TestCase):
    """Test cases for feature selection functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create mixed data (numerical and categorical)
        self.X = pd.DataFrame({
            # Categorical features
            'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
            
            # Numerical features - some correlated
            **{f'num_{i}': np.random.randn(n_samples) for i in range(n_features-2)}
        })
        
        # Add highly correlated feature for correlation testing
        self.X['num_corr'] = self.X['num_0'] + np.random.randn(n_samples) * 0.01
        
        # Create targets
        self.y_reg = (self.X['num_0'] * 2 + 
                     self.X['num_1'] * -1 + 
                     np.random.randn(n_samples) * 0.1)
        
        self.y_class = (self.y_reg > self.y_reg.median()).astype(int)
    
    def test_feature_selector_initialization(self):
        """Test FeatureSelector initialization with different parameters."""
        selector = FeatureSelector()
        self.assertEqual(selector.selection_method, 'combined')
        self.assertEqual(selector.feature_percentile, 80.0)
        
        selector = FeatureSelector(
            selection_method='tree_importance',
            max_features=10,
            feature_percentile=70.0
        )
        self.assertEqual(selector.selection_method, 'tree_importance')
        self.assertEqual(selector.max_features, 10)
        self.assertEqual(selector.feature_percentile, 70.0)
    
    def test_correlation_filtering(self):
        """Test correlation-based feature filtering."""
        selector = FeatureSelector(selection_method='correlation')
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        # Should remove the highly correlated feature
        self.assertLess(X_selected.shape[1], self.X.shape[1])
        self.assertIn('category_1', X_selected.columns)  # Should keep categorical features
        self.assertIn('num_0', X_selected.columns)  # Should keep original feature
    
    def test_tree_importance_selection(self):
        """Test tree-based feature importance selection."""
        selector = FeatureSelector(
            selection_method='tree_importance',
            max_features=10
        )
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        self.assertEqual(X_selected.shape[1], 10)
        self.assertIsNotNone(selector.selected_features_)
        self.assertIsNotNone(selector.feature_scores_)
        
        # Test with classification
        X_selected_class = selector.fit_transform(self.X, self.y_class, task_type='classification')
        self.assertEqual(X_selected_class.shape[1], 10)
    
    def test_lasso_selection(self):
        """Test LASSO-based feature selection."""
        selector = FeatureSelector(selection_method='lasso')
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        # LASSO should select a subset of features
        self.assertLessEqual(X_selected.shape[1], self.X.shape[1])
        self.assertGreater(X_selected.shape[1], 0)  # Should select at least some features
        
        # Test with classification
        X_selected_class = selector.fit_transform(self.X, self.y_class, task_type='classification')
        self.assertLessEqual(X_selected_class.shape[1], self.X.shape[1])
    
    def test_univariate_selection(self):
        """Test univariate feature selection."""
        selector = FeatureSelector(
            selection_method='univariate',
            feature_percentile=50.0
        )
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        # Should select approximately 50% of features
        expected_features = int(self.X.shape[1] * 0.5)
        self.assertAlmostEqual(X_selected.shape[1], expected_features, delta=2)
    
    def test_combined_selection(self):
        """Test combined feature selection method."""
        selector = FeatureSelector(
            selection_method='combined',
            max_features=8
        )
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        self.assertEqual(X_selected.shape[1], 8)
        self.assertIsInstance(selector.feature_scores_, dict)
    
    def test_transform_consistency(self):
        """Test that transform produces consistent results."""
        selector = FeatureSelector(selection_method='tree_importance', max_features=5)
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        # Transform on same data should produce same result
        X_transformed = selector.transform(self.X)
        pd.testing.assert_frame_equal(X_selected, X_transformed)
        
        # Transform on subset should work
        X_subset = self.X.iloc[:50]
        X_transformed_subset = selector.transform(X_subset)
        self.assertEqual(X_transformed_subset.shape[1], X_selected.shape[1])
        self.assertEqual(X_transformed_subset.shape[0], 50)
    
    def test_convenience_function(self):
        """Test the apply_feature_selection_to_pipeline convenience function."""
        X_selected, selector = apply_feature_selection_to_pipeline(
            self.X, self.y_reg,
            task_type='regression',
            method='tree_importance',
            max_features=6,
            verbose=False
        )
        
        self.assertEqual(X_selected.shape[1], 6)
        self.assertIsInstance(selector, FeatureSelector)
        self.assertEqual(len(selector.selected_features_), 6)
    
    def test_missing_data_handling(self):
        """Test feature selection with missing data."""
        # Create dataset with missing values
        X_missing = self.X.copy()
        
        # Introduce random missing values
        np.random.seed(42)
        
        # Add missing values to numerical columns
        for col in ['num_0', 'num_1', 'num_2']:
            missing_mask = np.random.rand(len(X_missing)) < 0.2  # 20% missing
            X_missing.loc[missing_mask, col] = np.nan
        
        # Add missing values to categorical columns
        for col in ['category_1', 'category_2']:
            missing_mask = np.random.rand(len(X_missing)) < 0.15  # 15% missing
            X_missing.loc[missing_mask, col] = None
        
        # Test different selection methods with missing data
        methods = ['univariate', 'tree_importance', 'lasso', 'correlation', 'rfe', 'combined']
        
        for method in methods:
            with self.subTest(method=method):
                selector = FeatureSelector(selection_method=method, max_features=5)
                
                # Should handle missing data without errors
                X_selected = selector.fit_transform(X_missing, self.y_reg, task_type='regression')
                
                # Verify results
                self.assertIsInstance(X_selected, pd.DataFrame)
                self.assertLessEqual(X_selected.shape[1], 5)
                self.assertEqual(X_selected.shape[0], X_missing.shape[0])
                
                # Should not contain missing values after imputation
                self.assertFalse(X_selected.isnull().any().any())
                
                # Test transform on new data with missing values
                X_new_missing = X_missing.iloc[:20].copy()
                X_transformed = selector.transform(X_new_missing)
                
                self.assertIsInstance(X_transformed, pd.DataFrame)
                self.assertEqual(X_transformed.shape[1], X_selected.shape[1])
                self.assertFalse(X_transformed.isnull().any().any())
    
    def test_all_missing_data(self):
        """Test handling of completely missing columns."""
        X_all_missing = self.X.copy()
        
        # Create a column with all missing values
        X_all_missing['all_missing'] = np.nan
        X_all_missing['all_missing_cat'] = None
        
        selector = FeatureSelector(selection_method='combined', max_features=5)
        
        # Should handle all missing data gracefully
        X_selected = selector.fit_transform(X_all_missing, self.y_reg, task_type='regression')
        
        self.assertIsInstance(X_selected, pd.DataFrame)
        self.assertFalse(X_selected.isnull().any().any())
        self.assertLessEqual(X_selected.shape[1], 5)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        selector = FeatureSelector()
        
        # Test with very small dataset
        X_small = self.X.iloc[:10, :5]
        y_small = self.y_reg.iloc[:10]
        
        X_selected = selector.fit_transform(X_small, y_small, task_type='regression')
        self.assertLessEqual(X_selected.shape[1], X_small.shape[1])
        
        # Test transform before fit should raise error
        fresh_selector = FeatureSelector()
        with self.assertRaises(ValueError):
            fresh_selector.transform(self.X)
    
    def test_feature_info(self):
        """Test feature information retrieval."""
        selector = FeatureSelector(selection_method='tree_importance', max_features=5)
        X_selected = selector.fit_transform(self.X, self.y_reg, task_type='regression')
        
        info = selector.get_feature_info()
        
        self.assertIn('selected_features', info)
        self.assertIn('n_selected', info)
        self.assertIn('selection_method', info)
        self.assertIn('feature_scores', info)
        
        self.assertEqual(info['n_selected'], 5)
        self.assertEqual(info['selection_method'], 'tree_importance')
        self.assertEqual(len(info['selected_features']), 5)


if __name__ == '__main__':
    unittest.main()
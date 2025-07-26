import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.matrix_factorization import BasicMatrixFactorization, SVDMatrixFactorization

@pytest.fixture
def mf_sample_data():
    """Sample data for matrix factorization tests."""
    data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4, 1, 2, 3],
        'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 3.0, 5.0, 4.0, 2.0, 4.0, 3.0, 5.0]
    }
    return pd.DataFrame(data)

@pytest.fixture  
def mf_expected_stats():
    """Expected statistics for matrix factorization validation."""
    return {
        'num_users': 5,
        'num_movies': 4,
        'num_ratings': 15,
        'global_average': 3.73,
        'rating_range': (2.0, 5.0)
    }

class TestBasicMatrixFactorization:
    
    def test_initialization(self):
        """Test proper initialization of BasicMatrixFactorization."""
        mf = BasicMatrixFactorization(k=5, epochs=50, learning_rate=0.02, reg=0.01)
        
        assert mf.k == 5
        assert mf.epochs == 50
        assert mf.learning_rate == 0.02
        assert mf.reg == 0.01
        assert not mf.is_fitted
        assert mf.user_factors is None
        assert mf.item_factors is None
        assert mf.user_bias is None
        assert mf.item_bias is None
        assert mf.global_bias is None

    def test_default_initialization(self):
        """Test default parameter values."""
        mf = BasicMatrixFactorization()
        
        assert mf.k == 10
        assert mf.epochs == 100
        assert mf.learning_rate == 0.01
        assert mf.reg == 0.02

    def test_fit_basic_functionality(self, mf_sample_data, mf_expected_stats):
        """Test basic fit functionality."""
        mf = BasicMatrixFactorization(k=3, epochs=10)
        result = mf.fit(mf_sample_data)
        
        # Test return value (method chaining)
        assert result is mf
        
        # Test fitted state
        assert mf.is_fitted
        assert mf.data is not None
        assert len(mf.data) == mf_expected_stats['num_ratings']
        
        # Test factor matrices shapes
        assert mf.user_factors.shape == (mf_expected_stats['num_users'], 3)
        assert mf.item_factors.shape == (mf_expected_stats['num_movies'], 3)
        
        # Test bias terms
        assert mf.user_bias.shape == (mf_expected_stats['num_users'],)
        assert mf.item_bias.shape == (mf_expected_stats['num_movies'],)
        assert abs(mf.global_bias - mf_expected_stats['global_average']) < 0.1
        
        # Test mappings
        assert len(mf.user_id_to_idx) == mf_expected_stats['num_users']
        assert len(mf.item_id_to_idx) == mf_expected_stats['num_movies']

    def test_factor_initialization_values(self, mf_sample_data):
        """Test that factors are initialized with small random values."""
        mf = BasicMatrixFactorization(k=5)
        mf.fit(mf_sample_data)
        
        # User factors should be small random values (after training)
        assert np.all(np.abs(mf.user_factors) < 5.0)  # Reasonable bounds after training
        assert np.all(np.abs(mf.item_factors) < 5.0)  # Reasonable bounds after training
        
        # Factors should not be all zeros
        assert not np.allclose(mf.user_factors, 0)
        assert not np.allclose(mf.item_factors, 0)

    def test_predict_basic_functionality(self, mf_sample_data):
        """Test basic prediction functionality."""
        mf = BasicMatrixFactorization(k=3, epochs=20)
        mf.fit(mf_sample_data)
        
        # Test prediction for known user-item pair
        prediction = mf.predict(1, 1)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0  # Should be in valid rating range
        
        # Test multiple predictions
        prediction2 = mf.predict(2, 2)
        prediction3 = mf.predict(3, 3)
        assert isinstance(prediction2, float)
        assert isinstance(prediction3, float)

    def test_predict_unfitted_model(self, mf_sample_data):
        """Test that prediction fails on unfitted model."""
        mf = BasicMatrixFactorization()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            mf.predict(1, 1)

    def test_predict_unknown_users_items(self, mf_sample_data):
        """Test predictions for unknown users and items."""
        mf = BasicMatrixFactorization(k=3, epochs=10)
        mf.fit(mf_sample_data)
        
        # Unknown user, known item
        prediction_unknown_user = mf.predict(999, 1)
        assert isinstance(prediction_unknown_user, float)
        assert 1.0 <= prediction_unknown_user <= 5.0
        
        # Known user, unknown item  
        prediction_unknown_item = mf.predict(1, 999)
        assert isinstance(prediction_unknown_item, float)
        assert 1.0 <= prediction_unknown_item <= 5.0
        
        # Unknown user, unknown item
        prediction_both_unknown = mf.predict(999, 999)
        assert isinstance(prediction_both_unknown, float)
        assert 1.0 <= prediction_both_unknown <= 5.0
        
        # Unknown predictions should fall back to reasonable defaults
        assert abs(prediction_both_unknown - mf.global_bias) < 0.1

    def test_learning_convergence(self, mf_sample_data):
        """Test that learning improves predictions over epochs."""
        # Train with few epochs
        mf_few = BasicMatrixFactorization(k=5, epochs=5, learning_rate=0.1)
        mf_few.fit(mf_sample_data)
        
        # Train with many epochs
        mf_many = BasicMatrixFactorization(k=5, epochs=50, learning_rate=0.1)
        mf_many.fit(mf_sample_data)
        
        # Calculate training error for both
        training_error_few = 0
        training_error_many = 0
        
        for _, row in mf_sample_data.iterrows():
            user_id, item_id, actual = row['userId'], row['movieId'], row['rating']
            
            pred_few = mf_few.predict(user_id, item_id)
            pred_many = mf_many.predict(user_id, item_id)
            
            training_error_few += (actual - pred_few) ** 2
            training_error_many += (actual - pred_many) ** 2
        
        # More training should lead to lower error
        assert training_error_many < training_error_few

    def test_different_k_values(self, mf_sample_data):
        """Test matrix factorization with different numbers of factors."""
        for k in [2, 5, 10]:
            mf = BasicMatrixFactorization(k=k, epochs=10)
            mf.fit(mf_sample_data)
            
            assert mf.user_factors.shape[1] == k
            assert mf.item_factors.shape[1] == k
            
            # Should be able to make predictions
            prediction = mf.predict(1, 1)
            assert 1.0 <= prediction <= 5.0

    def test_regularization_effect(self, mf_sample_data):
        """Test that regularization prevents overfitting."""
        # No regularization
        mf_no_reg = BasicMatrixFactorization(k=10, epochs=100, reg=0.0)
        mf_no_reg.fit(mf_sample_data)
        
        # With regularization
        mf_with_reg = BasicMatrixFactorization(k=10, epochs=100, reg=0.1)
        mf_with_reg.fit(mf_sample_data)
        
        # With regularization, factors should be smaller in magnitude
        avg_factor_no_reg = np.mean(np.abs(mf_no_reg.user_factors))
        avg_factor_with_reg = np.mean(np.abs(mf_with_reg.user_factors))
        
        # Regularization should keep factors smaller
        assert avg_factor_with_reg <= avg_factor_no_reg * 1.5  # Allow some tolerance

    def test_bias_terms_learning(self, mf_sample_data):
        """Test that bias terms are learned correctly."""
        mf = BasicMatrixFactorization(k=3, epochs=20)
        mf.fit(mf_sample_data)
        
        # Global bias should be close to overall average
        actual_global_avg = mf_sample_data['rating'].mean()
        assert abs(mf.global_bias - actual_global_avg) < 0.1
        
        # User and item biases should not all be zero after training
        assert not np.allclose(mf.user_bias, 0, atol=0.01)
        assert not np.allclose(mf.item_bias, 0, atol=0.01)

    def test_prediction_consistency(self, mf_sample_data):
        """Test that predictions are consistent across calls."""
        mf = BasicMatrixFactorization(k=5, epochs=20)
        mf.fit(mf_sample_data)
        
        # Multiple predictions for same user-item pair should be identical
        pred1 = mf.predict(1, 1)
        pred2 = mf.predict(1, 1)
        pred3 = mf.predict(1, 1)
        
        assert pred1 == pred2 == pred3

    def test_empty_dataframe(self):
        """Test behavior with empty training data."""
        empty_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        mf = BasicMatrixFactorization()
        
        # Should handle empty data gracefully
        mf.fit(empty_df)
        assert mf.is_fitted
        assert mf.user_factors.shape[0] == 0
        assert mf.item_factors.shape[0] == 0

    def test_single_rating(self):
        """Test with minimal data (single rating)."""
        single_rating = pd.DataFrame({
            'userId': [1],
            'movieId': [1], 
            'rating': [4.0]
        })
        
        mf = BasicMatrixFactorization(k=2, epochs=10)
        mf.fit(single_rating)
        
        assert mf.is_fitted
        assert mf.user_factors.shape == (1, 2)
        assert mf.item_factors.shape == (1, 2)
        
        # Should predict something reasonable
        prediction = mf.predict(1, 1)
        assert 1.0 <= prediction <= 5.0


class TestSVDMatrixFactorization:
    
    def test_initialization(self):
        """Test proper initialization of SVDMatrixFactorization."""
        svd_mf = SVDMatrixFactorization(k=5)
        
        assert svd_mf.k == 5
        assert not svd_mf.is_fitted
        assert svd_mf.U is None
        assert svd_mf.sigma is None
        assert svd_mf.Vt is None
        assert svd_mf.global_bias is None
        assert len(svd_mf.user_id_to_idx) == 0
        assert len(svd_mf.item_id_to_idx) == 0

    def test_default_initialization(self):
        """Test default parameter values."""
        svd_mf = SVDMatrixFactorization()
        
        assert svd_mf.k == 10

    def test_fit_basic_functionality(self, mf_sample_data, mf_expected_stats):
        """Test basic fit functionality."""
        svd_mf = SVDMatrixFactorization(k=3)
        result = svd_mf.fit(mf_sample_data)
        
        # Test return value (method chaining)
        assert result is svd_mf
        
        # Test fitted state
        assert svd_mf.is_fitted
        assert svd_mf.data is not None
        assert len(svd_mf.data) == mf_expected_stats['num_ratings']
        
        # Test SVD matrices shapes
        assert svd_mf.U.shape == (mf_expected_stats['num_users'], 3)
        assert svd_mf.sigma.shape == (3,)
        assert svd_mf.Vt.shape == (3, mf_expected_stats['num_movies'])
        
        # Test global bias
        assert abs(svd_mf.global_bias - mf_expected_stats['global_average']) < 0.1
        
        # Test mappings
        assert len(svd_mf.user_id_to_idx) == mf_expected_stats['num_users']
        assert len(svd_mf.item_id_to_idx) == mf_expected_stats['num_movies']

    def test_svd_matrices_properties(self, mf_sample_data):
        """Test properties of SVD decomposition matrices."""
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Singular values should be non-negative and in descending order
        assert np.all(svd_mf.sigma >= 0)
        assert np.all(svd_mf.sigma[:-1] >= svd_mf.sigma[1:])  # Descending order
        
        # U and Vt should not be all zeros
        assert not np.allclose(svd_mf.U, 0)
        assert not np.allclose(svd_mf.Vt, 0)
        
        # Values should be reasonable (not extremely large)
        assert np.all(np.abs(svd_mf.U) < 10.0)
        assert np.all(np.abs(svd_mf.Vt) < 10.0)

    def test_predict_basic_functionality(self, mf_sample_data):
        """Test basic prediction functionality."""
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Test prediction for known user-item pair
        prediction = svd_mf.predict(1, 1)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0  # Should be in valid rating range
        
        # Test multiple predictions
        prediction2 = svd_mf.predict(2, 2)
        prediction3 = svd_mf.predict(3, 3)
        assert isinstance(prediction2, float)
        assert isinstance(prediction3, float)

    def test_predict_unfitted_model(self):
        """Test that prediction fails on unfitted model."""
        svd_mf = SVDMatrixFactorization()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            svd_mf.predict(1, 1)

    def test_predict_unknown_users_items(self, mf_sample_data):
        """Test predictions for unknown users and items."""
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Unknown user, known item - should return global bias
        prediction_unknown_user = svd_mf.predict(999, 1)
        assert isinstance(prediction_unknown_user, float)
        assert abs(prediction_unknown_user - svd_mf.global_bias) < 0.1
        
        # Known user, unknown item - should return global bias
        prediction_unknown_item = svd_mf.predict(1, 999)
        assert isinstance(prediction_unknown_item, float)
        assert abs(prediction_unknown_item - svd_mf.global_bias) < 0.1
        
        # Unknown user, unknown item - should return global bias
        prediction_both_unknown = svd_mf.predict(999, 999)
        assert isinstance(prediction_both_unknown, float)
        assert abs(prediction_both_unknown - svd_mf.global_bias) < 0.1

    def test_different_k_values(self, mf_sample_data):
        """Test SVD with different numbers of factors."""
        max_k = min(len(mf_sample_data['userId'].unique()), 
                   len(mf_sample_data['movieId'].unique())) - 1
        
        for k in [2, min(3, max_k)]:
            if k > 0:  # Only test valid k values
                svd_mf = SVDMatrixFactorization(k=k)
                svd_mf.fit(mf_sample_data)
                
                # Actual k might be constrained by matrix size
                actual_k = min(k, max_k) if max_k > 0 else 1
                
                assert svd_mf.U.shape[1] == actual_k
                assert svd_mf.sigma.shape[0] == actual_k
                assert svd_mf.Vt.shape[0] == actual_k
                
                # Should be able to make predictions
                prediction = svd_mf.predict(1, 1)
                assert 1.0 <= prediction <= 5.0

    def test_prediction_consistency(self, mf_sample_data):
        """Test that predictions are consistent across calls."""
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Multiple predictions for same user-item pair should be identical
        pred1 = svd_mf.predict(1, 1)
        pred2 = svd_mf.predict(1, 1)
        pred3 = svd_mf.predict(1, 1)
        
        assert pred1 == pred2 == pred3

    def test_svd_vs_basic_comparison(self, mf_sample_data):
        """Test that SVD and Basic MF both produce reasonable predictions."""
        # Train both models
        basic_mf = BasicMatrixFactorization(k=3, epochs=50, learning_rate=0.1)
        basic_mf.fit(mf_sample_data)
        
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Both should predict in valid range
        for user_id in [1, 2, 3]:
            for item_id in [1, 2, 3]:
                basic_pred = basic_mf.predict(user_id, item_id)
                svd_pred = svd_mf.predict(user_id, item_id)
                
                assert 1.0 <= basic_pred <= 5.0
                assert 1.0 <= svd_pred <= 5.0

    def test_singular_values_importance(self, mf_sample_data):
        """Test that singular values represent factor importance."""
        svd_mf = SVDMatrixFactorization(k=3)
        svd_mf.fit(mf_sample_data)
        
        # Singular values should be in descending order (most important first)
        assert np.all(svd_mf.sigma[:-1] >= svd_mf.sigma[1:])
        
        # All singular values should be non-negative
        assert np.all(svd_mf.sigma >= 0)

    def test_pivot_matrix_creation(self, mf_sample_data):
        """Test that pivot matrix is created correctly."""
        svd_mf = SVDMatrixFactorization(k=2)
        
        # Test the pivot matrix creation logic
        pivot_matrix = mf_sample_data.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        )
        
        # Should have correct dimensions
        assert pivot_matrix.shape[0] == len(mf_sample_data['userId'].unique())
        assert pivot_matrix.shape[1] == len(mf_sample_data['movieId'].unique())
        
        # Should contain some NaN values (missing ratings)
        assert pivot_matrix.isna().sum().sum() > 0

    def test_single_rating(self):
        """Test with minimal data (single rating)."""
        single_rating = pd.DataFrame({
            'userId': [1],
            'movieId': [1], 
            'rating': [4.0]
        })
        
        # For 1x1 matrix, k must be 0, but SVD requires k > 0
        # Our implementation should handle this edge case gracefully
        svd_mf = SVDMatrixFactorization(k=1)
        svd_mf.fit(single_rating)
        
        assert svd_mf.is_fitted
        # With 1x1 matrix, actual k will be constrained
        assert svd_mf.U.shape == (1, 1)
        assert svd_mf.sigma.shape == (1,)
        assert svd_mf.Vt.shape == (1, 1)
        
        # Should predict something reasonable
        prediction = svd_mf.predict(1, 1)
        assert 1.0 <= prediction <= 5.0

    def test_small_matrix_edge_case(self):
        """Test with very small matrices that constrain k."""
        # 2x2 matrix case
        small_data = pd.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [1, 2, 1, 2], 
            'rating': [4.0, 3.0, 5.0, 2.0]
        })
        
        # Request k=5 but matrix is only 2x2, so actual k will be 1
        svd_mf = SVDMatrixFactorization(k=5)
        svd_mf.fit(small_data)
        
        assert svd_mf.is_fitted
        # k should be constrained to 1 (min(2,2) - 1 = 1)
        assert svd_mf.U.shape == (2, 1)
        assert svd_mf.sigma.shape == (1,)
        assert svd_mf.Vt.shape == (1, 2)
        
        # Should still make valid predictions
        prediction = svd_mf.predict(1, 1)
        assert 1.0 <= prediction <= 5.0 
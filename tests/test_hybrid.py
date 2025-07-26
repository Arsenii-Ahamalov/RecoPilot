import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.hybrid import HybridRecommender
from src.algorithms.base import BaseRecommender
from src.algorithms.baselines import GlobalAverageRecommender, UserAverageRecommender
from src.algorithms.matrix_factorization import BasicMatrixFactorization


class MockRecommender(BaseRecommender):
    """Mock recommender for testing purposes."""
    
    def __init__(self, prediction_value=3.5, should_fail=False):
        super().__init__()
        self.prediction_value = prediction_value
        self.should_fail = should_fail
        
    def fit(self, ratings_df):
        self.data = ratings_df
        self.is_fitted = True
        return self
        
    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.should_fail:
            raise Exception("Mock prediction failure")
        return self.prediction_value


class TestHybridRecommenderInitialization:
    """Test HybridRecommender initialization."""
    
    def test_init_with_weights(self):
        """Test initialization with explicit weights."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        weights = {'alg1': 0.6, 'alg2': 0.4}
        
        hybrid = HybridRecommender(algorithms, weights)
        
        assert hybrid.algorithms == algorithms
        assert hybrid.weights == weights
        assert not hybrid.is_fitted
    
    def test_init_without_weights(self):
        """Test initialization with auto-generated equal weights."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        alg3 = MockRecommender(5.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2, 'alg3': alg3}
        
        hybrid = HybridRecommender(algorithms)
        
        expected_weight = 1.0 / 3
        assert abs(hybrid.weights['alg1'] - expected_weight) < 1e-10
        assert abs(hybrid.weights['alg2'] - expected_weight) < 1e-10
        assert abs(hybrid.weights['alg3'] - expected_weight) < 1e-10
        assert abs(sum(hybrid.weights.values()) - 1.0) < 1e-10
    
    def test_init_weights_validation_sum(self):
        """Test that weights must sum to 1.0."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        weights = {'alg1': 0.6, 'alg2': 0.5}  # Sum = 1.1
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            HybridRecommender(algorithms, weights)
    
    def test_init_weights_validation_names(self):
        """Test that algorithm names must match weight names."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        weights = {'alg1': 0.6, 'different_name': 0.4}
        
        with pytest.raises(ValueError, match="Algorithm names and weight names must match"):
            HybridRecommender(algorithms, weights)
    
    def test_init_single_algorithm(self):
        """Test initialization with single algorithm."""
        alg = MockRecommender(3.5)
        algorithms = {'solo': alg}
        
        hybrid = HybridRecommender(algorithms)
        
        assert hybrid.weights['solo'] == 1.0
        assert len(hybrid.algorithms) == 1


class TestHybridRecommenderFitting:
    """Test HybridRecommender fitting functionality."""
    
    def test_fit_success(self):
        """Test successful fitting of all algorithms."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        hybrid = HybridRecommender(algorithms)
        
        ratings_df = pd.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [101, 102, 101, 103],
            'rating': [4.0, 3.5, 5.0, 2.5]
        })
        
        result = hybrid.fit(ratings_df)
        
        assert result is hybrid  # Method chaining
        assert hybrid.is_fitted
        assert alg1.is_fitted
        assert alg2.is_fitted
        assert hybrid.data.equals(ratings_df)
    
    def test_fit_invalid_algorithm_type(self):
        """Test fitting with non-BaseRecommender algorithm."""
        class NotARecommender:
            pass
        
        invalid_alg = NotARecommender()
        valid_alg = MockRecommender(3.0)
        
        algorithms = {'valid': valid_alg, 'invalid': invalid_alg}
        hybrid = HybridRecommender(algorithms)
        
        ratings_df = pd.DataFrame({
            'userId': [1, 2],
            'movieId': [101, 102],
            'rating': [4.0, 3.5]
        })
        
        with pytest.raises(TypeError, match="Algorithm 'invalid' must inherit from BaseRecommender"):
            hybrid.fit(ratings_df)
    
    def test_fit_empty_dataframe(self):
        """Test fitting with empty DataFrame."""
        alg = MockRecommender(3.0)
        hybrid = HybridRecommender({'alg': alg})
        
        empty_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        
        hybrid.fit(empty_df)
        assert hybrid.is_fitted


class TestHybridRecommenderPrediction:
    """Test HybridRecommender prediction functionality."""
    
    def test_predict_weighted_combination(self):
        """Test that predictions are properly weighted."""
        alg1 = MockRecommender(2.0)  # Always predicts 2.0
        alg2 = MockRecommender(4.0)  # Always predicts 4.0
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        weights = {'alg1': 0.3, 'alg2': 0.7}
        
        hybrid = HybridRecommender(algorithms, weights)
        
        ratings_df = pd.DataFrame({
            'userId': [1], 'movieId': [101], 'rating': [3.0]
        })
        hybrid.fit(ratings_df)
        
        prediction = hybrid.predict(1, 102)
        expected = 0.3 * 2.0 + 0.7 * 4.0  # 0.6 + 2.8 = 3.4
        
        assert abs(prediction - expected) < 1e-10
    
    def test_predict_clamping(self):
        """Test that predictions are clamped to [1.0, 5.0]."""
        alg1 = MockRecommender(0.5)   # Below range
        alg2 = MockRecommender(6.0)   # Above range
        
        algorithms = {'low': alg1, 'high': alg2}
        weights = {'low': 1.0, 'high': 0.0}  # Only use low algorithm
        
        hybrid = HybridRecommender(algorithms, weights)
        hybrid.fit(pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]}))
        
        prediction = hybrid.predict(1, 102)
        assert prediction == 1.0  # Clamped to minimum
        
        # Test upper clamp
        weights = {'low': 0.0, 'high': 1.0}  # Only use high algorithm
        hybrid = HybridRecommender(algorithms, weights)
        hybrid.fit(pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]}))
        
        prediction = hybrid.predict(1, 102)
        assert prediction == 5.0  # Clamped to maximum
    
    def test_predict_not_fitted(self):
        """Test prediction before fitting raises error."""
        alg = MockRecommender(3.0)
        hybrid = HybridRecommender({'alg': alg})
        
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            hybrid.predict(1, 101)
    
    def test_predict_algorithm_failure_handling(self):
        """Test handling when some algorithms fail."""
        alg1 = MockRecommender(2.0, should_fail=False)
        alg2 = MockRecommender(4.0, should_fail=True)  # This will fail
        alg3 = MockRecommender(3.0, should_fail=False)
        
        algorithms = {'good1': alg1, 'bad': alg2, 'good2': alg3}
        weights = {'good1': 0.4, 'bad': 0.3, 'good2': 0.3}
        
        hybrid = HybridRecommender(algorithms, weights)
        hybrid.fit(pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]}))
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            prediction = hybrid.predict(1, 102)
        
        # Should only use successful algorithms
        expected = 0.4 * 2.0 + 0.3 * 3.0  # 0.8 + 0.9 = 1.7
        assert abs(prediction - expected) < 1e-10
        
        # Should print warning about failed algorithm
        mock_print.assert_called_once()
        assert "bad prediction failed" in str(mock_print.call_args)
    
    def test_predict_all_algorithms_fail(self):
        """Test when all algorithms fail to predict."""
        alg1 = MockRecommender(3.0, should_fail=True)
        alg2 = MockRecommender(4.0, should_fail=True)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        hybrid = HybridRecommender(algorithms)
        hybrid.fit(pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]}))
        
        with pytest.raises(ValueError, match="All algorithms failed to predict"):
            hybrid.predict(1, 102)


class TestHybridRecommenderInfo:
    """Test HybridRecommender information method."""
    
    def test_get_algorithm_info(self):
        """Test get_algorithm_info method."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'mock1': alg1, 'mock2': alg2}
        weights = {'mock1': 0.6, 'mock2': 0.4}
        
        hybrid = HybridRecommender(algorithms, weights)
        
        info = hybrid.get_algorithm_info()
        
        assert 'mock1' in info
        assert 'mock2' in info
        
        assert info['mock1']['class'] == 'MockRecommender'
        assert info['mock1']['weight'] == 0.6
        assert info['mock1']['fitted'] == False
        
        assert info['mock2']['class'] == 'MockRecommender'
        assert info['mock2']['weight'] == 0.4
        assert info['mock2']['fitted'] == False
        
        # Test after fitting
        ratings_df = pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]})
        hybrid.fit(ratings_df)
        
        info_after_fit = hybrid.get_algorithm_info()
        assert info_after_fit['mock1']['fitted'] == True
        assert info_after_fit['mock2']['fitted'] == True


class TestHybridRecommenderIntegration:
    """Test HybridRecommender with real algorithms."""
    
    def test_integration_with_real_algorithms(self):
        """Test hybrid system with actual recommendation algorithms."""
        # Create sample data
        ratings_df = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [101, 102, 103, 101, 102, 104, 102, 103, 104],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        # Create real algorithms
        global_avg = GlobalAverageRecommender()
        user_avg = UserAverageRecommender()
        
        algorithms = {
            'global': global_avg,
            'user': user_avg
        }
        weights = {
            'global': 0.3,
            'user': 0.7
        }
        
        hybrid = HybridRecommender(algorithms, weights)
        hybrid.fit(ratings_df)
        
        # Test prediction
        prediction = hybrid.predict(1, 104)  # User 1, unseen movie 104
        
        # Verify it's a reasonable prediction
        assert 1.0 <= prediction <= 5.0
        assert isinstance(prediction, (int, float))
    
    def test_integration_with_matrix_factorization(self):
        """Test hybrid system with matrix factorization."""
        # Create larger sample data for MF
        np.random.seed(42)
        user_ids = np.repeat(range(1, 11), 10)  # 10 users
        movie_ids = np.tile(range(101, 111), 10)  # 10 movies
        ratings = np.random.uniform(1, 5, 100)
        
        ratings_df = pd.DataFrame({
            'userId': user_ids,
            'movieId': movie_ids,
            'rating': ratings
        })
        
        # Create algorithms
        global_avg = GlobalAverageRecommender()
        mf = BasicMatrixFactorization(k=5, epochs=10, learning_rate=0.01)
        
        algorithms = {
            'baseline': global_avg,
            'mf': mf
        }
        
        hybrid = HybridRecommender(algorithms)  # Equal weights
        hybrid.fit(ratings_df)
        
        # Test prediction
        prediction = hybrid.predict(1, 101)
        
        assert 1.0 <= prediction <= 5.0
        
        # Test info
        info = hybrid.get_algorithm_info()
        assert info['baseline']['class'] == 'GlobalAverageRecommender'
        assert info['mf']['class'] == 'BasicMatrixFactorization'
        assert all(alg_info['fitted'] for alg_info in info.values())
    
    def test_different_weight_distributions(self):
        """Test various weight distributions."""
        # Create test data where global average != user averages
        ratings_df = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [101, 102, 103, 101, 102, 104, 105, 106, 107],
            'rating': [5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]  # User 1: avg=5.0, User 2: avg=2.0, Global avg=3.33
        })
        
        global_avg = GlobalAverageRecommender()
        user_avg = UserAverageRecommender()
        
        algorithms = {'global': global_avg, 'user': user_avg}
        
        # Test various weight combinations
        weight_sets = [
            {'global': 1.0, 'user': 0.0},    # Only global
            {'global': 0.0, 'user': 1.0},    # Only user
            {'global': 0.5, 'user': 0.5},    # Equal
            {'global': 0.9, 'user': 0.1},    # Heavily global
            {'global': 0.1, 'user': 0.9},    # Heavily user
        ]
        
        predictions = []
        for weights in weight_sets:
            hybrid = HybridRecommender(algorithms, weights)
            hybrid.fit(ratings_df)
            pred = hybrid.predict(1, 999)  # Unseen movie
            predictions.append(pred)
            assert 1.0 <= pred <= 5.0
        
        # Predictions should vary with different weights
        assert len(set(predictions)) > 1, "Different weights should produce different predictions"


class TestHybridRecommenderEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_rating_dataset(self):
        """Test with minimal dataset."""
        ratings_df = pd.DataFrame({
            'userId': [1],
            'movieId': [101],
            'rating': [4.0]
        })
        
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(5.0)
        
        hybrid = HybridRecommender({'alg1': alg1, 'alg2': alg2})
        hybrid.fit(ratings_df)
        
        prediction = hybrid.predict(1, 102)
        expected = 0.5 * 3.0 + 0.5 * 5.0  # Equal weights
        assert abs(prediction - expected) < 1e-10
    
    def test_weight_precision(self):
        """Test weight validation with floating point precision."""
        alg1 = MockRecommender(3.0)
        alg2 = MockRecommender(4.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        
        # Test weights that sum to 1.0 within tolerance
        weights_close = {'alg1': 0.6, 'alg2': 0.4000001}  # Slightly over 1.0
        hybrid = HybridRecommender(algorithms, weights_close)  # Should work
        
        # Test weights that are clearly wrong
        weights_wrong = {'alg1': 0.6, 'alg2': 0.5}  # Sum = 1.1
        with pytest.raises(ValueError):
            HybridRecommender(algorithms, weights_wrong)
    
    def test_very_small_weights(self):
        """Test with very small weights."""
        alg1 = MockRecommender(1.0)
        alg2 = MockRecommender(5.0)
        
        algorithms = {'alg1': alg1, 'alg2': alg2}
        weights = {'alg1': 0.001, 'alg2': 0.999}
        
        hybrid = HybridRecommender(algorithms, weights)
        hybrid.fit(pd.DataFrame({'userId': [1], 'movieId': [101], 'rating': [3.0]}))
        
        prediction = hybrid.predict(1, 102)
        # Should be very close to alg2's prediction (5.0)
        assert abs(prediction - 5.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__]) 
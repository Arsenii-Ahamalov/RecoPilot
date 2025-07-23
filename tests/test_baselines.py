import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.baselines import (
    GlobalAverageRecommender,
    UserAverageRecommender, 
    MovieAverageRecommender,
    BiasRecommender
)


@pytest.fixture
def sample_ratings():
    """Create sample ratings data for testing."""
    data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movieId': [101, 102, 103, 101, 102, 104, 102, 103, 104],
        'rating': [5.0, 3.0, 4.0, 2.0, 4.0, 3.0, 1.0, 5.0, 4.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def expected_stats(sample_ratings):
    """Calculate expected statistics from sample data."""
    return {
        'global_mean': sample_ratings['rating'].mean(),  # 3.444...
        'user_means': {
            1: sample_ratings[sample_ratings['userId'] == 1]['rating'].mean(),  # 4.0
            2: sample_ratings[sample_ratings['userId'] == 2]['rating'].mean(),  # 3.0
            3: sample_ratings[sample_ratings['userId'] == 3]['rating'].mean(),  # 3.333...
        },
        'movie_means': {
            101: sample_ratings[sample_ratings['movieId'] == 101]['rating'].mean(),  # 3.5
            102: sample_ratings[sample_ratings['movieId'] == 102]['rating'].mean(),  # 2.667...
            103: sample_ratings[sample_ratings['movieId'] == 103]['rating'].mean(),  # 4.5
            104: sample_ratings[sample_ratings['movieId'] == 104]['rating'].mean(),  # 3.5
        }
    }


class TestGlobalAverageRecommender:
    
    def test_initialization(self):
        """Test proper initialization."""
        recommender = GlobalAverageRecommender()
        assert not recommender.is_fitted
        assert recommender.global_average is None
        assert recommender.data is None
    
    def test_fit(self, sample_ratings, expected_stats):
        """Test fitting the model."""
        recommender = GlobalAverageRecommender()
        result = recommender.fit(sample_ratings)
        
        # Check return value
        assert result is recommender  # Should return self
        
        # Check fitted state
        assert recommender.is_fitted
        assert recommender.data is not None
        
        # Check calculated average
        assert abs(recommender.global_average - expected_stats['global_mean']) < 1e-10
    
    def test_predict_before_fit(self):
        """Test prediction before fitting raises error."""
        recommender = GlobalAverageRecommender()
        with pytest.raises(ValueError, match="Model hasn't been fitted yet"):
            recommender.predict(1, 101)
    
    def test_predict_after_fit(self, sample_ratings, expected_stats):
        """Test prediction after fitting."""
        recommender = GlobalAverageRecommender()
        recommender.fit(sample_ratings)
        
        # Should return global average for any user/movie pair
        prediction = recommender.predict(1, 101)
        assert abs(prediction - expected_stats['global_mean']) < 1e-10
        
        # Should be same for different user/movie pairs
        assert recommender.predict(1, 101) == recommender.predict(999, 999)
    
    def test_inherited_methods(self, sample_ratings):
        """Test that inherited methods work correctly."""
        recommender = GlobalAverageRecommender()
        recommender.fit(sample_ratings)
        
        # Test predict_for_user
        predictions = recommender.predict_for_user(1, [101, 102])
        assert len(predictions) == 2
        assert all(isinstance(pred, tuple) for pred in predictions)
        
        # Test recommend
        recommendations = recommender.recommend(1, k=2)
        assert len(recommendations) <= 2
        assert all(isinstance(rec, tuple) for rec in recommendations)


class TestUserAverageRecommender:
    
    def test_initialization(self):
        """Test proper initialization."""
        recommender = UserAverageRecommender()
        assert not recommender.is_fitted
        assert recommender.users_average is None
    
    def test_fit(self, sample_ratings, expected_stats):
        """Test fitting the model."""
        recommender = UserAverageRecommender()
        result = recommender.fit(sample_ratings)
        
        assert result is recommender
        assert recommender.is_fitted
        
        # Check user averages are calculated correctly
        for user_id, expected_avg in expected_stats['user_means'].items():
            assert abs(recommender.users_average[user_id] - expected_avg) < 1e-10
    
    def test_predict_known_user(self, sample_ratings, expected_stats):
        """Test prediction for known users."""
        recommender = UserAverageRecommender()
        recommender.fit(sample_ratings)
        
        # Should return user's average regardless of movie
        for user_id, expected_avg in expected_stats['user_means'].items():
            prediction = recommender.predict(user_id, 999)  # Any movie
            assert abs(prediction - expected_avg) < 1e-10
    
    def test_predict_unknown_user(self, sample_ratings):
        """Test prediction for unknown users raises KeyError."""
        recommender = UserAverageRecommender()
        recommender.fit(sample_ratings)
        
        with pytest.raises(KeyError):
            recommender.predict(999, 101)  # Unknown user


class TestMovieAverageRecommender:
    
    def test_initialization(self):
        """Test proper initialization."""
        recommender = MovieAverageRecommender()
        assert not recommender.is_fitted
        assert recommender.movie_average is None
    
    def test_fit(self, sample_ratings, expected_stats):
        """Test fitting the model."""
        recommender = MovieAverageRecommender()
        result = recommender.fit(sample_ratings)
        
        assert result is recommender
        assert recommender.is_fitted
        
        # Check movie averages are calculated correctly
        for movie_id, expected_avg in expected_stats['movie_means'].items():
            assert abs(recommender.movie_average[movie_id] - expected_avg) < 1e-10
    
    def test_predict_known_movie(self, sample_ratings, expected_stats):
        """Test prediction for known movies."""
        recommender = MovieAverageRecommender()
        recommender.fit(sample_ratings)
        
        # Should return movie's average regardless of user
        for movie_id, expected_avg in expected_stats['movie_means'].items():
            prediction = recommender.predict(999, movie_id)  # Any user
            assert abs(prediction - expected_avg) < 1e-10
    
    def test_predict_unknown_movie(self, sample_ratings):
        """Test prediction for unknown movies raises KeyError."""
        recommender = MovieAverageRecommender()
        recommender.fit(sample_ratings)
        
        with pytest.raises(KeyError):
            recommender.predict(1, 999)  # Unknown movie


class TestBiasRecommender:
    
    def test_initialization(self):
        """Test proper initialization."""
        recommender = BiasRecommender()
        assert not recommender.is_fitted
        assert recommender.global_average is None
        assert recommender.users_average is None
        assert recommender.movies_average is None
    
    def test_fit(self, sample_ratings, expected_stats):
        """Test fitting the model."""
        recommender = BiasRecommender()
        result = recommender.fit(sample_ratings)
        
        assert result is recommender
        assert recommender.is_fitted
        
        # Check all averages are calculated
        assert abs(recommender.global_average - expected_stats['global_mean']) < 1e-10
        
        for user_id, expected_avg in expected_stats['user_means'].items():
            assert abs(recommender.users_average[user_id] - expected_avg) < 1e-10
            
        for movie_id, expected_avg in expected_stats['movie_means'].items():
            assert abs(recommender.movies_average[movie_id] - expected_avg) < 1e-10
    
    def test_predict_bias_calculation(self, sample_ratings, expected_stats):
        """Test that bias calculation is correct."""
        recommender = BiasRecommender()
        recommender.fit(sample_ratings)
        
        # Test specific prediction
        user_id, movie_id = 1, 101
        prediction = recommender.predict(user_id, movie_id)
        
        # Calculate expected prediction manually
        global_mean = expected_stats['global_mean']
        user_bias = expected_stats['user_means'][user_id] - global_mean
        movie_bias = expected_stats['movie_means'][movie_id] - global_mean
        expected_prediction = global_mean + user_bias + movie_bias
        
        # Note: Your clamping function has wrong order - this test will help catch that
        assert abs(prediction - expected_prediction) < 1e-10
    
    def test_predict_clamping(self, sample_ratings):
        """Test that predictions are clamped to valid range."""
        recommender = BiasRecommender()
        recommender.fit(sample_ratings)
        
        # Test multiple predictions
        for user_id in [1, 2, 3]:
            for movie_id in [101, 102, 103, 104]:
                prediction = recommender.predict(user_id, movie_id)
                assert 1.0 <= prediction <= 5.0, f"Prediction {prediction} out of range for user {user_id}, movie {movie_id}"
    
    def test_predict_unknown_user_movie(self, sample_ratings):
        """Test prediction for unknown users/movies raises KeyError."""
        recommender = BiasRecommender()
        recommender.fit(sample_ratings)
        
        with pytest.raises(KeyError):
            recommender.predict(999, 101)  # Unknown user
            
        with pytest.raises(KeyError):
            recommender.predict(1, 999)  # Unknown movie


class TestIntegration:
    """Test integration between all baseline algorithms."""
    
    def test_all_algorithms_fit_and_predict(self, sample_ratings):
        """Test that all algorithms can fit and predict."""
        algorithms = [
            GlobalAverageRecommender(),
            UserAverageRecommender(), 
            MovieAverageRecommender(),
            BiasRecommender()
        ]
        
        for algo in algorithms:
            # Should fit without error
            algo.fit(sample_ratings)
            
            # Should predict for known user/movie pairs
            prediction = algo.predict(1, 101)
            assert isinstance(prediction, (int, float))
            assert 1.0 <= prediction <= 5.0
    
    def test_performance_comparison(self, sample_ratings):
        """Test that algorithms give different predictions (basic sanity check)."""
        algorithms = [
            GlobalAverageRecommender(),
            UserAverageRecommender(), 
            MovieAverageRecommender(),
            BiasRecommender()
        ]
        
        for algo in algorithms:
            algo.fit(sample_ratings)
        
        # Get predictions from all algorithms
        predictions = [algo.predict(1, 101) for algo in algorithms]
        
        # Not all predictions should be exactly the same 
        # (unless data is very specific)
        assert len(set(predictions)) >= 2, "Algorithms should give different predictions"


if __name__ == "__main__":
    pytest.main([__file__]) 
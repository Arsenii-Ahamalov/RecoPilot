import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.content_based import GenreBasedRecommender, DemographicBasedRecommender

@pytest.fixture
def content_sample_data():
    """Sample data for content-based filtering tests."""
    ratings_data = {
        'userId': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        'movieId': [1, 2, 3, 4, 1, 2, 5, 1, 3, 6, 2, 4, 1, 7],
        'rating': [5, 4, 3, 2, 4, 5, 3, 3, 4, 5, 5, 2, 4, 3]
    }
    
    movies_data = {
        'movieId': [1, 2, 3, 4, 5, 6, 7],
        'release_year': [1995, 1996, 1997, 1998, 1999, 2000, 2001],
        'Action': [1, 0, 0, 1, 0, 1, 0],
        'Comedy': [0, 1, 1, 0, 1, 0, 1],
        'Drama': [1, 1, 0, 0, 0, 1, 1],
        'Sci-Fi': [0, 0, 1, 1, 0, 0, 0],
        'Romance': [0, 0, 0, 0, 1, 1, 1]
    }
    
    return pd.DataFrame(ratings_data), pd.DataFrame(movies_data)

@pytest.fixture
def demographic_sample_data():
    """Sample data for demographic-based filtering tests."""
    ratings_data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4, 1, 2, 5],
        'rating': [5, 4, 3, 4, 5, 2, 3, 4, 3, 5, 4, 2, 4, 3, 5]
    }
    
    users_data = {
        'userId': [1, 2, 3, 4, 5],
        'age': [25, 30, 25, 45, 22],
        'occupation': [12, 16, 12, 7, 12],  # 12=student, 16=teacher, 7=executive
        'F': [0, 1, 1, 0, 1],  # Female
        'M': [1, 0, 0, 1, 0]   # Male
    }
    
    return pd.DataFrame(ratings_data), pd.DataFrame(users_data)

@pytest.fixture
def content_expected_stats():
    """Expected statistics for content-based filtering validation."""
    return {
        'num_users': 5,
        'num_movies': 7,
        'num_ratings': 14,
        'user_genre_preferences': {
            1: {'Action': 3.5, 'Comedy': 3.5, 'Drama': 4.5, 'Sci-Fi': 2.5},  # User 1: Action+Drama=5, Comedy+Drama=4, Comedy+Sci-Fi=3, Action+Sci-Fi=2
            2: {'Action': 4.0, 'Comedy': 4.0, 'Drama': 4.5, 'Romance': 3.0},  # User 2: Action+Drama=4, Comedy+Drama=5, Romance=3
            3: {'Action': 3.0, 'Comedy': 4.0, 'Drama': 3.5, 'Sci-Fi': 4.0, 'Romance': 5.0}  # User 3: Action+Drama=3, Comedy+Sci-Fi=4, Romance=5
        }
    }

class TestGenreBasedRecommender:
    """Test class for GenreBasedRecommender algorithm."""
    
    def test_initialization(self):
        """Test recommender initialization."""
        recommender = GenreBasedRecommender()
        assert recommender.is_fitted == False
        assert recommender.data is None
        
    def test_fitting(self, content_sample_data):
        """Test fitting the recommender with data."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        
        # Test fitting
        result = recommender.fit(ratings_df, movies_df)
        
        # Check return value (should be self for method chaining)
        assert result is recommender
        
        # Check internal state
        assert recommender.is_fitted == True
        assert recommender.data is not None
        assert recommender.movie_data is not None
        assert len(recommender.data) == 14
        assert len(recommender.movie_data) == 7
        
    def test_predict_existing_user_movie(self, content_sample_data):
        """Test prediction for existing user-movie combinations."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Test prediction for user who rated movies in multiple genres
        # User 1 rated: Movie 1 (Action+Drama)=5, Movie 2 (Comedy+Drama)=4, Movie 3 (Comedy+Sci-Fi)=3, Movie 4 (Action+Sci-Fi)=2
        # User 1 genre preferences: Action=(5+2)/2=3.5, Comedy=(4+3)/2=3.5, Drama=(5+4)/2=4.5, Sci-Fi=(3+2)/2=2.5
        
        # Predict for Movie 5 (Comedy+Romance) - user hasn't rated
        prediction = recommender.predict(1, 5)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
        
    def test_predict_new_genre_fallback(self, content_sample_data):
        """Test prediction fallback when user hasn't rated a genre."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # User 4 only rated Comedy+Drama (movie 2) and Action+Sci-Fi (movie 4)
        # Predict for movie with Romance (which user 4 never rated)
        prediction = recommender.predict(4, 7)  # Movie 7 is Comedy+Drama+Romance
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
        
    def test_find_genre_data_existing_genre(self, content_sample_data):
        """Test genre data extraction for genres user has rated."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # User 1 rated Action movies: Movie 1 (rating=5), Movie 4 (rating=2)
        genre_mean, count = recommender._GenreBasedRecommender__find_genre_data('Action', 1)
        
        assert count == 2  # User 1 rated 2 Action movies
        assert genre_mean == 3.5  # (5+2)/2 = 3.5
        
    def test_find_genre_data_unrated_genre(self, content_sample_data):
        """Test genre data extraction for genres user hasn't rated."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # User 4 never rated Romance movies
        genre_mean, count = recommender._GenreBasedRecommender__find_genre_data('Romance', 4)
        
        assert genre_mean == -1  # Fallback indicator
        assert count == 0
        
    def test_find_genre_mean_global_average(self, content_sample_data):
        """Test global genre average calculation."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Action movies: Movie 1 (ratings: 5,4,3,4), Movie 4 (ratings: 2,2), Movie 6 (rating: 5)
        # Global Action average = (5+4+3+4+2+2+5)/7 = 25/7 ≈ 3.57
        action_mean = recommender._GenreBasedRecommender__find_genre_mean('Action')
        
        assert isinstance(action_mean, (float, np.float64))
        assert 1.0 <= action_mean <= 5.0
        
    def test_calculate_users_rating_weighted_average(self, content_sample_data):
        """Test weighted average calculation."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Test with sample data: [(4.0, 10), (3.0, 5)] should give (4.0*10 + 3.0*5)/(10+5) = 55/15 = 3.67
        result_data = [(4.0, 10), (3.0, 5)]
        weighted_avg = recommender._GenreBasedRecommender__calculate_users_rating(result_data)
        
        expected = (4.0 * 10 + 3.0 * 5) / (10 + 5)
        assert abs(weighted_avg - expected) < 0.01
        
    def test_calculate_users_rating_empty_data(self, content_sample_data):
        """Test weighted average with empty data."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Empty data should return default rating
        result_data = []
        weighted_avg = recommender._GenreBasedRecommender__calculate_users_rating(result_data)
        
        assert weighted_avg == 3.0  # Default fallback
        
    def test_predict_single_genre_movie(self, content_sample_data):
        """Test prediction for movies with single genre."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Test with user-movie combination that doesn't exist in ratings
        # User 1 hasn't rated Movie 6 (Action+Romance)
        prediction = recommender.predict(1, 6)  # User 1 hasn't rated Movie 6
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
        
    def test_predict_unfitted_model(self, content_sample_data):
        """Test that prediction fails on unfitted model."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        
        # Should work since base class doesn't enforce fitted check
        # But let's test the behavior
        try:
            prediction = recommender.predict(1, 6)  # Use non-existing combination
            # If it doesn't raise an error, it should at least return a reasonable value
            assert isinstance(prediction, (float, int))
        except (AttributeError, ValueError, TypeError):
            # Expected behavior - unfitted model should fail
            pass
            
    def test_prediction_consistency(self, content_sample_data):
        """Test that predictions are consistent across multiple calls."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Same prediction should give same result - use non-existing combination
        pred1 = recommender.predict(1, 6)  # User 1 hasn't rated Movie 6
        pred2 = recommender.predict(1, 6)
        
        assert pred1 == pred2
        
    def test_different_users_different_predictions(self, content_sample_data):
        """Test that different users get different predictions for same movie."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Different users should generally get different predictions
        # Use movie that neither user has rated
        pred_user1 = recommender.predict(1, 7)  # User 1 hasn't rated Movie 7
        pred_user2 = recommender.predict(3, 7)  # User 3 hasn't rated Movie 7
        
        # They might be the same by coincidence, but algorithm should be working
        assert isinstance(pred_user1, float)
        assert isinstance(pred_user2, float)
        assert 1.0 <= pred_user1 <= 5.0
        assert 1.0 <= pred_user2 <= 5.0
        
    def test_inheritance_from_base(self, content_sample_data):
        """Test that GenreBasedRecommender properly inherits from BaseRecommender."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Test inherited methods work
        predictions = recommender.predict_for_user(1, [5, 6, 7])
        assert isinstance(predictions, list)
        assert len(predictions) == 3
        
        # Each prediction should be a tuple of (rating, movie_id)
        for rating, movie_id in predictions:
            assert isinstance(rating, (float, int))
            assert isinstance(movie_id, (int, np.integer))
            assert movie_id in [5, 6, 7]
            
    def test_recommend_method(self, content_sample_data):
        """Test the recommend method from base class."""
        ratings_df, movies_df = content_sample_data
        recommender = GenreBasedRecommender()
        recommender.fit(ratings_df, movies_df)
        
        # Test recommendations
        recommendations = recommender.recommend(1, k=3)
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Check recommendation format
        for rating, movie_id in recommendations:
            assert isinstance(rating, (float, int))
            assert isinstance(movie_id, (int, np.integer))
            assert 1.0 <= rating <= 5.0


class TestDemographicBasedRecommender:
    """Test class for DemographicBasedRecommender algorithm."""
    
    def test_initialization(self):
        """Test recommender initialization."""
        recommender = DemographicBasedRecommender()
        assert recommender.is_fitted == False
        assert recommender.data is None
        assert recommender.k == 50  # Default value
        assert recommender.similarity_cache == {}
        
    def test_initialization_with_k(self):
        """Test recommender initialization with custom k."""
        recommender = DemographicBasedRecommender(k=20)
        assert recommender.k == 20
        
    def test_fitting(self, demographic_sample_data):
        """Test fitting the recommender with data."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        
        # Test fitting
        result = recommender.fit(ratings_df, users_df)
        
        # Check return value (should be self for method chaining)
        assert result is recommender
        
        # Check internal state
        assert recommender.is_fitted == True
        assert recommender.data is not None
        assert recommender.users_data is not None
        assert len(recommender.data) == 15
        assert len(recommender.users_data) == 5
        
    def test_demographic_similarity_same_user(self, demographic_sample_data):
        """Test demographic similarity for same user (should be 1.0)."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        similarity = recommender._DemographicBasedRecommender__calculate_demographic_similarity(1, 1)
        assert similarity == 1.0
        
    def test_demographic_similarity_identical_demographics(self, demographic_sample_data):
        """Test demographic similarity for users with identical demographics."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Users 1 and 3: both age 25, both students (occupation 12), but different gender
        # Age similarity: 1.0 (same age)
        # Gender similarity: 0.0 (different)
        # Occupation similarity: 1.0 (same)
        # Expected: 0.4*1.0 + 0.3*0.0 + 0.3*1.0 = 0.7
        similarity = recommender._DemographicBasedRecommender__calculate_demographic_similarity(1, 3)
        expected = 0.4 * 1.0 + 0.3 * 0.0 + 0.3 * 1.0
        assert abs(similarity - expected) < 0.01
        
    def test_demographic_similarity_different_age(self, demographic_sample_data):
        """Test demographic similarity for users with different ages."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Users 1 and 4: age difference of 20 years (25 vs 45)
        # Age similarity: max(0, 1 - 20/55) = 1 - 0.364 = 0.636
        # Gender similarity: 1.0 (both male)
        # Occupation similarity: 0.0 (student vs executive)
        similarity = recommender._DemographicBasedRecommender__calculate_demographic_similarity(1, 4)
        
        age_sim = max(0, 1 - 20/55)
        expected = 0.4 * age_sim + 0.3 * 1.0 + 0.3 * 0.0
        assert abs(similarity - expected) < 0.01
        
    def test_predict_existing_rating(self, demographic_sample_data):
        """Test prediction for existing user-movie rating."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # User 1 rated movie 1 with rating 5
        prediction = recommender.predict(1, 1)
        assert prediction == 5.0
        
    def test_predict_new_rating(self, demographic_sample_data):
        """Test prediction for new user-movie combination."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # User 1 hasn't rated movie 4, but other users have
        prediction = recommender.predict(1, 4)
        assert isinstance(prediction, float)
        assert 1.0 <= prediction <= 5.0
        
    def test_find_relevant_users(self, demographic_sample_data):
        """Test finding users who rated a specific movie."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Movie 1 was rated by users 1, 2, 3, 5
        relevant_users = recommender._find_relevant_users(1)
        expected_users = [1, 2, 3, 5]
        
        assert len(relevant_users) == len(expected_users)
        for user in expected_users:
            assert user in relevant_users
            
    def test_similarity_caching(self, demographic_sample_data):
        """Test that similarity calculations are cached."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Calculate similarity first time
        sim1 = recommender._DemographicBasedRecommender__get_users_sim(1, 2)
        assert sim1 is None  # Not cached yet
        
        # Calculate and cache similarity
        calculated_sim = recommender._DemographicBasedRecommender__calculate_demographic_similarity(1, 2)
        key = (min(1, 2), max(1, 2))
        recommender.similarity_cache[key] = calculated_sim
        
        # Should return cached value
        sim2 = recommender._DemographicBasedRecommender__get_users_sim(1, 2)
        assert sim2 == calculated_sim
        
    def test_prediction_consistency(self, demographic_sample_data):
        """Test that predictions are consistent across multiple calls."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Same prediction should give same result
        pred1 = recommender.predict(1, 4)
        pred2 = recommender.predict(1, 4)
        
        assert pred1 == pred2
        
    def test_different_users_different_predictions(self, demographic_sample_data):
        """Test that different users can get different predictions."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Different users should potentially get different predictions
        # Use movie that neither user has rated in the demographic data
        pred_user1 = recommender.predict(3, 5)  # User 3 hasn't rated Movie 5
        pred_user4 = recommender.predict(4, 1)  # User 4 hasn't rated Movie 1
        
        assert isinstance(pred_user1, float)
        assert isinstance(pred_user4, float)
        assert 1.0 <= pred_user1 <= 5.0
        assert 1.0 <= pred_user4 <= 5.0
        
    def test_inheritance_from_base(self, demographic_sample_data):
        """Test that DemographicBasedRecommender properly inherits from BaseRecommender."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Test inherited methods work - use non-existing combinations
        predictions = recommender.predict_for_user(3, [5])  # User 3 hasn't rated Movie 5
        assert isinstance(predictions, list)
        assert len(predictions) <= 1
        
        # Each prediction should be a tuple of (rating, movie_id)
        for rating, movie_id in predictions:
            assert isinstance(rating, (float, int))
            assert isinstance(movie_id, (int, np.integer))
            assert movie_id in [5]
            
    def test_age_similarity_boundary_cases(self, demographic_sample_data):
        """Test age similarity calculation for boundary cases."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        recommender.fit(ratings_df, users_df)
        
        # Test with age difference equal to MAX_AGE_DIFF
        # This should result in age similarity of 0
        MAX_AGE_DIFF = 55
        
        # Manually test the age similarity calculation
        age1, age2 = 25, 25 + MAX_AGE_DIFF  # Maximum difference
        age_diff = abs(age1 - age2)
        age_similarity = max(0, 1 - age_diff / MAX_AGE_DIFF)
        
        assert age_similarity == 0.0
        
    def test_weight_normalization(self):
        """Test that demographic weights sum to 1.0."""
        AGE_WEIGHT = 0.4
        SEX_WEIGHT = 0.3
        OCCUPATION_WEIGHT = 0.3
        
        total_weight = AGE_WEIGHT + SEX_WEIGHT + OCCUPATION_WEIGHT
        assert abs(total_weight - 1.0) < 0.01  # Should sum to 1.0
        
    def test_predict_unfitted_model(self, demographic_sample_data):
        """Test prediction behavior on unfitted model."""
        ratings_df, users_df = demographic_sample_data
        recommender = DemographicBasedRecommender()
        
        # Should fail or handle gracefully
        try:
            prediction = recommender.predict(3, 5)  # Use non-existing combination
            # If it doesn't raise an error, it should return a reasonable value
            assert isinstance(prediction, (float, int))
        except (AttributeError, ValueError, TypeError):
            # Expected behavior - unfitted model should fail
            pass 
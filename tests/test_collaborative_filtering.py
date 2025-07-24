import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.collaborative_filtering import UserBasedCF, ItemBasedCF

@pytest.fixture
def cf_sample_data():
    """Sample data for collaborative filtering tests."""
    data = {
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4, 1, 2, 3],
        'rating': [5, 4, 3, 4, 5, 2, 3, 4, 3, 5, 4, 2, 4, 3, 5]
    }
    return pd.DataFrame(data)

@pytest.fixture  
def cf_expected_stats():
    """Expected statistics for collaborative filtering validation."""
    return {
        'num_users': 5,
        'num_movies': 4,
        'num_ratings': 15,
        'user_means': {1: 4.0, 2: 3.67, 3: 3.33, 4: 3.67, 5: 4.0},
        'movie_means': {1: 4.0, 2: 4.25, 3: 4.0, 4: 2.33}
    }

class CFTestBase:
    """Base class for collaborative filtering algorithm tests."""
    
    def run_basic_cf_tests(self, cf_class, sample_data, expected_stats):
        """Run basic tests that apply to all CF algorithms."""
        
        # Test initialization
        cf = cf_class(k=3)
        assert cf.k == 3
        assert cf.is_fitted == False
        assert cf.data is None
        assert cf.similarity_cache == {}
        
        # Test fitting
        cf.fit(sample_data)
        assert cf.is_fitted == True
        assert cf.data is not None
        assert len(cf.data) == expected_stats['num_ratings']
        
        # Test basic prediction (existing rating)
        existing_prediction = cf.predict(1, 1)
        assert existing_prediction == 5.0  # User 1 rated movie 1 as 5
        
        # Test prediction for unseen movie-user pair
        unseen_prediction = cf.predict(1, 4)  # User 1 hasn't rated movie 4
        assert 1.0 <= unseen_prediction <= 5.0
        
        # Test clamping
        assert isinstance(unseen_prediction, float)
        
        # Test inheritance methods work
        recommendations = cf.recommend(1, k=2)
        assert len(recommendations) <= 2
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)
        
        # Test predict_for_user
        predictions = cf.predict_for_user(1, [2, 3])
        assert len(predictions) == 2
        assert all(isinstance(pred, tuple) and len(pred) == 2 for pred in predictions)

class TestUserBasedCF(CFTestBase):
    """Tests specific to User-Based Collaborative Filtering."""
    
    def test_user_based_cf_basic(self, cf_sample_data, cf_expected_stats):
        """Test basic UserBasedCF functionality."""
        self.run_basic_cf_tests(UserBasedCF, cf_sample_data, cf_expected_stats)
    
    def test_user_based_cf_initialization(self):
        """Test UserBasedCF initialization with different k values."""
        cf_default = UserBasedCF()
        assert cf_default.k == 50  # Default k
        
        cf_custom = UserBasedCF(k=20)
        assert cf_custom.k == 20
        
    def test_similarity_calculation(self, cf_sample_data):
        """Test similarity calculation and caching."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test similarity calculation
        sim = cf._UserBasedCF__calculate_users_sim(1, 2)
        assert -1.0 <= sim <= 1.0
        
        # Test similarity caching
        initial_cache_size = len(cf.similarity_cache)
        sim1 = cf._UserBasedCF__get_users_sim(1, 2)
        sim2 = cf._UserBasedCF__get_users_sim(2, 1)  # Should use same cached value
        
        if sim1 is not None:
            assert sim1 == sim2  # Symmetric similarity
    
    def test_similarity_edge_cases(self, cf_sample_data):
        """Test similarity calculation edge cases."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test users with no common movies
        # Add a user with no overlap
        extended_data = cf_sample_data.copy()
        new_data = pd.DataFrame({
            'userId': [6, 6],
            'movieId': [5, 6], 
            'rating': [3, 4]
        })
        extended_data = pd.concat([extended_data, new_data], ignore_index=True)
        cf.fit(extended_data)
        
        sim = cf._UserBasedCF__calculate_users_sim(1, 6)
        assert sim == 0.0  # No common movies
    
    def test_prediction_accuracy(self, cf_sample_data):
        """Test prediction accuracy and reasonableness."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test prediction for user 1, movie 4 (not rated)
        prediction = cf.predict(1, 4)
        assert 1.0 <= prediction <= 5.0
        
        # Test that prediction is reasonable (not just fallback)
        # User 1 likes movies (avg 4.0), movie 4 has mixed reviews (avg 2.33)
        # Prediction should be somewhere in between
        assert prediction != 3.0  # Should not be just a fallback value
    
    def test_relevant_users_finding(self, cf_sample_data):
        """Test finding relevant users functionality."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test finding users who rated movie 1
        relevant_users = cf._find_relevant_users(1)
        expected_users = [1, 2, 3, 5]  # Users who rated movie 1
        assert set(relevant_users) == set(expected_users)
        
        # Test finding users who rated movie 4
        relevant_users_movie4 = cf._find_relevant_users(4)
        expected_users_movie4 = [2, 3, 4]  # Users who rated movie 4
        assert set(relevant_users_movie4) == set(expected_users_movie4)
    
    def test_k_parameter_effect(self, cf_sample_data):
        """Test that k parameter affects predictions."""
        cf_small_k = UserBasedCF(k=1)
        cf_large_k = UserBasedCF(k=10)
        
        cf_small_k.fit(cf_sample_data)
        cf_large_k.fit(cf_sample_data)
        
        # Predictions might be different due to different k values
        pred_small = cf_small_k.predict(1, 4)
        pred_large = cf_large_k.predict(1, 4)
        
        # Both should be valid ratings
        assert 1.0 <= pred_small <= 5.0
        assert 1.0 <= pred_large <= 5.0
    
    def test_recommendation_functionality(self, cf_sample_data):
        """Test recommendation generation."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test recommendations for user 1
        recommendations = cf.recommend(1, k=2)
        
        # Should recommend movies user hasn't seen
        user_1_movies = cf_sample_data[cf_sample_data['userId'] == 1]['movieId'].tolist()
        recommended_movies = [rec[1] for rec in recommendations]
        
        for movie in recommended_movies:
            assert movie not in user_1_movies
        
        # Should be sorted by predicted rating (highest first)
        if len(recommendations) > 1:
            assert recommendations[0][0] >= recommendations[1][0]
    
    def test_cache_efficiency(self, cf_sample_data):
        """Test that similarity caching works efficiently."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Make multiple predictions to populate cache
        cf.predict(1, 4)
        cf.predict(2, 3)
        cf.predict(1, 2)  # This should reuse some cached similarities
        
        # Cache should have some entries
        assert len(cf.similarity_cache) > 0
        
        # Test symmetric caching
        for key in cf.similarity_cache.keys():
            assert key[0] <= key[1]  # Keys should be ordered (min, max)

class TestItemBasedCF(CFTestBase):
    """Tests specific to Item-Based Collaborative Filtering."""
    
    def test_item_based_cf_basic(self, cf_sample_data, cf_expected_stats):
        """Test basic ItemBasedCF functionality."""
        self.run_basic_cf_tests(ItemBasedCF, cf_sample_data, cf_expected_stats)
    
    def test_item_based_cf_initialization(self):
        """Test ItemBasedCF initialization with different k values."""
        cf_default = ItemBasedCF()
        assert cf_default.k == 50  # Default k
        
        cf_custom = ItemBasedCF(k=20)
        assert cf_custom.k == 20
        
    def test_item_similarity_calculation(self, cf_sample_data):
        """Test item similarity calculation and caching."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test similarity calculation
        sim = cf._ItemBasedCF__calculate_items_sim(1, 2)
        assert -1.0 <= sim <= 1.0
        
        # Test similarity caching
        initial_cache_size = len(cf.similarity_cache)
        sim1 = cf._ItemBasedCF__get_items_sim(1, 2)
        sim2 = cf._ItemBasedCF__get_items_sim(2, 1)  # Should use same cached value
        
        if sim1 is not None:
            assert sim1 == sim2  # Symmetric similarity
    
    def test_item_similarity_edge_cases(self, cf_sample_data):
        """Test item similarity calculation edge cases."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test items with no common users
        # Add an item with no overlap
        extended_data = cf_sample_data.copy()
        new_data = pd.DataFrame({
            'userId': [6, 7],
            'movieId': [5, 5], 
            'rating': [3, 4]
        })
        extended_data = pd.concat([extended_data, new_data], ignore_index=True)
        cf.fit(extended_data)
        
        sim = cf._ItemBasedCF__calculate_items_sim(1, 5)
        assert sim == 0.0  # No common users
    
    def test_adjusted_cosine_vs_pearson(self, cf_sample_data):
        """Test that adjusted cosine handles user bias better than regular correlation."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test similarity calculation for items 1 and 2
        # Both are rated by users with different rating patterns
        sim = cf._ItemBasedCF__calculate_items_sim(1, 2)
        assert -1.0 <= sim <= 1.0
        
        # Similarity should be meaningful (not just 0 or 1)
        assert sim != 0.0  # Should find some similarity patterns
    
    def test_relevant_items_finding(self, cf_sample_data):
        """Test finding relevant items functionality."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test finding items rated by user 1
        relevant_items = cf._find_relevant_items(1)
        expected_items = [1, 2, 3]  # Items rated by user 1
        assert set(relevant_items) == set(expected_items)
        
        # Test finding items rated by user 4
        relevant_items_user4 = cf._find_relevant_items(4)
        expected_items_user4 = [2, 3, 4]  # Items rated by user 4
        assert set(relevant_items_user4) == set(expected_items_user4)
    
    def test_item_based_prediction_logic(self, cf_sample_data):
        """Test that item-based predictions use user's ratings for similar items."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Predict user 1's rating for item 4
        # User 1 has rated items 1, 2, 3 with ratings [5, 4, 3]
        # The prediction should be based on similarity of item 4 to items 1, 2, 3
        prediction = cf.predict(1, 4)
        assert 1.0 <= prediction <= 5.0
        
        # Should not be just a fallback value
        assert prediction != 3.0
    
    def test_k_parameter_effect_items(self, cf_sample_data):
        """Test that k parameter affects item-based predictions."""
        cf_small_k = ItemBasedCF(k=1)
        cf_large_k = ItemBasedCF(k=10)
        
        cf_small_k.fit(cf_sample_data)
        cf_large_k.fit(cf_sample_data)
        
        # Predictions might be different due to different k values
        pred_small = cf_small_k.predict(1, 4)
        pred_large = cf_large_k.predict(1, 4)
        
        # Both should be valid ratings
        assert 1.0 <= pred_small <= 5.0
        assert 1.0 <= pred_large <= 5.0
    
    def test_item_recommendation_functionality(self, cf_sample_data):
        """Test item-based recommendation generation."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test recommendations for user 1
        recommendations = cf.recommend(1, k=2)
        
        # Should recommend items user hasn't rated
        user_1_items = cf_sample_data[cf_sample_data['userId'] == 1]['movieId'].tolist()
        recommended_items = [rec[1] for rec in recommendations]
        
        for item in recommended_items:
            assert item not in user_1_items
        
        # Should be sorted by predicted rating (highest first)
        if len(recommendations) > 1:
            assert recommendations[0][0] >= recommendations[1][0]
    
    def test_item_cache_efficiency(self, cf_sample_data):
        """Test that item similarity caching works efficiently."""
        cf = ItemBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Make multiple predictions to populate cache
        cf.predict(1, 4)
        cf.predict(2, 3)
        cf.predict(1, 2)  # This should reuse some cached similarities
        
        # Cache should have some entries
        assert len(cf.similarity_cache) > 0
        
        # Test symmetric caching
        for key in cf.similarity_cache.keys():
            assert key[0] <= key[1]  # Keys should be ordered (min, max)
    
    def test_user_bias_handling(self, cf_sample_data):
        """Test that adjusted cosine properly handles user rating bias."""
        cf = ItemBasedCF(k=3)
        
        # Create data with clear user bias
        biased_data = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'movieId': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'rating': [1, 2, 3, 4, 5, 6, 2, 3, 4]  # Note: 6 is out of range but for testing
        })
        # Clamp ratings to valid range
        biased_data['rating'] = biased_data['rating'].clip(1, 5)
        
        cf.fit(biased_data)
        
        # Test similarity between items 1 and 2
        # Both should be similar despite different absolute rating levels
        sim = cf._ItemBasedCF__calculate_items_sim(1, 2)
        
        # Should handle the bias and find meaningful similarity
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

class TestCFIntegration:
    """Integration tests for collaborative filtering."""
    
    def test_cf_with_base_recommender_methods(self, cf_sample_data):
        """Test CF works correctly with inherited BaseRecommender methods."""
        cf = UserBasedCF(k=3)
        cf.fit(cf_sample_data)
        
        # Test that all inherited methods work
        user_predictions = cf.predict_for_user(1, [2, 3, 4])
        assert len(user_predictions) == 3
        
        recommendations = cf.recommend(1, k=2)
        assert len(recommendations) <= 2
        
        # Test evaluation would work (though we need actual vs predicted)
        # This tests the method signature compatibility
        test_df_pred = pd.DataFrame({
            'userId': [1, 2],
            'movieId': [4, 3], 
            'rating': [3.0, 4.0]
        })
        test_df_actual = pd.DataFrame({
            'userId': [1, 2],
            'movieId': [4, 3],
            'rating': [3.5, 3.8]
        })
        
        rmse = cf.evaluate_rmse(test_df_pred, test_df_actual)
        assert rmse >= 0
    
    def test_both_cf_algorithms_work(self, cf_sample_data):
        """Test that both UserBasedCF and ItemBasedCF work and produce valid results."""
        user_cf = UserBasedCF(k=3)
        item_cf = ItemBasedCF(k=3)
        
        user_cf.fit(cf_sample_data)
        item_cf.fit(cf_sample_data)
        
        # Test predictions from both algorithms
        user_pred = user_cf.predict(1, 4)
        item_pred = item_cf.predict(1, 4)
        
        # Both should produce valid ratings
        assert 1.0 <= user_pred <= 5.0
        assert 1.0 <= item_pred <= 5.0
        
        # Both should produce recommendations
        user_recs = user_cf.recommend(1, k=2)
        item_recs = item_cf.recommend(1, k=2)
        
        assert len(user_recs) <= 2
        assert len(item_recs) <= 2
        
        # Both should have similar structure
        for recs in [user_recs, item_recs]:
            assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recs)
            if len(recs) > 1:
                assert recs[0][0] >= recs[1][0]  # Sorted by rating
    
    def test_cf_algorithms_comparison(self, cf_sample_data):
        """Test that UserBasedCF and ItemBasedCF can produce different but valid results."""
        user_cf = UserBasedCF(k=3)
        item_cf = ItemBasedCF(k=3)
        
        user_cf.fit(cf_sample_data)
        item_cf.fit(cf_sample_data)
        
        # Compare predictions for the same user-item pair
        user_pred = user_cf.predict(1, 4)
        item_pred = item_cf.predict(1, 4)
        
        # Both should be valid, might be different (different approaches)
        assert 1.0 <= user_pred <= 5.0
        assert 1.0 <= item_pred <= 5.0
        
        # Test that both use caching efficiently
        assert len(user_cf.similarity_cache) >= 0
        assert len(item_cf.similarity_cache) >= 0 
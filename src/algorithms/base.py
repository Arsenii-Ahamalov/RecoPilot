from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.

    This class provides a consistent interface for training, predicting, and 
    evaluating recommendation algorithms. All specific algorithms (baselines,
    collaborative filtering, matrix factorization, etc.) should inherit from
    this class.

    Attributes:
        is_fitted (bool): Whether the model has been trained
        data (DataFrame): Training data with columns ['userId', 'movieId', 'rating']
    """
    
    def __init__(self):
        """
        Initialize the base recommender.

        Sets up the initial state for the recommendation algorithm. All algorithms
        start in an unfitted state and must call fit() before making predictions.

        Attributes initialized:
            is_fitted (bool): False - indicates model needs training
            data (DataFrame): None - will store training data after fitting

        Example:
            recommender = SomeAlgorithm()  # Inherits from BaseRecommender
            print(recommender.is_fitted)   # False
            recommender.fit(train_data)    # Now ready to use
            print(recommender.is_fitted)   # True
        """
        self.is_fitted = False
        self.data = None

    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame) -> 'BaseRecommender':
        """
        Train the recommendation algorithm on rating data.

        This method should learn the model parameters from the training data.
        Each algorithm will implement this differently (e.g., computing averages,
        building similarity matrices, factorizing matrices).

        Args:
            ratings_df (DataFrame): Training data with columns ['userId', 'movieId', 'rating']

        Returns:
            self: The fitted recommender instance (for method chaining)
        """
        pass

    @abstractmethod  
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-movie pair.

        This is the core prediction method that each algorithm must implement
        with its own logic. Should return a predicted rating between 1.0 and 5.0.

        Each algorithm implements this differently:
        - Global Average: Returns same rating for all user-movie pairs
        - User Average: Returns user's personal average rating
        - Movie Average: Returns movie's average rating across all users
        - Collaborative Filtering: Uses similar users' ratings to predict
        - Matrix Factorization: Uses latent factors to predict

        Args:
            user_id (int): The user ID to predict for
            item_id (int): The movie ID to predict for

        Returns:
            float: Predicted rating for the user-movie pair

        Raises:
            ValueError: If model hasn't been fitted yet
        """
        pass

    def predict_for_user(self, user_id: int, item_ids: list) -> list:
        """
        Predict ratings for multiple movies for one user.

        This method calls predict() for each movie, creates (rating, movie_id) pairs,
        and returns them sorted by predicted rating (highest first).

        Args:
            user_id (int): The user ID to predict for
            item_ids (list): List of movie IDs to predict for

        Returns:
            list: List of (predicted_rating, movie_id) tuples, sorted by rating (descending)
        """
        result = []
        for item in item_ids:
            result.append((self.predict(user_id, item), item))
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list:
        """
        Generate top-K movie recommendations for a user.

        This method predicts ratings for all movies the user hasn't seen,
        sorts them by predicted rating, and returns the top K recommendations.

        Args:
            user_id (int): The user ID to recommend for
            k (int, optional): Number of recommendations to return. Default: 10
            exclude_seen (bool, optional): Whether to exclude movies user has rated. Default: True

        Returns:
            list: List of (predicted_rating, movie_id) tuples, sorted by rating (highest first)
        """
        if exclude_seen:
            users_seen_movies = self.data[self.data['userId'] == user_id]['movieId'].unique()
            users_unseen_movies = self.data[~self.data['movieId'].isin(users_seen_movies)]['movieId'].unique()
        else:
            users_unseen_movies = self.data['movieId'].unique()
        predictions = self.predict_for_user(user_id, users_unseen_movies)
        return predictions[:k]

    def evaluate_rmse(self, test_df_predictions: pd.DataFrame, test_df_actual: pd.DataFrame) -> float:
        """
        Calculate Root Mean Square Error on test data.

        This method evaluates the algorithm's prediction accuracy by comparing
        predicted ratings to actual ratings in the test set.

        Args:
            test_df_predictions (DataFrame): Test data with columns ['userId', 'movieId', 'rating']
            test_df_actual (DataFrame): Test data with columns ['userId', 'movieId', 'rating']
        Returns:
            float: RMSE score (lower is better)
  
        """
        return np.sqrt(np.mean((test_df_predictions['rating'] - test_df_actual['rating'])**2))

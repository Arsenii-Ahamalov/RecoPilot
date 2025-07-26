from .base import BaseRecommender
import pandas as pd
import numpy as np

class GlobalAverageRecommender(BaseRecommender):
    """
    Global Average Baseline Recommender.
    
    This simple baseline predicts the same rating (global average) for all user-item pairs.
    It serves as a basic benchmark that more sophisticated algorithms should outperform.
    
    The prediction formula is:
        prediction = global_average_rating
    
    Attributes:
        global_average (float): The mean rating across all users and items in the training data.
    
    Example:
        >>> recommender = GlobalAverageRecommender()
        >>> recommender.fit(ratings_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)
    """
    
    def __init__(self):
        """Initialize the Global Average Recommender."""
        super().__init__()
        self.global_average = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'GlobalAverageRecommender':
        """
        Train the model by calculating the global average rating.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            GlobalAverageRecommender: Self for method chaining
        """
        self.global_average = ratings_df['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using global average.
        
        Args:
            user_id (int): User ID (ignored - same prediction for all users)
            item_id (int): Item ID (ignored - same prediction for all items)
            
        Returns:
            float: The global average rating
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.global_average


class UserAverageRecommender(BaseRecommender):
    """
    User Average Baseline Recommender.
    
    This baseline predicts ratings based on each user's personal average rating.
    It captures user-specific rating tendencies (some users rate higher/lower on average).
    
    The prediction formula is:
        prediction = user_average_rating
    
    Attributes:
        users_average (pd.Series): Average rating for each user, indexed by userId.
    
    Example:
        >>> recommender = UserAverageRecommender()
        >>> recommender.fit(ratings_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)  # Returns user 1's average
    """
    
    def __init__(self):
        """Initialize the User Average Recommender."""
        super().__init__()
        self.users_average = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'UserAverageRecommender':
        """
        Train the model by calculating average rating for each user.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            UserAverageRecommender: Self for method chaining
        """
        self.users_average = ratings_df.groupby('userId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating based on user's personal average rating.
        
        Args:
            user_id (int): User ID to get average rating for
            item_id (int): Item ID (ignored - same prediction for all items by this user)
            
        Returns:
            float: The user's average rating
            
        Raises:
            ValueError: If model hasn't been fitted yet
            KeyError: If user_id not found in training data
        """
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.users_average[user_id]


class MovieAverageRecommender(BaseRecommender):
    """
    Movie Average Baseline Recommender.
    
    This baseline predicts ratings based on each movie's average rating across all users.
    It captures item-specific rating patterns (some movies are generally rated higher/lower).
    
    The prediction formula is:
        prediction = movie_average_rating
    
    Attributes:
        movie_average (pd.Series): Average rating for each movie, indexed by movieId.
    
    Example:
        >>> recommender = MovieAverageRecommender()
        >>> recommender.fit(ratings_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)  # Returns movie 101's average
    """
    
    def __init__(self):
        """Initialize the Movie Average Recommender."""
        super().__init__()
        self.movie_average = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'MovieAverageRecommender':
        """
        Train the model by calculating average rating for each movie.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            MovieAverageRecommender: Self for method chaining
        """
        self.movie_average = ratings_df.groupby('movieId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating based on movie's average rating.
        
        Args:
            user_id (int): User ID (ignored - same prediction for all users for this movie)
            item_id (int): Item ID to get average rating for
            
        Returns:
            float: The movie's average rating
            
        Raises:
            ValueError: If model hasn't been fitted yet
            KeyError: If item_id not found in training data
        """
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.movie_average[item_id]


class BiasRecommender(BaseRecommender):
    """
    Bias-Based Baseline Recommender.
    
    This baseline combines global, user, and movie biases to make predictions.
    It captures both user tendencies and movie characteristics relative to the global average.
    
    The prediction formula is:
        prediction = global_average + user_bias + movie_bias
        where:
        - user_bias = user_average - global_average
        - movie_bias = movie_average - global_average
    
    This is equivalent to: user_average + movie_average - global_average
    
    Attributes:
        global_average (float): Overall mean rating in the dataset
        users_average (pd.Series): Average rating for each user
        movies_average (pd.Series): Average rating for each movie
    
    Example:
        >>> recommender = BiasRecommender()
        >>> recommender.fit(ratings_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)
        # Returns: global_avg + (user1_avg - global_avg) + (movie101_avg - global_avg)
    """
    
    def __init__(self):
        """Initialize the Bias Recommender."""
        super().__init__()
        self.global_average = None
        self.users_average = None
        self.movies_average = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'BiasRecommender':
        """
        Train the model by calculating global, user, and movie averages.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            BiasRecommender: Self for method chaining
        """
        self.global_average = ratings_df['rating'].mean()
        self.users_average = ratings_df.groupby('userId')['rating'].mean()
        self.movies_average = ratings_df.groupby('movieId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using combined user and movie biases.
        
        Args:
            user_id (int): User ID to get bias for
            item_id (int): Item ID to get bias for
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
            KeyError: If user_id or item_id not found in training data
        """
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        
        user_bias = self.users_average[user_id] - self.global_average
        movie_bias = self.movies_average[item_id] - self.global_average
        prediction = self.global_average + user_bias + movie_bias
        
        return max(1.0, min(5.0, prediction))
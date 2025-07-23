from .base import BaseRecommender
import pandas as pd
import numpy as np
class GlobalAverageRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.global_average = None
    def fit(self, ratings_df: pd.DataFrame) -> 'GlobalAverageRecommender':
        self.global_average = ratings_df['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.global_average
class UserAverageRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.users_average = None
    def fit(self, ratings_df: pd.DataFrame) -> 'UserAverageRecommender':
        self.users_average = ratings_df.groupby('userId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.users_average[user_id]
class MovieAverageRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.movie_average = None
    def fit(self, ratings_df: pd.DataFrame) -> 'MovieAverageRecommender':
        self.movie_average = ratings_df.groupby('movieId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        return self.movie_average[item_id]
class BiasRecommender(BaseRecommender):
    def __init__(self):
        super().__init__()
        self.global_average = None
        self.users_average = None
        self.movies_average = None
    def fit(self, ratings_df: pd.DataFrame) -> 'BiasRecommender':
        self.global_average = ratings_df['rating'].mean()
        self.users_average = ratings_df.groupby('userId')['rating'].mean()
        self.movies_average = ratings_df.groupby('movieId')['rating'].mean()
        self.is_fitted = True
        self.data = ratings_df
        return self
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model hasn't been fitted yet")
        user_bias = self.users_average[user_id] - self.global_average
        movies_bias = self.movies_average[item_id] - self.global_average
        prediction = self.global_average + user_bias + movies_bias
        return max(1.0, min(5.0, prediction))
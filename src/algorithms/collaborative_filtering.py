from .base import BaseRecommender
import pandas as pd
import numpy as np
class UserBasedCF(BaseRecommender):
    def __init__(self, k : int = 50):
        super().__init__()
        self.k = k
        self.similarity_cache = {}
    def fit(self, ratings_df: pd.DataFrame) -> 'UserCFRecommender':
        self.data = ratings_df
        self.is_fitted = True
    def predict(self, user_id: int, item_id: int) -> float:
        existing_rating = self.data[
        (self.data['userId'] == user_id) & 
        (self.data['movieId'] == item_id)
        ]
    
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0] 

        relevant_users = self._find_relevant_users(item_id)
        similarities = []
        for other_user in relevant_users:
            users_sim = self.__get_users_sim(user_id,other_user)
            if users_sim is None:
                users_sim = self.__calculate_users_sim(user_id,other_user)
                key = (min(user_id, other_user), max(user_id, other_user))
                self.similarity_cache[key] = users_sim
                similarities.append((users_sim,other_user))
            else:
                similarities.append((users_sim,other_user))

        similarities.sort(reverse = True)
        if len(similarities) < self.k:
            result = self.__calculate_rating(similarities, item_id, user_id)
        else:
            result = self.__calculate_rating(similarities[:self.k], item_id, user_id)
        return result
    def _find_relevant_users(self, item_id: int) -> list:
        return self.data[self.data['movieId'] == item_id]['userId'].unique()
    def __get_users_sim(self,user1 : int , user2 : int) -> float:
        key = (min(user1, user2), max(user1, user2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        else:
            return None
    def __calculate_users_sim(self,user1 : int , user2 : int) -> float:
        user1_ratings = self.data[self.data['userId'] == user1]
        user2_ratings = self.data[self.data['userId'] == user2]
        
        common_movies = set(user1_ratings['movieId']).intersection(set(user2_ratings['movieId']))
        
        if len(common_movies) == 0:
            return 0.0
        
        user1_common = user1_ratings[user1_ratings['movieId'].isin(common_movies)].set_index('movieId')['rating']
        user2_common = user2_ratings[user2_ratings['movieId'].isin(common_movies)].set_index('movieId')['rating']
        
        mean1 = user1_common.mean()
        mean2 = user2_common.mean()
        
        numerator = sum((user1_common[movie] - mean1) * (user2_common[movie] - mean2) for movie in common_movies)
        
        sum1_sq = sum((user1_common[movie] - mean1) ** 2 for movie in common_movies)
        sum2_sq = sum((user2_common[movie] - mean2) ** 2 for movie in common_movies)
        
        denominator = (sum1_sq * sum2_sq) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    def __calculate_rating(self, similarities : list, item_id : int, user_id : int) -> float:
        
        target_user_mean = self.data[self.data['userId'] == user_id]['rating'].mean()
        
        numerator = 0
        denominator = 0
        
        for similarity, other_user in similarities:
            other_rating = self.data[
                (self.data['userId'] == other_user) & 
                (self.data['movieId'] == item_id)
            ]['rating'].iloc[0]
            
            other_user_mean = self.data[self.data['userId'] == other_user]['rating'].mean()
            
            numerator += similarity * (other_rating - other_user_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return target_user_mean
        
        prediction = target_user_mean + (numerator / denominator)
        return max(1.0, min(5.0, prediction))

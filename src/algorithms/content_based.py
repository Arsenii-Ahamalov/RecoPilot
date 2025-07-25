from .base import BaseRecommender
import pandas as pd
import numpy as np

class GenreBasedRecommender(BaseRecommender):
    def fit(self, ratings_df: pd.DataFrame, movie_df: pd.DataFrame) -> 'GenreBasedRecommender':
        self.data = ratings_df
        self.movie_data = movie_df
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
    
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0] 

        movie_info = self.movie_data[self.movie_data['movieId'] == item_id].iloc[0]
        genres = [col for col in self.movie_data.columns if movie_info[col] == 1 and col not in ['movieId','release_year']]
        result_data = []
        for genre in genres:
            genre_data = self.__find_genre_data(genre, user_id)
            if genre_data[0] == -1:
                genre_mean = self.__find_genre_mean(genre)
                result_data.append((genre_mean, 1))
            else:
                result_data.append(genre_data)
        return self.__calculate_users_rating(result_data)
    
    def __find_genre_data(self, genre: str, user_id: int) -> tuple:
        users_movies_id = self.data[self.data['userId'] == user_id]['movieId'].values
        users_movie = self.movie_data[self.movie_data['movieId'].isin(users_movies_id)]
        
        genre_movies_id = users_movie[users_movie[genre] == 1]['movieId'].values
        movies_count = len(genre_movies_id)
        
        if movies_count == 0:
            return (-1, 0)
        
        genre_mean = self.data[(self.data['userId'] == user_id) & 
                              (self.data['movieId'].isin(genre_movies_id))]['rating'].mean()
        
        return (genre_mean, movies_count)
    
    def __find_genre_mean(self, genre: str) -> float:
        genre_movies = self.movie_data[self.movie_data[genre] == 1]['movieId'].values
        global_genre_mean = self.data[self.data['movieId'].isin(genre_movies)]['rating'].mean()
        return global_genre_mean
    
    def __calculate_users_rating(self, result_data: list) -> float:
        total_weighted_score = 0
        total_weight = 0
        
        for rating, weight in result_data:
            total_weighted_score += rating * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 3.0
class DemographicBasedRecommender(BaseRecommender):
    def __init__(self, k: int = 50):
        super().__init__()
        self.k = k
        self.similarity_cache = {}
        
    def fit(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> 'DemographicBasedRecommender':
        self.data = ratings_df
        self.users_data = users_df
        self.is_fitted = True
        return self
        
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
            users_sim = self.__get_users_sim(user_id, other_user)
            if users_sim is None:
                users_sim = self.__calculate_demographic_similarity(user_id, other_user)
                key = (min(user_id, other_user), max(user_id, other_user))
                self.similarity_cache[key] = users_sim
                similarities.append((users_sim, other_user))
            else:
                similarities.append((users_sim, other_user))

        similarities.sort(reverse=True)
        if len(similarities) < self.k:
            result = self.__calculate_rating(similarities, item_id, user_id)
        else:
            result = self.__calculate_rating(similarities[:self.k], item_id, user_id)
        return result
        
    def _find_relevant_users(self, item_id: int) -> list:
        return self.data[self.data['movieId'] == item_id]['userId'].unique()
        
    def __get_users_sim(self, user1: int, user2: int) -> float:
        key = (min(user1, user2), max(user1, user2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        else:
            return None
            
    def __calculate_demographic_similarity(self, user1: int, user2: int) -> float:
        AGE_WEIGHT = 0.4
        SEX_WEIGHT = 0.3
        OCCUPATION_WEIGHT = 0.3
        MAX_AGE_DIFF = 55
        user1 = self.users_data[self.users_data['userId'] == user1].iloc[0]
        user2 = self.users_data[self.users_data['userId'] == user2].iloc[0]
        sex_similarity = 1 if user1['F'] == user2['F'] else 0
        occupation_similarity = 1 if user1['occupation'] == user2['occupation'] else 0
        age_diff = user1['age'] - user2['age']
        age_similarity = max(0, 1 - abs(age_diff) / MAX_AGE_DIFF)
        return AGE_WEIGHT * age_similarity + SEX_WEIGHT * sex_similarity + OCCUPATION_WEIGHT * occupation_similarity

        
    def __calculate_rating(self, similarities: list, item_id: int, user_id: int) -> float:
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
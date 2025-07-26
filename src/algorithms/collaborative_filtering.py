from .base import BaseRecommender
import pandas as pd
import numpy as np

class UserBasedCF(BaseRecommender):
    """
    User-Based Collaborative Filtering Recommender.
    
    This algorithm finds users with similar rating patterns and uses their ratings
    to predict ratings for the target user. It's based on the principle that users
    who agreed in the past will agree in the future.
    
    The prediction formula is:
        prediction = user_mean + Σ(similarity * (other_user_rating - other_user_mean)) / Σ|similarity|
    
    Algorithm steps:
    1. Find users who have rated the target item
    2. Calculate similarity between target user and these users (using Pearson correlation)
    3. Select k most similar users
    4. Predict rating based on weighted average of similar users' ratings
    
    Attributes:
        k (int): Number of most similar users to consider for prediction
        similarity_cache (dict): Cache of computed user similarities for efficiency
    
    Example:
        >>> cf = UserBasedCF(k=30)
        >>> cf.fit(ratings_df)
        >>> prediction = cf.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, k: int = 50):
        """
        Initialize User-Based Collaborative Filtering recommender.
        
        Args:
            k (int): Number of most similar users to consider. Higher k means
                    more users influence the prediction but may include less similar users.
                    Default: 50
        """
        super().__init__()
        self.k = k
        self.similarity_cache = {}
    
    def fit(self, ratings_df: pd.DataFrame) -> 'UserBasedCF':
        """
        Train the model by storing the ratings data.
        
        For collaborative filtering, training simply means storing the data.
        Similarities are computed on-demand during prediction.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            UserBasedCF: Self for method chaining
        """
        self.data = ratings_df
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using collaborative filtering.
        
        Algorithm:
        1. Check if user already rated this item (return actual rating)
        2. Find users who have rated this item
        3. Calculate similarities between target user and these users
        4. Select k most similar users
        5. Predict rating using weighted average of similar users' ratings
        
        Args:
            user_id (int): Target user ID
            item_id (int): Target item ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Check if user already rated this item
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
        
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0] 

        # Find users who have rated this item
        relevant_users = self._find_relevant_users(item_id)
        similarities = []
        
        # Calculate similarities with relevant users
        for other_user in relevant_users:
            users_sim = self.__get_users_sim(user_id, other_user)
            if users_sim is None:
                users_sim = self.__calculate_users_sim(user_id, other_user)
                key = (min(user_id, other_user), max(user_id, other_user))
                self.similarity_cache[key] = users_sim
            similarities.append((users_sim, other_user))

        # Sort by similarity (highest first) and select top k
        similarities.sort(reverse=True)
        if len(similarities) < self.k:
            result = self.__calculate_rating(similarities, item_id, user_id)
        else:
            result = self.__calculate_rating(similarities[:self.k], item_id, user_id)
        return result
    
    def _find_relevant_users(self, item_id: int) -> list:
        """
        Find all users who have rated the given item.
        
        Args:
            item_id (int): Item ID to find users for
            
        Returns:
            list: List of user IDs who have rated this item
        """
        return self.data[self.data['movieId'] == item_id]['userId'].unique()
    
    def __get_users_sim(self, user1: int, user2: int) -> float:
        """
        Get cached similarity between two users if available.
        
        Args:
            user1 (int): First user ID
            user2 (int): Second user ID
            
        Returns:
            float or None: Cached similarity if available, None otherwise
        """
        key = (min(user1, user2), max(user1, user2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        else:
            return None
    
    def __calculate_users_sim(self, user1: int, user2: int) -> float:
        """
        Calculate Pearson correlation coefficient between two users.
        
        The Pearson correlation measures linear correlation between two users'
        rating patterns, considering their individual rating means.
        
        Formula:
            r = Σ((x_i - x̄)(y_i - ȳ)) / √(Σ(x_i - x̄)² * Σ(y_i - ȳ)²)
        
        Args:
            user1 (int): First user ID
            user2 (int): Second user ID
            
        Returns:
            float: Pearson correlation coefficient between -1 and 1
                  1 = perfect positive correlation
                  0 = no correlation
                  -1 = perfect negative correlation
        """
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
    
    def __calculate_rating(self, similarities: list, item_id: int, user_id: int) -> float:
        """
        Calculate predicted rating using weighted average of similar users' ratings.
        
        Formula:
            prediction = user_mean + Σ(similarity * (other_rating - other_mean)) / Σ|similarity|
        
        Args:
            similarities (list): List of (similarity, user_id) tuples, sorted by similarity
            item_id (int): Target item ID
            user_id (int): Target user ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
        """
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


class ItemBasedCF(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender.
    
    This algorithm finds items with similar rating patterns and uses their ratings
    from the target user to predict ratings. It's based on the principle that
    items that were rated similarly in the past will be rated similarly in the future.
    
    The prediction formula is:
        prediction = item_mean + Σ(similarity * (user_rating_for_item - item_mean)) / Σ|similarity|
    
    Algorithm steps:
    1. Find items that the target user has rated
    2. Calculate similarity between target item and these items (using adjusted cosine)
    3. Select k most similar items
    4. Predict rating based on weighted average of user's ratings for similar items
    
    Attributes:
        k (int): Number of most similar items to consider for prediction
        similarity_cache (dict): Cache of computed item similarities for efficiency
    
    Example:
        >>> icf = ItemBasedCF(k=30)
        >>> icf.fit(ratings_df)
        >>> prediction = icf.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, k: int = 50):
        """
        Initialize Item-Based Collaborative Filtering recommender.
        
        Args:
            k (int): Number of most similar items to consider. Higher k means
                    more items influence the prediction but may include less similar items.
                    Default: 50
        """
        super().__init__()
        self.k = k
        self.similarity_cache = {}
    
    def fit(self, ratings_df: pd.DataFrame) -> 'ItemBasedCF':
        """
        Train the model by storing the ratings data.
        
        For collaborative filtering, training simply means storing the data.
        Similarities are computed on-demand during prediction.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            ItemBasedCF: Self for method chaining
        """
        self.data = ratings_df
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using item-based collaborative filtering.
        
        Algorithm:
        1. Check if user already rated this item (return actual rating)
        2. Find items that this user has rated
        3. Calculate similarities between target item and these items
        4. Select k most similar items
        5. Predict rating using weighted average of user's ratings for similar items
        
        Args:
            user_id (int): Target user ID
            item_id (int): Target item ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Check if user already rated this item
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
        
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0]
        
        # Find items that this user has rated
        relevant_items = self._find_relevant_items(user_id)
        similarities = []
        
        # Calculate similarities with relevant items
        for other_item in relevant_items:
            item_sim = self.__get_items_sim(item_id, other_item)
            if item_sim is None:
                item_sim = self.__calculate_items_sim(item_id, other_item)
                key = (min(item_id, other_item), max(item_id, other_item))
                self.similarity_cache[key] = item_sim
            similarities.append((item_sim, other_item))
        
        # Sort by similarity (highest first) and select top k
        similarities.sort(reverse=True)
        if len(similarities) < self.k:
            result = self.__calculate_rating(similarities, item_id, user_id)
        else:
            result = self.__calculate_rating(similarities[:self.k], item_id, user_id)
        return result
    
    def _find_relevant_items(self, user_id: int) -> list:
        """
        Find all items that the given user has rated.
        
        Args:
            user_id (int): User ID to find rated items for
            
        Returns:
            list: List of item IDs that this user has rated
        """
        return self.data[self.data['userId'] == user_id]['movieId'].unique()
    
    def __get_items_sim(self, item1: int, item2: int) -> float:
        """
        Get cached similarity between two items if available.
        
        Args:
            item1 (int): First item ID
            item2 (int): Second item ID
            
        Returns:
            float or None: Cached similarity if available, None otherwise
        """
        key = (min(item1, item2), max(item1, item2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        else:
            return None
    
    def __calculate_items_sim(self, item1: int, item2: int) -> float:
        """
        Calculate adjusted cosine similarity between two items.
        
        Adjusted cosine similarity accounts for different users' rating scales
        by subtracting each user's average rating before computing similarity.
        
        Formula:
            sim = Σ((r_u,i - r̄_u)(r_u,j - r̄_u)) / √(Σ(r_u,i - r̄_u)² * Σ(r_u,j - r̄_u)²)
        
        Args:
            item1 (int): First item ID
            item2 (int): Second item ID
            
        Returns:
            float: Adjusted cosine similarity between -1 and 1
        """
        item1_ratings = self.data[self.data['movieId'] == item1]
        item2_ratings = self.data[self.data['movieId'] == item2]
        
        # Find users who rated both items
        common_users = set(item1_ratings['userId']).intersection(set(item2_ratings['userId']))
        
        if len(common_users) == 0:
            return 0.0
        
        numerator = 0
        sum1_sq = 0
        sum2_sq = 0
        
        # Calculate adjusted cosine similarity
        for user in common_users:
            user_mean = self.data[self.data['userId'] == user]['rating'].mean()
            
            item1_rating = item1_ratings[item1_ratings['userId'] == user]['rating'].iloc[0]
            item2_rating = item2_ratings[item2_ratings['userId'] == user]['rating'].iloc[0]
            
            # Adjust ratings by user's mean
            rating1_adjusted = item1_rating - user_mean
            rating2_adjusted = item2_rating - user_mean
            
            numerator += rating1_adjusted * rating2_adjusted
            sum1_sq += rating1_adjusted * rating1_adjusted
            sum2_sq += rating2_adjusted * rating2_adjusted
        
        denominator = (sum1_sq * sum2_sq) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def __calculate_rating(self, similarities: list, item_id: int, user_id: int) -> float:
        """
        Calculate predicted rating using weighted average of user's ratings for similar items.
        
        Formula:
            prediction = item_mean + Σ(similarity * (user_rating - item_mean)) / Σ|similarity|
        
        Args:
            similarities (list): List of (similarity, item_id) tuples, sorted by similarity
            item_id (int): Target item ID
            user_id (int): Target user ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
        """
        if not similarities:
            return 3.0  # Default rating if no similarities found
        
        target_item_mean = self.data[self.data['movieId'] == item_id]['rating'].mean()
        
        numerator = 0
        denominator = 0
        
        for similarity, other_item in similarities:
            user_rating_for_item = self.data[
                (self.data['userId'] == user_id) & 
                (self.data['movieId'] == other_item)
            ]['rating'].iloc[0]
            
            other_item_mean = self.data[self.data['movieId'] == other_item]['rating'].mean()
            
            numerator += similarity * (user_rating_for_item - other_item_mean)
            denominator += abs(similarity)
        
        if denominator == 0:
            return target_item_mean
        
        prediction = target_item_mean + (numerator / denominator)
        return max(1.0, min(5.0, prediction))
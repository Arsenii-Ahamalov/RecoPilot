from .base import BaseRecommender
import pandas as pd
import numpy as np

class GenreBasedRecommender(BaseRecommender):
    """
    Genre-Based Content Filtering Recommender.
    
    This algorithm predicts ratings based on genre preferences learned from a user's
    rating history. It analyzes which genres a user likes/dislikes and predicts
    ratings for new movies based on their genre composition.
    
    The prediction algorithm:
    1. Extract genres of the target movie
    2. For each genre, calculate user's average rating for that genre
    3. If user hasn't rated any movies in a genre, use global genre average
    4. Combine genre preferences using weighted average (weight = number of movies rated)
    
    Attributes:
        data (pd.DataFrame): User ratings data
        movie_data (pd.DataFrame): Movie metadata with genre information
    
    Example:
        >>> recommender = GenreBasedRecommender()
        >>> recommender.fit(ratings_df, movies_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)
    """
    
    def __init__(self):
        """Initialize the Genre-Based Recommender."""
        super().__init__()
    
    def fit(self, ratings_df: pd.DataFrame, movie_df: pd.DataFrame) -> 'GenreBasedRecommender':
        """
        Train the model by storing ratings and movie data.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            movie_df (pd.DataFrame): Movie metadata with genre columns (binary encoded)
                                   Expected columns: 'movieId', genre columns (0/1), optional 'release_year'
            
        Returns:
            GenreBasedRecommender: Self for method chaining
        """
        self.data = ratings_df
        self.movie_data = movie_df
        self.user_means = self.data.groupby('userId')['rating'].mean()
        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating based on user's genre preferences.
        
        Algorithm:
        1. Check if user already rated this item (return actual rating)
        2. Extract target movie's genres
        3. For each genre, get user's preference (average rating for that genre)
        4. If user hasn't rated movies in a genre, use global genre average
        5. Combine genre preferences using weighted average
        
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
            
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
        
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0] 

        movie_row = self.movie_data[self.movie_data['movieId'] == item_id]
        if movie_row.empty:
            return 3.0
        movie_info = movie_row.iloc[0]
        genre_cols = [c for c in self.movie_data.columns if c not in ['movieId', 'release_year']]
        movie_genre_mask = movie_info[genre_cols].values.astype(bool)
        selected_genres = [g for g, m in zip(genre_cols, movie_genre_mask) if m]

        if not selected_genres:
            return 3.0

        users_movies = self.data[self.data['userId'] == user_id][['movieId', 'rating']]
        if users_movies.empty:
            genre_means = np.array([self.__find_genre_mean(g) for g in selected_genres], dtype=float)
            return float(np.mean(genre_means)) if len(genre_means) else 3.0

        users_movie_meta = self.movie_data.set_index('movieId').reindex(users_movies['movieId'].values)
        genre_matrix = users_movie_meta[selected_genres].values.astype(int)
        ratings_vec = users_movies['rating'].values.reshape(-1, 1)
        counts = genre_matrix.sum(axis=0)
        with np.errstate(invalid='ignore'):
            sums = (genre_matrix * ratings_vec).sum(axis=0)
            user_genre_means = np.divide(sums, counts, where=counts != 0)

        for idx, g in enumerate(selected_genres):
            if counts[idx] == 0 or not np.isfinite(user_genre_means[idx]):
                user_genre_means[idx] = self.__find_genre_mean(g)
                counts[idx] = 1

        weighted = np.dot(user_genre_means, counts)
        denom = counts.sum()
        if denom == 0:
            return 3.0
        return float(weighted / denom)
    
    def __find_genre_data(self, genre: str, user_id: int) -> tuple:
        """
        Find user's rating pattern for a specific genre.
        
        Args:
            genre (str): Genre name (column name in movie_data)
            user_id (int): User ID to analyze
            
        Returns:
            tuple: (average_rating, movie_count) for this genre, or (-1, 0) if no movies rated
        """
        users_movies = self.data[self.data['userId'] == user_id][['movieId', 'rating']]
        if users_movies.empty:
            return (-1, 0)
        meta = self.movie_data.set_index('movieId').reindex(users_movies['movieId'].values)
        mask = meta[genre].values.astype(bool)
        count = int(mask.sum())
        if count == 0:
            return (-1, 0)
        genre_mean = float(users_movies['rating'].values[mask].mean())
        return (genre_mean, count)
    
    def __find_genre_mean(self, genre: str) -> float:
        """
        Calculate global average rating for a specific genre.
        
        Args:
            genre (str): Genre name (column name in movie_data)
            
        Returns:
            float: Global average rating for movies in this genre
        """
        genre_movies = self.movie_data[self.movie_data[genre] == 1]['movieId'].values
        if len(genre_movies) == 0:
            return 3.0
        return float(self.data[self.data['movieId'].isin(genre_movies)]['rating'].mean())
    
    def __calculate_users_rating(self, result_data: list) -> float:
        """
        Calculate final prediction by combining genre preferences.
        
        Uses weighted average where weight is the number of movies the user
        has rated in each genre (more rated movies = higher confidence).
        
        Args:
            result_data (list): List of (rating, weight) tuples for each genre
            
        Returns:
            float: Weighted average rating, or 3.0 if no valid data
        """
        if not result_data:
            return 3.0
        arr = np.array(result_data, dtype=float)
        weights = arr[:, 1]
        scores = arr[:, 0]
        denom = weights.sum()
        if denom == 0:
            return 3.0
        return float(np.dot(scores, weights) / denom)


class DemographicBasedRecommender(BaseRecommender):
    """
    Demographic-Based Content Filtering Recommender.
    
    This algorithm predicts ratings based on demographic similarity between users.
    It finds users with similar demographic profiles (age, gender, occupation)
    and uses their ratings to predict ratings for the target user.
    
    The similarity calculation considers:
    - Age similarity: 1 - |age_diff| / max_age_diff
    - Gender similarity: 1 if same gender, 0 otherwise  
    - Occupation similarity: 1 if same occupation, 0 otherwise
    
    Final similarity = age_weight * age_sim + gender_weight * gender_sim + occupation_weight * occupation_sim
    
    Attributes:
        k (int): Number of most similar users to consider for prediction
        similarity_cache (dict): Cache of computed demographic similarities
        data (pd.DataFrame): User ratings data
        users_data (pd.DataFrame): User demographic data
    
    Example:
        >>> recommender = DemographicBasedRecommender(k=30)
        >>> recommender.fit(ratings_df, users_df)
        >>> prediction = recommender.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, k: int = 50):
        """
        Initialize Demographic-Based Recommender.
        
        Args:
            k (int): Number of most demographically similar users to consider.
                    Default: 50
        """
        super().__init__()
        self.k = k
        self.similarity_cache = {}
        
    def fit(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> 'DemographicBasedRecommender':
        """
        Train the model by storing ratings and user demographic data.
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            users_df (pd.DataFrame): User demographics with columns ['userId', 'age', 'F', 'occupation']
                                    'F' should be 1 for female, 0 for male
            
        Returns:
            DemographicBasedRecommender: Self for method chaining
        """
        self.data = ratings_df
        self.users_data = users_df
        self.is_fitted = True
        return self
        
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating based on demographically similar users' ratings.
        
        Algorithm:
        1. Check if user already rated this item (return actual rating)
        2. Find users who have rated this item
        3. Calculate demographic similarity with these users
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
        Get cached demographic similarity between two users if available.
        
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
            
    def __calculate_demographic_similarity(self, user1: int, user2: int) -> float:
        """
        Calculate demographic similarity between two users.
        
        Combines age, gender, and occupation similarities with predefined weights:
        - Age weight: 0.4 (most important - similar life stages)
        - Gender weight: 0.3 (important for many preferences)
        - Occupation weight: 0.3 (indicates similar lifestyle/income)
        
        Args:
            user1 (int): First user ID
            user2 (int): Second user ID
            
        Returns:
            float: Demographic similarity between 0 and 1
        """
        AGE_WEIGHT = 0.4
        SEX_WEIGHT = 0.3
        OCCUPATION_WEIGHT = 0.3
        MAX_AGE_DIFF = 55  
        
        user1_data = self.users_data[self.users_data['userId'] == user1].iloc[0]
        user2_data = self.users_data[self.users_data['userId'] == user2].iloc[0]
        
        sex_similarity = 1 if user1_data['F'] == user2_data['F'] else 0
        occupation_similarity = 1 if user1_data['occupation'] == user2_data['occupation'] else 0
        
        age_diff = abs(user1_data['age'] - user2_data['age'])
        age_similarity = max(0, 1 - age_diff / MAX_AGE_DIFF)
        
        return (AGE_WEIGHT * age_similarity + 
                SEX_WEIGHT * sex_similarity + 
                OCCUPATION_WEIGHT * occupation_similarity)
        
    def __calculate_rating(self, similarities: list, item_id: int, user_id: int) -> float:
        """
        Calculate predicted rating using weighted average of demographically similar users.
        
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


ContentBased = GenreBasedRecommender 
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
        self.user_mean = self.data.groupby('userId')['rating'].mean()
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
            
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
        
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0] 

        relevant_users = self._find_relevant_users(item_id)

        if len(relevant_users) == 0:
            return float(self.user_mean.get(user_id, self.data['rating'].mean()))

        target_r = self.data[self.data['userId'] == user_id][['movieId', 'rating']].rename(columns={'rating': 'r_u'})
        others_r = self.data[self.data['userId'].isin(relevant_users)][['userId', 'movieId', 'rating']].rename(columns={'rating': 'r_v'})

        merged = others_r.merge(target_r, on='movieId', how='inner')
        if merged.empty:
            return float(self.user_mean.get(user_id, self.data['rating'].mean()))

        merged['x'] = merged['r_u']
        merged['y'] = merged['r_v']
        merged['x2'] = merged['x'] * merged['x']
        merged['y2'] = merged['y'] * merged['y']
        merged['xy'] = merged['x'] * merged['y']

        agg = merged.groupby('userId').agg(
            n=('x', 'size'),
            sum_x=('x', 'sum'),
            sum_y=('y', 'sum'),
            sum_x2=('x2', 'sum'),
            sum_y2=('y2', 'sum'),
            sum_xy=('xy', 'sum')
        )

        num = agg['sum_xy'] - (agg['sum_x'] * agg['sum_y'] / agg['n'])
        den = np.sqrt((agg['sum_x2'] - (agg['sum_x'] ** 2) / agg['n']) * (agg['sum_y2'] - (agg['sum_y'] ** 2) / agg['n']))
        with np.errstate(divide='ignore', invalid='ignore'):
            sims_series = (num / den).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        item_ratings = self.data[self.data['movieId'] == item_id][['userId', 'rating']].rename(columns={'rating': 'other_item_rating'})
        sim_df = sims_series.rename('similarity').reset_index().merge(item_ratings, on='userId', how='inner')

        for _, row in sim_df.iterrows():
            ou = int(row['userId'])
            key = (min(user_id, ou), max(user_id, ou))
            self.similarity_cache[key] = float(row['similarity'])

        sim_df = sim_df.sort_values('similarity', ascending=False)
        if len(sim_df) > self.k:
            sim_df = sim_df.head(self.k)

        target_user_mean = float(self.user_mean.get(user_id, self.data['rating'].mean()))
        other_means = self.user_mean.reindex(sim_df['userId'].values).values
        sims = sim_df['similarity'].values
        other_item_r = sim_df['other_item_rating'].values

        denom = np.sum(np.abs(sims))
        if denom == 0:
            return float(target_user_mean)
        num = np.sum(sims * (other_item_r - other_means))
        prediction = target_user_mean + (num / denom)
        return float(np.clip(prediction, 1.0, 5.0))
    
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
        u1 = self.data[self.data['userId'] == user1][['movieId', 'rating']].rename(columns={'rating': 'x'})
        u2 = self.data[self.data['userId'] == user2][['movieId', 'rating']].rename(columns={'rating': 'y'})
        merged = u1.merge(u2, on='movieId', how='inner')
        if merged.empty:
            return 0.0
        n = merged.shape[0]
        sum_x = merged['x'].sum()
        sum_y = merged['y'].sum()
        sum_x2 = (merged['x'] ** 2).sum()
        sum_y2 = (merged['y'] ** 2).sum()
        sum_xy = (merged['x'] * merged['y']).sum()
        num = sum_xy - (sum_x * sum_y / n)
        den = np.sqrt((sum_x2 - (sum_x ** 2) / n) * (sum_y2 - (sum_y ** 2) / n))
        if den == 0 or not np.isfinite(den):
            return 0.0
        return float(num / den)
    
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
        if not similarities:
            return float(self.user_mean.get(user_id, self.data['rating'].mean()))
        sim_arr = np.array([s for s, _ in similarities], dtype=float)
        other_users = np.array([u for _, u in similarities], dtype=int)
        target_user_mean = float(self.user_mean.get(user_id, self.data['rating'].mean()))
        other_means = self.user_mean.reindex(other_users).values
        item_ratings = self.data[self.data['movieId'] == item_id][['userId', 'rating']]
        item_ratings = item_ratings.set_index('userId').reindex(other_users)['rating'].values
        denom = np.sum(np.abs(sim_arr))
        if denom == 0 or np.isnan(denom):
            return float(target_user_mean)
        num = np.sum(sim_arr * (item_ratings - other_means))
        prediction = target_user_mean + (num / denom)
        return float(np.clip(prediction, 1.0, 5.0))


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
        self.item_mean = self.data.groupby('movieId')['rating'].mean()
        self.user_mean = self.data.groupby('userId')['rating'].mean()


        df = self.data[['userId', 'movieId', 'rating']].copy()
        df['adj'] = df['rating'] - df['userId'].map(self.user_mean)

        self.item_users = (
            df.groupby('movieId')['userId']
            .apply(lambda s: s.values.astype('int32'))
            .to_dict()
        )
        self.item_adj = (
            df.groupby('movieId')['adj']
            .apply(lambda s: s.values.astype('float32'))
            .to_dict()
        )
        self.item_norm = {i: float(np.linalg.norm(v)) for i, v in self.item_adj.items()}

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
            
        existing_rating = self.data[
            (self.data['userId'] == user_id) & 
            (self.data['movieId'] == item_id)
        ]
        
        if not existing_rating.empty:
            return existing_rating['rating'].iloc[0]
        
        relevant_items = self._find_relevant_items(user_id)

        MAX_RELEVANT = max(3 * self.k, 200)
        if len(relevant_items) > MAX_RELEVANT:
            user_r = self.data[self.data['userId'] == user_id][['movieId', 'rating']].copy()
            mu = float(self.user_mean.get(user_id, self.data['rating'].mean()))
            user_r['abs_dev'] = (user_r['rating'] - mu).abs()
            relevant_items = (
                user_r.sort_values('abs_dev', ascending=False)
                .head(MAX_RELEVANT)['movieId']
                .values
            )
        similarities = []
        
        for other_item in relevant_items:
            item_sim = self.__get_items_sim(item_id, other_item)
            if item_sim is None:
                item_sim = self.__calculate_items_sim(item_id, other_item)
                key = (min(item_id, other_item), max(item_id, other_item))
                self.similarity_cache[key] = item_sim
            similarities.append((item_sim, other_item))
        
        similarities.sort(reverse=True)
        if len(similarities) > self.k:
            similarities = similarities[:self.k]
        return self.__calculate_rating(similarities, item_id, user_id)
    
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
<<<<<<< HEAD
        u1 = self.item_users.get(item1)
        u2 = self.item_users.get(item2)
        if u1 is None or u2 is None:
            return 0.0
        common, idx1, idx2 = np.intersect1d(u1, u2, assume_unique=False, return_indices=True)
        if common.size < 2:
=======
        i1 = self.data[self.data['movieId'] == item1][['userId', 'rating']].rename(columns={'rating': 'r1'})
        i2 = self.data[self.data['movieId'] == item2][['userId', 'rating']].rename(columns={'rating': 'r2'})
        merged = i1.merge(i2, on='userId', how='inner')
        if merged.empty:
            return 0.0
        user_means = self.user_mean.reindex(merged['userId'].values).values
        r1_adj = merged['r1'].values - user_means
        r2_adj = merged['r2'].values - user_means
        num = np.dot(r1_adj, r2_adj)
        den = np.sqrt(np.dot(r1_adj, r1_adj) * np.dot(r2_adj, r2_adj))
        if den == 0 or not np.isfinite(den):
>>>>>>> c2d05ae063b34468e553adf90b0a33230077e573
            return 0.0
        v1 = self.item_adj[item1][idx1]
        v2 = self.item_adj[item2][idx2]
        num = float(np.dot(v1, v2))
        den = self.item_norm.get(item1, 0.0) * self.item_norm.get(item2, 0.0)
        if den == 0.0 or not np.isfinite(den):
            return 0.0
        sim = num / den
        alpha = 25.0
        sim *= (common.size / (common.size + alpha))
        if not np.isfinite(sim):
            return 0.0
        return float(sim)
    
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
            return float(3.0)
        target_item_mean = float(self.item_mean.get(item_id, self.data['rating'].mean()))
        sim_arr = np.array([s for s, _ in similarities], dtype=float)
        other_items = np.array([i for _, i in similarities], dtype=int)
        user_item_r = self.data[self.data['userId'] == user_id][['movieId', 'rating']]
        user_item_r = user_item_r.set_index('movieId').reindex(other_items)['rating'].values
        other_item_means = self.item_mean.reindex(other_items).values
        denom = np.sum(np.abs(sim_arr))
        if denom == 0 or np.isnan(denom):
            return float(target_item_mean)
        num = np.sum(sim_arr * (user_item_r - other_item_means))
        prediction = target_item_mean + (num / denom)
        return float(np.clip(prediction, 1.0, 5.0))
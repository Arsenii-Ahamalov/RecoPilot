from .base import BaseRecommender
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class BasicMatrixFactorization(BaseRecommender):
    def __init__(self, k: int = 10, epochs: int = 100, learning_rate: float = 0.01, reg: float = 0.02):
        super().__init__()
        self.k = k
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = reg
        
        self.user_factors = None
        self.item_factors = None
        
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
    
    def fit(self, ratings_df: pd.DataFrame) -> 'BasicMatrixFactorization':
        self.data = ratings_df
        
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.k))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.k))
        
        self.global_bias = ratings_df['rating'].mean()
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        for epoch in range(self.epochs):
            for _, row in ratings_df.iterrows():
                user_id = row['userId']
                item_id = row['movieId']
                rating = row['rating']
                
                user_idx = self.user_id_to_idx[user_id]
                item_idx = self.item_id_to_idx[item_id]
                
                prediction = self._predict_single(user_idx, item_idx)
                error = rating - prediction
                
                user_factors_old = self.user_factors[user_idx].copy()
                item_factors_old = self.item_factors[item_idx].copy()
                user_bias_old = self.user_bias[user_idx]
                item_bias_old = self.item_bias[item_idx]
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factors_old - self.reg * user_factors_old
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factors_old - self.reg * item_factors_old
                )
                
                self.user_bias[user_idx] += self.learning_rate * (
                    error - self.reg * user_bias_old
                )
                self.item_bias[item_idx] += self.learning_rate * (
                    error - self.reg * item_bias_old
                )
        
        self.is_fitted = True
        return self
    
    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        """Internal prediction method using indices"""
        return (
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) +
            self.user_bias[user_idx] +
            self.item_bias[item_idx] +
            self.global_bias
        )
    
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if user_id not in self.user_id_to_idx:
            return self.global_bias
        if item_id not in self.item_id_to_idx:
            return self.global_bias
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        prediction = self._predict_single(user_idx, item_idx)
        
        return np.clip(prediction, 1.0, 5.0)


class SVDMatrixFactorization(BaseRecommender):
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        
        self.U = None           
        self.sigma = None       
        self.Vt = None          
        
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        
        self.global_bias = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'SVDMatrixFactorization':
        self.data = ratings_df
        
        pivot_matrix = ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        )
        
        filled_matrix = pivot_matrix.fillna(0)
        
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(pivot_matrix.index)}
        self.item_id_to_idx = {mid: idx for idx, mid in enumerate(pivot_matrix.columns)}
        
        sparse_matrix = csr_matrix(filled_matrix.values)
        
        # Determine valid k value - must be less than min dimension
        max_k = min(sparse_matrix.shape) - 1
        actual_k = min(self.k, max_k)
        
        if actual_k <= 0:
            self.U = np.zeros((sparse_matrix.shape[0], 1))
            self.sigma = np.array([0.0])
            self.Vt = np.zeros((1, sparse_matrix.shape[1]))
        else:
            U, sigma, Vt = svds(sparse_matrix, k=actual_k)
            
            idx = np.argsort(sigma)[::-1]
            self.U = U[:, idx]
            self.sigma = sigma[idx]
            self.Vt = Vt[idx, :]
        
        self.global_bias = ratings_df['rating'].mean()
        
        self.is_fitted = True
        return self
    
    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        """Internal prediction method using indices"""
        user_factors = self.U[user_idx]              
        item_factors = self.Vt[:, item_idx]          
        
        prediction = np.sum(user_factors * self.sigma * item_factors)
        
        return prediction + self.global_bias
    
    def predict(self, user_id: int, item_id: int) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_id_to_idx:
            return self.global_bias
        
        if item_id not in self.item_id_to_idx:
            return self.global_bias
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        prediction = self._predict_single(user_idx, item_idx)
        
        return np.clip(prediction, 1.0, 5.0)
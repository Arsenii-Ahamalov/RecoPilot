from .base import BaseRecommender
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class BasicMatrixFactorization(BaseRecommender):
    """
    Basic Matrix Factorization using Gradient Descent.
    
    This algorithm factorizes the user-item rating matrix into two lower-dimensional
    matrices: user factors and item factors. It learns latent features that capture
    user preferences and item characteristics.
    
    Mathematical Model:
        R ≈ U × I^T + user_bias + item_bias + global_bias
        where:
        - R: user-item rating matrix (sparse)
        - U: user factors matrix (n_users × k)
        - I: item factors matrix (n_items × k)
        - k: number of latent factors
        
    Training uses Stochastic Gradient Descent with L2 regularization:
        Loss = Σ(rating - prediction)² + λ(||U||² + ||I||² + ||biases||²)
    
    Attributes:
        k (int): Number of latent factors
        epochs (int): Number of training iterations
        learning_rate (float): Learning rate for gradient descent
        reg (float): L2 regularization parameter
        user_factors (np.ndarray): Learned user factor matrix
        item_factors (np.ndarray): Learned item factor matrix
        user_bias (np.ndarray): User bias terms
        item_bias (np.ndarray): Item bias terms
        global_bias (float): Global rating average
    
    Example:
        >>> mf = BasicMatrixFactorization(k=50, epochs=100, learning_rate=0.01, reg=0.02)
        >>> mf.fit(ratings_df)
        >>> prediction = mf.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, k: int = 10, epochs: int = 100, learning_rate: float = 0.01, reg: float = 0.02):
        """
        Initialize Basic Matrix Factorization model.
        
        Args:
            k (int): Number of latent factors. Higher k can capture more complex patterns
                    but may lead to overfitting. Typical range: 10-200. Default: 10
            epochs (int): Number of training iterations. More epochs generally improve
                         accuracy but increase training time. Default: 100
            learning_rate (float): Step size for gradient descent. Higher values learn
                                  faster but may overshoot. Typical range: 0.001-0.1. Default: 0.01
            reg (float): L2 regularization strength. Prevents overfitting by penalizing
                        large factor values. Typical range: 0.001-0.1. Default: 0.02
        """
        super().__init__()
        self.k = k
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = reg
        
        # Will be initialized during fit()
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        
        # ID to matrix index mappings
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
    
    def fit(self, ratings_df: pd.DataFrame) -> 'BasicMatrixFactorization':
        """
        Train the matrix factorization model using gradient descent.
        
        Algorithm:
        1. Initialize user and item factors with small random values
        2. Initialize bias terms
        3. For each epoch:
           - For each known rating:
             a. Make prediction using current factors
             b. Calculate error
             c. Update factors and biases using gradients
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            BasicMatrixFactorization: Self for method chaining
        """
        self.data = ratings_df
        
        # Create mappings from IDs to matrix indices
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Initialize factor matrices with small random values (Xavier initialization)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.k))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.k))
        
        # Initialize bias terms
        self.global_bias = ratings_df['rating'].mean()
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # Stochastic Gradient Descent training
        for epoch in range(self.epochs):
            for _, row in ratings_df.iterrows():
                user_id = row['userId']
                item_id = row['movieId']
                rating = row['rating']
                
                # Convert IDs to indices
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
        """
        Internal prediction method using matrix indices.
        
        Prediction formula:
            rating = user_factors · item_factors + user_bias + item_bias + global_bias
        
        Args:
            user_idx (int): User matrix index
            item_idx (int): Item matrix index
            
        Returns:
            float: Raw prediction (not clamped)
        """
        return (
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) +
            self.user_bias[user_idx] +
            self.item_bias[item_idx] +
            self.global_bias
        )
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id (int): User ID
            item_id (int): Item ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
            KeyError: If user_id or item_id not found in training data
        """
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
    """
    SVD-based Matrix Factorization using Singular Value Decomposition.
    
    This algorithm uses mathematical SVD to decompose the user-item rating matrix
    into three matrices: U (user factors), Σ (singular values), and V^T (item factors).
    Unlike gradient descent approaches, SVD provides a direct mathematical solution.
    
    Mathematical Model:
        R = U × Σ × V^T
        where:
        - U: user factors matrix (n_users × k)
        - Σ: diagonal matrix of singular values (k × k) - importance weights
        - V^T: item factors matrix transposed (k × n_items)
        
    Prediction formula:
        rating = Σ(U[user,i] × σ[i] × V^T[i,item]) + global_bias
    
    Key advantages:
    - Mathematically optimal k-rank approximation
    - Faster training than gradient descent
    - Deterministic results (no random initialization)
    - Automatic feature importance via singular values
    
    Attributes:
        k (int): Number of factors to keep from SVD decomposition
        U (np.ndarray): User factors matrix
        sigma (np.ndarray): Singular values (importance weights)
        Vt (np.ndarray): Item factors matrix (transposed)
        global_bias (float): Global rating average
    
    Example:
        >>> svd_mf = SVDMatrixFactorization(k=50)
        >>> svd_mf.fit(ratings_df)
        >>> prediction = svd_mf.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize SVD Matrix Factorization model.
        
        Args:
            k (int): Number of singular values/factors to keep. Must be less than
                    min(n_users, n_items). Higher k captures more details but may
                    include noise. Typical range: 10-200. Default: 10
        """
        super().__init__()
        self.k = k
        
        self.U = None           
        self.sigma = None       
        self.Vt = None          
        
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        
        self.global_bias = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'SVDMatrixFactorization':
        """
        Train the SVD model by decomposing the user-item matrix.
        
        Algorithm:
        1. Create user-item matrix from sparse ratings data
        2. Fill missing values with 0 (for SVD compatibility)
        3. Apply truncated SVD to get top k factors
        4. Store resulting U, Σ, V^T matrices
        5. Sort factors by importance (descending singular values)
        
        Args:
            ratings_df (pd.DataFrame): Training data with columns ['userId', 'movieId', 'rating']
            
        Returns:
            SVDMatrixFactorization: Self for method chaining
        """
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
        """
        Internal prediction method using matrix indices.
        
        SVD prediction formula:
            rating = Σ(U[user,i] × σ[i] × V^T[i,item]) + global_bias
        
        Args:
            user_idx (int): User matrix index
            item_idx (int): Item matrix index
            
        Returns:
            float: Raw prediction (includes global bias)
        """
        user_factors = self.U[user_idx]              
        item_factors = self.Vt[:, item_idx]          
        
        prediction = np.sum(user_factors * self.sigma * item_factors)
        
        return prediction + self.global_bias
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using SVD factors.
        
        Args:
            user_id (int): User ID
            item_id (int): Item ID
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
        """
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
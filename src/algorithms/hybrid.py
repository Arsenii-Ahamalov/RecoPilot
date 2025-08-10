import pandas as pd
import numpy as np
from .base import BaseRecommender

class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommendation System using Weighted Combination Strategy.
    
    This class combines multiple recommendation algorithms to leverage their individual
    strengths and mitigate their weaknesses. The hybrid approach often outperforms
    individual algorithms by:
    - Combining diverse perspectives (collaborative, content-based, matrix factorization)
    - Handling cold-start problems better
    - Providing more robust predictions
    - Reducing the impact of individual algorithm failures
    
    Current Implementation: Weighted Mean Strategy
        final_prediction = Σ(weight_i × algorithm_i_prediction)
        where Σ(weight_i) = 1.0
    
    Attributes:
        algorithms (dict): Dictionary of {name: algorithm_instance} pairs
        weights (dict): Dictionary of {name: weight} pairs for combining predictions
    
    Example:
        >>> # Create individual algorithms
        >>> cf = UserBasedCF(k=30)
        >>> mf = BasicMatrixFactorization(k=50, epochs=100)
        >>> content = GenreBasedRecommender()
        
        >>> # Create hybrid system
        >>> hybrid = HybridRecommender(
        ...     algorithms={
        ...         'collaborative': cf,
        ...         'matrix_factorization': mf,
        ...         'content_based': content
        ...     },
        ...     weights={
        ...         'collaborative': 0.4,
        ...         'matrix_factorization': 0.4, 
        ...         'content_based': 0.2
        ...     }
        ... )
        
        >>> # Train all algorithms
        >>> hybrid.fit(ratings_df)
        
        >>> # Make hybrid predictions
        >>> prediction = hybrid.predict(user_id=1, item_id=101)
    """
    
    def __init__(self, algorithms: dict[str, BaseRecommender], weights: dict[str, float] = None,movies_df: pd.DataFrame = None,users_df: pd.DataFrame = None):
        """
        Initialize the Hybrid Recommender system.
        
        Args:
            algorithms (dict): Dictionary mapping algorithm names to algorithm instances.
                             All algorithms must inherit from BaseRecommender.
                             Example: {'cf': UserBasedCF(), 'mf': BasicMatrixFactorization()}
            weights (dict, optional): Dictionary mapping algorithm names to their weights.
                                     Weights must sum to 1.0. If None, equal weights are assigned
                                     automatically. Example: {'cf': 0.6, 'mf': 0.4}
        
        Raises:
            ValueError: If weights don't sum to 1.0 (within tolerance)
            ValueError: If algorithm names and weight names don't match
        """
        super().__init__() 
        self.algorithms = algorithms
        self.movies_df = movies_df
        self.users_df = users_df
        if weights is None:
            n_algs = len(algorithms)
            self.weights = {name: 1.0/n_algs for name in algorithms.keys()}
        else:
            self.weights = weights
            
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        if set(self.algorithms.keys()) != set(self.weights.keys()):
            raise ValueError("Algorithm names and weight names must match")
    
    def fit(self, ratings_df: pd.DataFrame) -> 'HybridRecommender':
        """
        Fit component algorithms for the hybrid model, or skip fitting.

        Behavior
          - Trains the algorithms in self.algorithms on ratings_df.
          - Special cases:
            - 'GenreBasedRecommender' is fitted with (ratings_df, self.movies_df)
            - 'DemographicBasedRecommender' is fitted with (ratings_df, self.users_df)
          - Validates that all algorithms inherit from BaseRecommender.
          - Sets self.data = ratings_df.

        Args
        - ratings_df: Training data with columns ['userId', 'movieId', 'rating'].

        Returns
        - HybridRecommender: self (for chaining).

        Raises
        - TypeError: if any algorithm in self.algorithms is not a BaseRecommender
                     (checked when skip_fit is False).
        """
        self.data = ratings_df  
   
        for name, algorithm in self.algorithms.items():
            if name == 'GenreBasedRecommender':
                algorithm.fit(ratings_df,self.movies_df)
            elif name == 'DemographicBasedRecommender':
                algorithm.fit(ratings_df,self.users_df)
            else:
                algorithm.fit(ratings_df)
        for name, algorithm in self.algorithms.items():
            if not isinstance(algorithm, BaseRecommender):
                raise TypeError(f"Algorithm '{name}' must inherit from BaseRecommender")

        self.is_fitted = True
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Make a hybrid prediction by combining individual algorithm predictions.
        
        The weighted combination strategy computes:
            final_prediction = Σ(weight_i × prediction_i)
        
        If any algorithm fails to make a prediction, it's excluded from the
        combination and a warning is printed. If all algorithms fail, an
        exception is raised.
        
        Args:
            user_id (int): User ID for prediction
            item_id (int): Item ID for prediction
            
        Returns:
            float: Predicted rating (clamped to [1.0, 5.0] range)
            
        Raises:
            ValueError: If model hasn't been fitted yet
            ValueError: If all algorithms fail to predict
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        total_prediction = 0.0
        successful_predictions = 0
        
        for name, algorithm in self.algorithms.items():
            try:
                pred = algorithm.predict(user_id, item_id)
                total_prediction += self.weights[name] * pred
                successful_predictions += 1
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
        
        if successful_predictions == 0:
            raise ValueError("All algorithms failed to predict")
        
        return np.clip(total_prediction, 1.0, 5.0)
    
    def get_algorithm_info(self) -> dict:
        """
        Get information about all algorithms in the hybrid system.
        
        Useful for debugging, monitoring, and understanding the hybrid composition.
        
        Returns:
            dict: Information about each algorithm including:
                 - class: Class name of the algorithm
                 - weight: Weight assigned to this algorithm
                 - fitted: Whether the algorithm has been trained
        
        Example:
            >>> info = hybrid.get_algorithm_info()
            >>> print(info)
            {
                'collaborative': {
                    'class': 'UserBasedCF', 
                    'weight': 0.4, 
                    'fitted': True
                },
                'matrix_factorization': {
                    'class': 'BasicMatrixFactorization',
                    'weight': 0.4,
                    'fitted': True
                }
            }
        """
        return {
            name: {
                'class': type(alg).__name__,
                'weight': self.weights[name],
                'fitted': alg.is_fitted if hasattr(alg, 'is_fitted') else False
            }
            for name, alg in self.algorithms.items()
        }
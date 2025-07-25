# RecoPilot - Future Improvements TODO

## 1. Reimplement Architecture to Reduce Code Repetition

### What Exactly Repeats:
- **User similarity calculation framework**: Caching mechanism, K-nearest neighbors logic, similarity cache management
- **Rating prediction patterns**: User mean calculations, weighted averages, numerator/denominator patterns  
- **Existing rating checks**: All predict() methods check for existing ratings first with identical logic
- **fit() method patterns**: Setting is_fitted=True, storing data, returning self for method chaining
- **Base class inheritance**: recommend() and predict_for_user() methods are identical across algorithms

### Proposed Solution:
Create `MemoryBasedRecommender` base class that contains:
- Common similarity caching logic
- Standard K-nearest neighbors framework  
- Shared rating prediction formula
- Abstract similarity calculation method for subclasses to implement

## 2. Think About Smarter Weight Strategy

### Current Limitations:
- Simple equal weights or basic formulas
- No adaptation based on data characteristics
- Fixed weights regardless of context

### Research Areas:
- **Adaptive weights based on data sparsity**: More weight to reliable features when data is sparse
- **Genre importance weights learned from data**: Automatically discover which genres matter most
- **Demographic feature importance via correlation analysis**: Learn age vs gender vs occupation importance  
- **Time-decay weights for recent vs old ratings**: Recent ratings should have more influence
- **Confidence-based weighting**: More ratings = higher confidence, adjust weights accordingly

## 3. Implement Non-Regression Based Algorithms

### Current State:
All algorithms predict exact ratings (regression approach)

### Alternative Approaches to Explore:
- **Classification-based approaches**: Predict rating bins (low/medium/high) instead of exact values
- **Ranking-based methods**: Focus on item ordering rather than rating prediction accuracy
- **Deep learning approaches**: Neural collaborative filtering, autoencoders
- **Association rule mining**: "Users who liked X also liked Y" patterns

## Implementation Priority:
1. **High Priority**: Architecture refactoring (reduces maintenance burden)
2. **Medium Priority**: Smarter weight strategy (improves existing algorithms)  
3. **Low Priority**: Non-regression algorithms (research/experimental phase)

---
*Last updated: 25.07.2025* 
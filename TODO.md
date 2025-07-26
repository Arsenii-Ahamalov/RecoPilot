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

## 2. Incorporate Temporal Information (Timestamps & Movie Age)

### Current Limitations:
- **No timestamp usage**: All ratings treated equally regardless of when they occurred
- **No movie age consideration**: Classic movies vs new releases treated identically
- **Static preferences assumption**: User tastes assumed constant over time
- **Missing trend awareness**: Can't capture seasonal patterns or popularity shifts

### Implementation Areas:
- **Time-Weighted Collaborative Filtering**: Recent ratings have higher influence in similarity calculations
- **Temporal Matrix Factorization**: Add time-specific bias terms and temporal factors
- **Movie Age as Content Feature**: Include release year, age categories (Classic/Retro/Modern) in content-based filtering
- **Session-Based Recommendations**: Weight recent viewing patterns more heavily
- **Seasonal/Trend Awareness**: Capture time-of-year preferences and popularity trends

### Specific Tasks:
1. **Modify UserBasedCF**: Add `time_decay_factor` parameter and time-weighted similarity calculation
2. **Extend ContentBased**: Add movie age features (age, decade, era) to similarity calculations
3. **Enhance BasicMatrixFactorization**: Include temporal bias terms and time bins (monthly/yearly)


## 3. Think About Smarter Weight Strategy

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

## 4. Implement Non-Regression Based Algorithms

### Current State:
All algorithms predict exact ratings (regression approach)

### Alternative Approaches to Explore:
- **Classification-based approaches**: Predict rating bins (low/medium/high) instead of exact values
- **Ranking-based methods**: Focus on item ordering rather than rating prediction accuracy
- **Deep learning approaches**: Neural collaborative filtering, autoencoders
- **Association rule mining**: "Users who liked X also liked Y" patterns

## Implementation Priority:
1. **High Priority**: Architecture refactoring (reduces maintenance burden)
2. **High Priority**: Temporal information integration (significant accuracy improvements expected)
3. **Medium Priority**: Smarter weight strategy (improves existing algorithms)  
4. **Low Priority**: Non-regression algorithms (research/experimental phase)

---
*Last updated: 26.07.2025* 
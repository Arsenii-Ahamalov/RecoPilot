# RecoPilot

**A Movie-Lens 1M based movie-recommendation project**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 1 Overview
RecoPilot is an end-to-end learning project that walks from data exploration through model deployment.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Exploratory data analysis (EDA) | ✅ Complete |
| 2 | Data preprocessing & feature engineering | ✅ Complete |
| 3 | Algorithm implementation (baselines → CF → MF → hybrid) | ✅ Complete |
| 4 | Evaluation & model selection | ✅ Complete |
| 5 | Flask API + web demo | ⏳ |

---

## 2 Project Structure
```text
RecoPilot/
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── movies.dat                # raw MovieLens files
│   ├── ratings.dat
│   ├── users.dat
│   ├── movies_processed.csv      # cleaned & tokenised
│   ├── ratings_processed.csv
│   └── users_processed.csv
│
├── src/
│   ├── notebooks/                # Phase notebooks
│   │   ├── data_exploration.ipynb    # Phase-1 EDA
│   │   └── data_preprocessing.ipynb  # Phase-2 preprocessing
│   └── algorithms/               # Recommendation algorithms (8 implementations)
│       ├── base.py              # Abstract base class
│       ├── baselines.py         # Global/user/movie averages (4 algorithms)
│       ├── collaborative_filtering.py # User-based & item-based CF
│       ├── matrix_factorization.py # Basic SGD & SVD implementations
│       ├── content_based.py     # Genre & demographic filtering
│       └── hybrid.py            # Weighted combination strategy
│
├── docs/
│   ├── phase1-data-exploration-summary.md
│   ├── phase2-data-preprocessing.md
│   ├── phase3-algorithm-implementation.md
│   └── phase4/
│       ├── conclusion.md                 # Phase 4 summary and results
│       └── plots/                        # Phase 4 result plots
│
├── models/                       # saved best parameters (unfitted JSON)
├── tests/                        # comprehensive algorithm tests (500+ cases)
│   ├── test_baselines.py         # baseline algorithm tests
│   ├── test_collaborative_filtering.py # CF algorithm tests
│   ├── test_matrix_factorization.py    # MF algorithm tests
│   └── test_hybrid.py            # hybrid system tests
└── app/                          # Flask API / front-end (future)
```

---

## 3 Installation
```bash
git clone https://github.com/Arsenii-Ahamalov/RecoPilot.git
cd RecoPilot
python -m venv venv           # create venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## 4 Data Preparation
Pre-processed CSV files (`movies_processed.csv`, `ratings_processed.csv`, `users_processed.csv`) live directly in the `data/` folder. To regenerate them, open and run **`src/data_preprocessing.ipynb`**.

---

## 5 Implemented Algorithms (Phase 3) ✅
1. **Baselines** – 4 algorithms: global, user, movie averages + bias model
2. **Memory-based CF** – UserBasedCF (Pearson) & ItemBasedCF (adj. cosine)
3. **Matrix factorization** – BasicMF (SGD) & SVD with bias terms
4. **Content-based** – Genre preferences & demographic similarity  
5. **Hybrid** – Flexible weighted combination with error handling

**Comprehensive features**: Full documentation, 500+ tests, production-ready code.  
**Next**: Phase 5 – Flask API + web demo.

---

## 6 Model Storage Strategy
**We store unfitted JSON configs with best hyperparameters** for production flexibility:
- **Unfitted configs**: Instantiate classes and fit on current data
- **Best parameters**: Frozen from Phase 4 tuning (see `models/*.json`)
- **Hybrid config**: Selected models and uniform weights saved in `models/Hybrid.json`

## 7 Road-map
- [x] **Phase 1**: Data exploration & analysis
- [x] **Phase 2**: Data preprocessing & feature engineering  
- [x] **Phase 3**: Algorithm implementation (8 algorithms + comprehensive testing)
- [x] **Phase 4**: Evaluation & model comparison
- [ ] **Phase 5**: Flask API + web demo  

---

## 7 License
MIT – see `LICENSE`.

---
*Last updated 2025-08-10*

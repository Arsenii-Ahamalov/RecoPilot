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
| 3 | Algorithm implementation (baselines → CF → MF → hybrid) | 🚀 Ready to start |
| 4 | Evaluation & model selection | ⏳ |
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
│   └── algorithms/               # Phase-3 recommendation algorithms
│       ├── base.py              # Abstract base class
│       ├── baselines.py         # Global/user/movie averages
│       ├── collaborative.py     # User-based & item-based CF
│       ├── matrix_factorization.py # SVD, Basic
│       ├── content_based.py     # Genre & demographic filtering
│       └── hybrid.py            # Combined approaches
│
├── docs/
│   ├── phase1-data-exploration-summary.md
│   └── phase2-data-preprocessing.md
│
├── models/                       # saved models
├── tests/                        # algorithm unit tests
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

## 5 Planned Algorithms (Phase 3)
1. **Baselines** – global / user / movie averages  
2. **Memory-based CF** – user-user & item-item (cosine / Pearson)  
3. **Matrix factorisation** – SVD / Basic 
4. **Content-based** – genre & year profile matching  
5. **Hybrid** – combine CF + content + demographics

Evaluation metrics: RMSE (prediction) and Precision@K / Recall@K / NDCG (top-N).

---

## 6 Road-map
- [ ] Implement baselines in `src/`  
- [ ] Build CF models & evaluation notebook  
- [ ] Train MF model, compare results  
- [ ] Save best model → `models/`  
- [ ] Expose recommendations via Flask API (`app/`)  
- [ ] Front-end demo page

---

## 7 License
MIT – see `LICENSE`.

---
*Last updated 2025-07-20*

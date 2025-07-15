# RecoPilot

**A starter template for building and deploying end-to-end recommender systems.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Docker](#docker)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Memory-based Collaborative Filtering**: User-User and Item-Item methods with cosine similarity.  
- **Model-based Collaborative Filtering**: Matrix factorization via SVD and ALS.  
- **Evaluation Metrics**: RMSE for rating prediction; Precision@K, Recall@K, MAP@K for top-N.  
- **Modular Codebase**: Clear separation between data processing, modeling, and serving.  
- **Command-Line Interface**: Fast, scriptable recommendations via terminal commands.  
- **REST API**: FastAPI endpoints for training and inference.  
- **Web Demo**: Simple React-free web page for interactive recommendations.  
- **Docker Support**: Containerized setup for consistent deployment.  
- **Reproducible Notebooks**: Example Jupyter notebooks for EDA and experiments.  
- **Unit Tests**: Automated tests for key components.


## Installation
1. **Clone the repository**:  
   `git clone https://github.com/<your-username>/RecoPilot.git && cd RecoPilot`  
2. **Create and activate virtual environment**:  
   `python -m venv venv`  
   `source venv/bin/activate  # Linux/macOS`  
   `venv\Scripts\activate     # Windows`  
3. **Install dependencies**:  
   `pip install -r requirements.txt`

## Project Structure
```
RecoPilot/
├── README.md               
├── LICENSE                 
├── requirements.txt        
├── data/
│   ├── raw/                
│   └── processed/          
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline.ipynb
│   └── 03_Factorization.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── baseline_model.py
│   ├── memory_based.py
│   ├── factorization.py
│   └── utils.py
├── app/
│   ├── main.py             
│   └── Dockerfile
├── models/                 
├── tests/
│   ├── test_data.py
│   └── test_models.py
└── docs/
    └── demo.gif
```

## Data Preparation
Place your dataset (e.g., `ratings.csv`) in `data/raw/`. The CSV should have columns: `user_id`, `item_id`, `rating`.
Run preprocessing:  
```
python src/data_preprocessing.py --input_path data/raw/ratings.csv --output_path data/processed/ratings_processed.csv
```

## Training
```
python src/train.py \
  --data_path data/processed/ratings_processed.csv \
  --model_type [baseline|memory|als|svd] \
  --params_path config/model_params.yaml \
  --output_dir models/
```
Trained models will be saved in `models/`.

## Evaluation
```
python src/evaluate.py \
  --model_path models/model.pkl \
  --test_data data/processed/test.csv \
  --metrics_path reports/metrics.json
```
Outputs RMSE, Precision@K, Recall@K, and MAP@K.

## Usage
### CLI
```
python src/predict.py --liked_items "The Matrix,Inception,Interstellar" --top_k 5
```
### API
```
uvicorn app.main:app --reload
```
- `POST /train` to train model  
- `GET /recommend?user_id={id}&k={top_k}` to get recommendations  
### Web Interface
Open `http://localhost:8000/`, enter user ID or liked items, click **Get Recommendations**.



## Testing
```
pytest --maxfail=1 --disable-warnings -q
```

## Configuration
- `config/model_params.yaml` for hyperparameters  
- `config/app_config.yaml` for API settings

## Contributing
1. Fork the repo  
2. Create a branch (`git checkout -b feature/my-feature`)  
3. Commit (`git commit -m 'Add feature'`)  
4. Push (`git push origin feature/my-feature`)  
5. Open a PR

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

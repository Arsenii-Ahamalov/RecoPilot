import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from pathlib import Path
import uuid

# Your algorithms
from src.algorithms.matrix_factorization import BasicMatrixFactorization
from src.algorithms.collaborative_filtering import ItemBasedCF

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# ---- Load base data once ----
ratings = pd.read_csv("data/ratings_processed.csv")

# Prefer MovieLens raw titles from movies.dat; fallback to processed CSV if needed
import re

def _extract_year(txt: str | None):
    if not isinstance(txt, str):
        return None
    m = re.search(r"\((\d{4})\)", txt)
    return int(m.group(1)) if m else None

def _build_movie_meta() -> dict[int, dict]:
    meta: dict[int, dict] = {}
    raw_path = Path("data/movies.dat")
    if raw_path.exists():
        # MovieLens 1M format: movieId::title::genres
        raw = pd.read_csv(
            raw_path, sep="::", engine="python",
            names=["movieId", "title", "genres"], encoding="latin-1"
        )
        for _, r in raw.iterrows():
            mid = int(r["movieId"])
            title = r["title"]
            meta[mid] = {"title": title, "year": _extract_year(title)}
        return meta

    # Fallback: try processed CSV and guess title/year columns if present
    try:
        movies = pd.read_csv("data/movies_processed.csv")
        title_col = next((c for c in movies.columns if c.lower() in {
            "title", "movie", "name", "movie_title", "original_title"
        }), None)
        year_col = next((c for c in movies.columns if c.lower() in {"year", "release_year"}), None)
        for _, r in movies.iterrows():
            mid = int(r["movieId"])
            title = r[title_col] if title_col else f"Movie {mid}"
            year = None
            if year_col:
                try:
                    year = int(r[year_col])
                except Exception:
                    year = None
            if year is None:
                year = _extract_year(str(title))
            meta[mid] = {"title": title, "year": year}
    except Exception:
        pass
    return meta

movie_meta = _build_movie_meta()

# Precompute popularity order (most-rated first)
_pop_counts = ratings.groupby("movieId").size().sort_values(ascending=False)
_popular_ids = _pop_counts.index.to_numpy(dtype=int)

# ---- Best single model (default): BasicMF ----
mf_params = json.loads(Path("models/BasicMatrixFactorization.json").read_text(encoding="utf-8"))
default_model = BasicMatrixFactorization(**mf_params).fit(ratings)

# ---- Session storage (in-memory) ----
session_id_to_user = {}         # sessionId -> synthetic userId
session_id_to_ratings = {}      # sessionId -> list of dicts {movieId, rating}
synthetic_user_start = 900000

# ---- Helpers ----
def _popularity_sample(n: int = 20, offset: int = 0):
    # rotate through popular list based on offset to vary results
    if len(_popular_ids) == 0:
        ids = []
    else:
        start = offset % len(_popular_ids)
        ids = np.roll(_popular_ids, -start)[:n]
    out = []
    for mid in ids:
        mm = movie_meta.get(int(mid))
        title = mm.get("title") if mm else f"Movie {int(mid)}"
        year = mm.get("year") if mm else None
        out.append({"movieId": int(mid), "title": title, "year": year})
    return out

def _append_session_ratings(base_df: pd.DataFrame, session_id: str) -> tuple[pd.DataFrame, int]:
    """Return (df_with_session, session_user_id)."""
    if session_id not in session_id_to_user:
        raise ValueError("Unknown sessionId")
    user_id = session_id_to_user[session_id]
    rows = session_id_to_ratings.get(session_id, [])
    if not rows:
        return base_df, user_id
    add_df = pd.DataFrame([{"userId": user_id, "movieId": r["movieId"], "rating": r["rating"]} for r in rows])
    return pd.concat([base_df, add_df], ignore_index=True), user_id

# ---- Routes ----
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/session/start")
def start_session():
    sid = uuid.uuid4().hex
    new_user_id = synthetic_user_start + len(session_id_to_user)
    session_id_to_user[sid] = new_user_id
    session_id_to_ratings[sid] = []
    return jsonify({"sessionId": sid, "userId": new_user_id})

@app.get("/movies/sample")
def movies_sample():
    size = int(request.args.get("size", 20))
    # allow client to pass a seed/offset to rotate results
    offset = int(request.args.get("offset", np.random.randint(0, max(1, len(_popular_ids)))))
    return jsonify(_popularity_sample(size, offset))

@app.post("/rate")
def rate():
    payload = request.get_json(force=True)
    session_id = payload["sessionId"]
    ratings_payload = payload["ratings"]  # [{movieId, rating}, ...]
    if session_id not in session_id_to_user:
        return jsonify({"error": "Unknown sessionId"}), 400
    # basic validation and clamp
    clean = []
    for r in ratings_payload:
        mid = int(r["movieId"])
        rt = float(r["rating"])
        rt = max(1.0, min(5.0, rt))
        clean.append({"movieId": mid, "rating": rt})
    session_id_to_ratings[session_id].extend(clean)
    return jsonify({"ok": True, "numRatings": len(session_id_to_ratings[session_id])})

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    user_id = int(payload["userId"])
    movie_id = int(payload["movieId"])
    unknown_user = user_id not in getattr(default_model, 'user_id_to_idx', {})
    unknown_item = movie_id not in getattr(default_model, 'item_id_to_idx', {})
    pred = float(default_model.predict(user_id, movie_id))
    mm = movie_meta.get(movie_id)
    title = mm.get("title") if mm else f"Movie {movie_id}"
    year = mm.get("year") if mm else None
    # avoid duplicating year if title already embeds it
    if isinstance(title, str) and year and f"({year})" in title:
        pass  # keep as-is
    return jsonify({
        "prediction": round(pred, 1),
        "movieId": movie_id,
        "title": title,
        "year": year,
        "unknownUser": bool(unknown_user),
        "unknownItem": bool(unknown_item)
    })

@app.get("/recommend")
def recommend():
    session_id = request.args.get("sessionId", "").strip()
    k = int(request.args.get("k", 10))

    if not session_id:
        return jsonify({"error": "sessionId required"}), 400

    if len(session_id_to_ratings.get(session_id, [])) == 0:
        return jsonify({"error": "no_ratings", "message": "Please rate some movies first."}), 400

    # Build temp DF with session ratings & recommend via fast ItemBasedCF
    tmp_df, user_id = _append_session_ratings(ratings, session_id)
    icf = ItemBasedCF(k=50).fit(tmp_df)
    recs = icf.recommend(user_id, k=k, exclude_seen=True)  # list of (score, movieId)

    out = []
    for score, mid in recs:
        mm = movie_meta.get(int(mid))
        title = mm.get("title") if mm else f"Movie {int(mid)}"
        year = mm.get("year") if mm else None
        out.append({"movieId": int(mid), "score": float(round(score, 1)), "title": title, "year": year})
    return jsonify(out)

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
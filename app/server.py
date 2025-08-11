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
import requests
from dotenv import load_dotenv

# Your algorithms
from src.algorithms.matrix_factorization import BasicMatrixFactorization, SVDMatrixFactorization
from src.algorithms.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.algorithms.content_based import GenreBasedRecommender, DemographicBasedRecommender

load_dotenv()  # load .env if present
app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

# ---- Load base data once ----
ratings = pd.read_csv("data/ratings_processed.csv")
try:
    users_df = pd.read_csv("data/users_processed.csv")
except Exception:
    users_df = pd.DataFrame(columns=["userId"])  # minimal fallback

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

# ---- TMDB posters (optional) ----
TMDB_API_KEY = os.environ.get("TMDB_API_KEY") or "81e33b129a44bb82c8779174ede456de"
_POSTER_CACHE: dict[tuple[str, int | None], str | None] = {}

def _normalize_title_for_search(t: str) -> str:
    t0 = t
    t0 = re.sub(r"\s*\((\d{4})\)\s*$", "", t0).strip()
    m = re.match(r"^(.*),\s*(The|An|A)$", t0)
    if m:
        t0 = f"{m.group(2)} {m.group(1)}"
    # remove extra parenthetical aliases
    t0 = re.split(r"\s*\([^\)]*\)\s*", t0)[0].strip() or t0
    return t0

def _fetch_poster_url(title: str | None, year: int | None) -> str | None:
    if not TMDB_API_KEY or not title:
        return None
    qtitle = _normalize_title_for_search(title)
    key = (qtitle.lower(), year)
    if key in _POSTER_CACHE:
        return _POSTER_CACHE[key]
    try:
        params = {"api_key": TMDB_API_KEY, "query": qtitle, "include_adult": False, "language": "en-US"}
        if year:
            params["year"] = int(year)
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        poster_path = None
        if results:
            for res in results:
                if res.get("poster_path"):
                    poster_path = res["poster_path"]
                    break
        if not poster_path and year:
            # retry without year if no result
            r2 = requests.get("https://api.themoviedb.org/3/search/movie", params={"api_key": TMDB_API_KEY, "query": title}, timeout=5)
            if r2.ok:
                results = r2.json().get("results", [])
                for res in results:
                    if res.get("poster_path"):
                        poster_path = res["poster_path"]
                        break
        url = f"https://image.tmdb.org/t/p/w342{poster_path}" if poster_path else None
        _POSTER_CACHE[key] = url
        return url
    except Exception:
        _POSTER_CACHE[key] = None
        return None

# ---- Best single model (default): BasicMF ----
mf_params = json.loads(Path("models/BasicMatrixFactorization.json").read_text(encoding="utf-8"))
default_model = BasicMatrixFactorization(**mf_params).fit(ratings)

# ---- Model registry (lazy-fit) ----
MODEL_PARAMS = {
    "BasicMF": json.loads(Path("models/BasicMatrixFactorization.json").read_text(encoding="utf-8")),
    "ItemCF": json.loads(Path("models/ItemBasedCF.json").read_text(encoding="utf-8")),
    "UserCF": json.loads(Path("models/UserBasedCF.json").read_text(encoding="utf-8")),
    "Genre": json.loads(Path("models/GenreBasedRecommender.json").read_text(encoding="utf-8")),
    "Demo": json.loads(Path("models/DemographicBasedRecommender.json").read_text(encoding="utf-8")),
    "SVD": json.loads(Path("models/SVDMatrixFactorization.json").read_text(encoding="utf-8")),
}
MODEL_CACHE: dict[str, object] = {"BasicMF": default_model}
ACTIVE_MODEL = "BasicMF"

def get_model(name: str):
    name = name or ACTIVE_MODEL
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    params = MODEL_PARAMS.get(name, {})
    if name == "BasicMF":
        m = BasicMatrixFactorization(**params).fit(ratings)
    elif name == "ItemCF":
        m = ItemBasedCF(**params).fit(ratings)
    elif name == "UserCF":
        m = UserBasedCF(**params).fit(ratings)
    elif name == "Genre":
        m = GenreBasedRecommender(); m.fit(ratings, pd.read_csv("data/movies_processed.csv"))
    elif name == "Demo":
        m = DemographicBasedRecommender(**params); m.fit(ratings, users_df)
    elif name == "SVD":
        m = SVDMatrixFactorization(**params).fit(ratings)
    else:
        raise ValueError("Unknown model")
    MODEL_CACHE[name] = m
    return m

# ---- Session storage (in-memory) ----
session_id_to_user = {}         # sessionId -> synthetic userId
session_id_to_ratings = {}      # sessionId -> list of dicts {movieId, rating}
synthetic_user_start = 900000

# ---- Helpers ----
def _items_from_ids(ids):
    out = []
    for mid in ids:
        mm = movie_meta.get(int(mid))
        title = mm.get("title") if mm else f"Movie {int(mid)}"
        year = mm.get("year") if mm else None
        poster = _fetch_poster_url(title, year)
        out.append({"movieId": int(mid), "title": title, "year": year, "posterUrl": poster})
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
    mode = request.args.get("mode", "random")
    size = max(1, min(100, size))
    if mode == "popular":
        ids = _popular_ids[:size]
    else:
        # random selection from full set
        if len(_popular_ids) == 0:
            ids = []
        else:
            idx = np.random.choice(_popular_ids, size=min(size, len(_popular_ids)), replace=False)
            ids = idx.tolist()
    return jsonify(_items_from_ids(ids))

@app.get("/movies/search")
def movies_search():
    q = (request.args.get("q") or "").strip()
    limit = int(request.args.get("limit", 20))
    limit = max(1, min(50, limit))
    if not q:
        return jsonify([])
    nq = _normalize_title_for_search(q).lower()
    # simple contains match over titles, ranked by popularity count
    matches = []
    for mid, meta in movie_meta.items():
        title = (meta.get("title") or "")
        tnorm = _normalize_title_for_search(str(title)).lower()
        if nq in tnorm:
            pop = int(_pop_counts.get(mid, 0)) if hasattr(_pop_counts, 'get') else 0
            matches.append((pop, mid))
    matches.sort(reverse=True)
    ids = [mid for _, mid in matches[:limit]]
    return jsonify(_items_from_ids(ids))

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
    model_name = payload.get("model") or ACTIVE_MODEL
    model = get_model(model_name)
    unknown_user = user_id not in getattr(model, 'user_id_to_idx', {})
    unknown_item = movie_id not in getattr(model, 'item_id_to_idx', {})
    pred = float(model.predict(user_id, movie_id))
    mm = movie_meta.get(movie_id)
    title = mm.get("title") if mm else f"Movie {movie_id}"
    year = mm.get("year") if mm else None
    # avoid duplicating year if title already embeds it
    if isinstance(title, str) and year and f"({year})" in title:
        pass  # keep as-is
    poster = _fetch_poster_url(title, year)
    return jsonify({
        "prediction": round(pred, 1),
        "movieId": movie_id,
        "title": title,
        "year": year,
        "unknownUser": bool(unknown_user),
        "unknownItem": bool(unknown_item),
        "model": model_name,
        "posterUrl": poster
    })

@app.get("/recommend")
def recommend():
    session_id = request.args.get("sessionId", "").strip()
    k = int(request.args.get("k", 10))
    model_name = request.args.get("model") or ACTIVE_MODEL

    if not session_id:
        return jsonify({"error": "sessionId required"}), 400

    if len(session_id_to_ratings.get(session_id, [])) == 0:
        return jsonify({"error": "no_ratings", "message": "Please rate some movies first."}), 400

    # Build temp DF with session ratings & recommend via fast ItemBasedCF
    tmp_df, user_id = _append_session_ratings(ratings, session_id)
    # Use model selection: ItemCF handles session ratings naturally; MF will fallback to default behavior
    if model_name == "ItemCF":
        params = MODEL_PARAMS.get("ItemCF", {})
        model = ItemBasedCF(**params).fit(tmp_df)
    elif model_name == "UserCF":
        params = MODEL_PARAMS.get("UserCF", {})
        model = UserBasedCF(**params).fit(tmp_df)
    else:
        model = get_model(model_name)
    recs = model.recommend(user_id, k=k, exclude_seen=True)

    out = []
    for score, mid in recs:
        mm = movie_meta.get(int(mid))
        title = mm.get("title") if mm else f"Movie {int(mid)}"
        year = mm.get("year") if mm else None
        poster = _fetch_poster_url(title, year)
        out.append({"movieId": int(mid), "score": float(round(score, 1)), "title": title, "year": year, "posterUrl": poster})
    return jsonify({"model": model_name, "items": out})

@app.get("/models")
def list_models():
    return jsonify({"active": ACTIVE_MODEL, "available": list(MODEL_PARAMS.keys())})

@app.post("/set_model")
def set_model():
    global ACTIVE_MODEL
    name = request.get_json(force=True).get("model")
    if name not in MODEL_PARAMS:
        return jsonify({"error": "unknown_model"}), 400
    ACTIVE_MODEL = name
    # warm it
    get_model(name)
    return jsonify({"active": ACTIVE_MODEL})

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
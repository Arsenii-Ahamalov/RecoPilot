# Phase 5 — Flask API + Web Demo ✅

This phase delivers a working end‑to‑end demo you can run locally: start a session, rate movies (random or by search), and get poster‑rich recommendations with switchable algorithms — all in a single page. Below is a concise recap of what was built, how to use it, and what’s next.

## Highlights
- ✅ Single‑page UI (`app/static/index.html`) with Pico.css
  - Start session, choose N random movies, rate via star widgets ⭐
  - Search by title and rate from results
  - Model selector: `BasicMF`, `ItemCF`, `UserCF`, `Genre`, `Demo`, `SVD`
  - Posters from TMDB, spinner states, and subtle fade‑in animations
- ✅ Flask backend (`app/server.py`)
  - In‑memory sessions and ratings per session
  - Movie sampling: random or popular
  - Title search with normalization and popularity ranking
  - TMDB poster integration with in‑memory cache
  - Model registry with lazy fitting and runtime switching
- ✅ Developer docs & assets
  - README demo GIF/MP4 embedded (`docs/assets/reco-demo.*`)
  - Updated run instructions and project structure

## How to run locally
```bash
python app/server.py
# open http://localhost:5000
```
Notes:
- Data is read from `data/` and model params from `models/`.
- Posters work out of the box; you can set `TMDB_API_KEY` to your own key if desired.

## API surface
- `POST /session/start` → `{ sessionId, userId }`
- `GET /movies/sample?size={int}&mode=random|popular` → list of `{ movieId, title, year, posterUrl }`
- `GET /movies/search?q={str}&limit={int}` → list of matched movies
- `POST /rate` with `{ sessionId, ratings:[{movieId, rating}] }` → persists session ratings
- `GET /recommend?sessionId={id}&k={int}&model={name}` → `{ model, items:[{movieId, title, year, posterUrl}] }`
- `GET /models` → available models; `POST /set_model` → set default
- `POST /predict` → kept for debugging (not exposed in UI)

## Performance notes
- BasicMF and SVD provide fast responses ✅
- Memory‑based CF and content‑based can be slower at request time ⚠️
  - Next: move heavy work off the request path (precompute top‑K item neighbors, prebuild content vectors, candidate generation, and short‑list scoring) and add a small LRU cache.

## Deployment status
- Attempted Render (free) → Out‑of‑memory due to 512MiB cap when using multiple workers.
- Decision: ship a high‑quality demo GIF/MP4 in README for repo visitors and keep local run instructions. Deployment can be revisited with Docker or a larger plan.

## Files touched in Phase 5
- `app/server.py`: sessions, sampling/search, TMDB posters, model registry, endpoints
- `app/static/index.html`: modern UI, stars, posters, spinners, fade‑in, search merged into rating section
- `README.md`: demo media, structure update, run instructions, Phase 5 marked complete
- `docs/assets/reco-demo.gif` and `docs/assets/reco-demo.mp4`: demo media

## Known limitations
- In‑memory sessions only (reset on restart); no DB persistence
- CF/content models not yet optimized with precomputed neighbors/candidates
- No authentication or multi‑tenant quotas
- Poster fetching depends on TMDB availability

## Next steps 🚀
- Precompute CF neighbors and content vectors; score only a small candidate set
- Add lightweight caching for recommendations
- Persist sessions/ratings (SQLite or Postgres) and restore via localStorage
- Explanations: “Because you liked X and Y” (neighbors or shared genres)
- Optional: Dockerfile + deploy to Spaces/Render with higher memory

— End of Phase 5 —



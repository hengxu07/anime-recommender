# Anime Recommendation System

A hybrid anime recommender combining content-based filtering and collaborative filtering, built with Python and deployed as a Streamlit web app.

Given an anime title, it returns similar anime via two signals:

- **Content Match** — cosine similarity on genre tags, rating, runtime, release year, and sentence embeddings of plot summaries (~6,800 IMDb titles)
- **Fans Also Watched** — item-item collaborative filtering from 42M real user ratings on MyAnimeList (267K users, 13K anime)

## How it works

**Content model**
1. Loads and cleans an IMDb anime dataset (~6,800 unique series)
2. One-hot encodes genres (weighted 3×) and normalises ratings, vote counts, runtime, and release year
3. Encodes plot summaries with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims) — captures meaning, not word overlap
4. Builds a cosine similarity matrix across the full 411-column feature matrix
5. At query time: fuzzy-matches the input title, filters by genre/votes/year, ranks by hybrid score (similarity + rating quality)

**Collaborative filtering model**
1. Reads 42M rows of MAL user ratings from `UserAnimeList.csv` (completed + scored only)
2. Builds a sparse user-item matrix with `scipy` (169 MB vs 13.8 GB dense equivalent)
3. Computes item-item cosine similarity across all 13K anime
4. At query time: looks up the anime's row in the similarity matrix and returns the top-N closest titles

## Usage

Run the web app:

```bash
streamlit run app.py
```

Or call `recommend()` directly from Python:

```python
# Basic usage
recommend("Attack on Titan")

# Tolerate typos — fuzzy search finds the closest match
recommend("Attack on Titen")

# Filter by genre
recommend("Attack on Titan", genre="Mystery")

# Only recent shows
recommend("Attack on Titan", after=2015)

# Only well-known titles (at least 1000 votes)
recommend("Spirited Away", min_votes=1000)

# Combine filters
recommend("Attack on Titan", genre="Action", after=2010, min_votes=500)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | required | Anime title (typos tolerated) |
| `n` | int | 10 | Number of results to return |
| `genre` | str | None | Filter results to this genre e.g. `"Mystery"` |
| `min_votes` | int | 50 | Exclude titles with fewer votes than this |
| `after` | int | None | Only return titles released after this year |
| `before` | int | None | Only return titles released before this year |
| `popularity_weight` | float | 0.1 | Blend similarity with rating quality (0.0 = pure similarity, 1.0 = pure rating) |

## Setup

```bash
pip install -r requirements.txt
python build_embeddings.py   # one-time: downloads model (~90 MB) and builds embeddings.npy
python build_cf.py           # one-time: reads UserAnimeList.csv (~5 GB) and builds cf_similarity.npy
```

Then run the web app:

```bash
streamlit run app.py
```

## Datasets

**Content model** — [Japanese Anime: An In-Depth IMDb Data Set](https://www.kaggle.com/datasets/lorentzyeung/all-japanese-anime-titles-in-imdb) by Lorentz Yeung. Place as `imdb_anime.csv`.

**Collaborative filtering** — MyAnimeList dataset from Kaggle. Place as `AnimeList.csv`, `UserAnimeList.csv`, and `UserList.csv`.

## Web App

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501`. Type an anime title, pick filters from the sidebar, and press Enter or click the button.

## Project Structure

```
app.py                # Streamlit web app — Content Match + Fans Also Watched tabs
recommender.py        # all model logic — load_model, recommend, load_cf_model, cf_recommend
build_embeddings.py   # one-time: compute sentence embeddings → embeddings.npy
build_cf.py           # one-time: build CF similarity matrix → cf_similarity.npy
anime_project.ipynb   # exploration notebook — data cleaning, modelling, analysis
requirements.txt      # pinned dependencies
imdb_anime.csv        # IMDb dataset (not tracked — download from Kaggle)
AnimeList.csv         # MAL anime metadata (not tracked — download from Kaggle)
UserAnimeList.csv     # MAL user ratings, ~5 GB (not tracked — download from Kaggle)
```

## Changelog

### Phase 6 — Collaborative Filtering
- Added item-item CF from 42M MAL user ratings (267K users, 13K anime)
- Sparse user-item matrix built with scipy (169 MB vs 13.8 GB dense equivalent)
- CF similarity matrix pre-computed via `build_cf.py` and saved to `cf_similarity.npy`
- Streamlit app updated with two tabs: **Content Match** (genre/plot features) and **Fans Also Watched** (audience overlap)
- Both JP and EN titles shown in CF results
- Spirited Away CF results surface the full Studio Ghibli catalogue — audience pattern content features cannot replicate

### Phase 5 — Sentence Embeddings
- Replaced TF-IDF with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions) — encodes meaning, not word overlap
- Pre-computed embeddings saved to `embeddings.npy` via `build_embeddings.py` (run once; excluded from git)
- Code Geass now ranks #3 for Neon Genesis Evangelion — previously shared zero TF-IDF vocabulary words despite identical genre/tone
- Feature matrix: 411 columns (4 numeric + 22 genres at 3× + 384 embeddings at 1×)

### Phase 4 — Streamlit Web App
- Extracted model logic from notebook into `recommender.py` (importable module with `load_model()` and `recommend()`)
- Built `app.py` Streamlit UI — search by title, genre dropdown, vote/year/popularity filters
- Model cached with `@st.cache_resource` so the similarity matrix builds once on startup (~20s), not on every query



### Phase 3 — Understanding What Shows Are About
- Added PCA visualisation (326 → 2 dimensions, 39.7% variance explained) — scatter plot confirms clusters form real spatial groups; Cluster 2 (Action/Adventure) sits isolated far right, comedy clusters group upper-left, shorts cluster at bottom
- Added K-Means clustering (K=20) — groups anime into 20 clusters; validated recommender achieves 10/10 same-cluster results for AoT, Spirited Away, and NGE
- Added `popularity_weight` parameter (default 0.1) — hybrid score blends content similarity with rating quality so highly-rated shows rank above equally-similar lower-rated ones
- Used Rating Norm (not votes) as the popularity signal to avoid audience mismatch: Pixar films have high IMDb vote counts but low anime-community ratings, so votes would surface wrong results
- Added TF-IDF on plot summaries (300 dimensions, weight 0.5×) — architecture in place for Phase 5 sentence embeddings; adds minimal signal on these short (15–35 word) summaries

### Phase 2 — Better Recommendations
- Added `precision_at_k()` — Precision@10 metric for objective, reproducible evaluation
- Added grid search across 60 weight combinations; confirmed current weights are in optimal region
- Added log-scaled runtime as a feature (weight 1×) — films now recommend films, series recommend series
- Added normalised release year as a feature (weight 1×) — era similarity creates 16-year mean gap between 1989 and 2019 show recommendations
- Baseline Precision@10: 96%

### Phase 1 — Polish & Usability
- Added fuzzy title search via `rapidfuzz` (tolerates typos and missing punctuation)
- Added genre filter, minimum votes filter, and year range filter
- Fixed genre weighting: genre columns weighted 3× so genre drives similarity, not popularity tier
- Deduplicated titles — kept the entry with the most votes per title

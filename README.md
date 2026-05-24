# Anime Recommendation System

A content-based anime recommender built with Python and pandas.

Given an anime title, it returns the most similar anime based on genre, rating, and popularity — with support for fuzzy search, genre filtering, vote thresholds, and year ranges.

## How it works

1. Loads and cleans an IMDb anime dataset (~6,800 unique series)
2. One-hot encodes genres and normalises ratings, vote counts, and runtime
3. Builds a cosine similarity matrix across all titles
4. At query time: fuzzy-matches the input title, then filters and ranks results

## Usage

Open `anime_project.ipynb` in Jupyter and run all cells. At the bottom, call `recommend()`:

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

## Setup

```bash
pip install -r requirements.txt
```

Then open `anime_project.ipynb` in Jupyter.

## Dataset

Data sourced from Kaggle: [Japanese Anime: An In-Depth IMDb Data Set](https://www.kaggle.com/datasets/lorentzyeung/all-japanese-anime-titles-in-imdb) by Lorentz Yeung.

The CSV is not included in this repo. Download it from Kaggle and place it in the project root as `imdb_anime.csv`.

## Project Structure

```
anime_project.ipynb   # main notebook — data cleaning, modelling, recommender
requirements.txt      # pinned dependencies
imdb_anime.csv        # dataset (not tracked in git — download from Kaggle)
```

## Changelog

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

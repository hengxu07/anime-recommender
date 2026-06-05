"""
One-time script: build CF similarity matrix from MAL user ratings.

Run this once (or when UserAnimeList.csv changes):
    python build_cf.py

Outputs:
    cf_similarity.npy   — (13713, 13713) item-item cosine similarity matrix
    cf_anime_meta.csv   — anime metadata with JP/EN titles
    cf_anime_enc.json   — anime_id → matrix index mapping
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ── Load and filter ratings ───────────────────────────────────────────────────
# UserAnimeList.csv is ~5GB — read in chunks and keep only completed + scored rows
print("Loading ratings (reads 5GB file in chunks, ~2 min)...")
chunks = []
for chunk in pd.read_csv('UserAnimeList.csv', chunksize=100_000,
                         usecols=['username', 'anime_id', 'my_score', 'my_status']):
    filtered = chunk[(chunk['my_status'] == 2) & (chunk['my_score'] > 0)]
    chunks.append(filtered)
ratings = pd.concat(chunks, ignore_index=True)
print(f"Filtered ratings: {len(ratings):,}")

# ── Build sparse user-item matrix ─────────────────────────────────────────────
user_enc  = {u: i for i, u in enumerate(ratings['username'].unique())}
anime_enc = {a: i for i, a in enumerate(ratings['anime_id'].unique())}

row  = ratings['username'].map(user_enc).values
col  = ratings['anime_id'].map(anime_enc).values
data = ratings['my_score'].values.astype(np.float32)

user_item = csr_matrix((data, (row, col)), shape=(len(user_enc), len(anime_enc)))
print(f"Sparse matrix: {user_item.shape}, {user_item.nnz:,} non-zeros, "
      f"{user_item.data.nbytes / 1e6:.0f} MB")

# ── Compute item-item cosine similarity ───────────────────────────────────────
# Each anime becomes a vector of user scores. Two anime are similar if the same
# users tend to rate them both highly — captures audience overlap, not content.
print("Computing CF similarity matrix (~1 min)...")
cf_similarity = cosine_similarity(user_item.T)
print(f"CF similarity matrix: {cf_similarity.shape}, "
      f"{cf_similarity.nbytes / 1e6:.0f} MB")

# ── Save artifacts ────────────────────────────────────────────────────────────
np.save('cf_similarity.npy', cf_similarity)

anime_meta = pd.read_csv('AnimeList.csv',
                         usecols=['anime_id', 'title', 'title_english', 'genre', 'score'])
anime_meta.to_csv('cf_anime_meta.csv', index=False)

with open('cf_anime_enc.json', 'w') as f:
    json.dump({str(k): v for k, v in anime_enc.items()}, f)

print("Saved: cf_similarity.npy, cf_anime_meta.csv, cf_anime_enc.json")

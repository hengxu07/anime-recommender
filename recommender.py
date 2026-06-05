import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from rapidfuzz import process


def load_model(csv_path='imdb_anime.csv'):
    """
    Load the dataset, engineer features, and build the similarity matrix.
    Returns (df_clean, feature_df2, similarity_matrix2).
    Call this once at startup and cache the result.
    """
    df = pd.read_csv(csv_path)

    # ── Clean ─────────────────────────────────────────────────────────────────
    df_clean = df[df['Episode'] == '0'].copy()
    df_clean = df_clean.drop(columns=[
        'Metascore', 'Gross', 'Stars', 'Certificate',
        'Summary', 'Episode', 'Episode Title'
    ])
    df_clean['Number of Votes'] = df_clean['Number of Votes'].str.replace(',', '').astype(float)
    df_clean['Runtime'] = df_clean['Runtime'].str.extract(r'(\d+)').astype(float)
    df_clean['Year'] = df_clean['Year'].str.extract(r'(\d{4})').astype(float)
    df_clean['User Rating'] = pd.to_numeric(df_clean['User Rating'], errors='coerce')
    df_clean = df_clean.dropna(subset=['User Rating'])
    df_clean['Runtime'] = df_clean['Runtime'].fillna(df_clean['Runtime'].median())
    df_clean['Year'] = df_clean['Year'].fillna(df_clean['Year'].median())

    # ── Normalise votes ───────────────────────────────────────────────────────
    df_clean['Votes Log'] = np.log1p(df_clean['Number of Votes'])
    df_clean['Rating Norm'] = (df_clean['User Rating'] - df_clean['User Rating'].min()) / \
                               (df_clean['User Rating'].max() - df_clean['User Rating'].min())
    df_clean['Votes Norm'] = (df_clean['Votes Log'] - df_clean['Votes Log'].min()) / \
                              (df_clean['Votes Log'].max() - df_clean['Votes Log'].min())

    # ── Deduplicate: keep the entry with the most votes per title ─────────────
    df_clean = df_clean.sort_values('Number of Votes', ascending=False)
    df_clean = df_clean.drop_duplicates(subset='Title', keep='first')
    df_clean = df_clean.reset_index(drop=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    genre_dummies  = df_clean['Genre'].str.get_dummies(sep=', ')
    rating_norm    = (df_clean['User Rating'] - df_clean['User Rating'].min()) / \
                     (df_clean['User Rating'].max() - df_clean['User Rating'].min())
    votes_norm_raw = df_clean['Votes Norm'] / df_clean['Votes Norm'].max()

    runtime_log  = np.log1p(df_clean['Runtime'])
    runtime_norm = (runtime_log - runtime_log.min()) / (runtime_log.max() - runtime_log.min())

    year_norm = (df_clean['Year'] - df_clean['Year'].min()) / \
                (df_clean['Year'].max() - df_clean['Year'].min())

    # ── Sentence embeddings ───────────────────────────────────────────────────
    # Pre-computed by build_embeddings.py — load from disk instead of recomputing.
    # Shape: (n_anime, 384). Each row is an L2-normalised 384-dim vector encoding
    # the *meaning* of the plot summary, not just word overlap like TF-IDF did.
    embeddings = np.load('embeddings.npy')
    embedding_weight = 1.0
    embedding_df = pd.DataFrame(
        embeddings * embedding_weight,
        columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
    )

    # ── Build feature matrix ──────────────────────────────────────────────────
    # Weights: genres 3x (dominant signal), numeric features 1x, embeddings 1x
    feature_df2 = pd.concat([
        df_clean[['Title']].reset_index(drop=True),
        (rating_norm    * 1.0).rename('Rating Norm').reset_index(drop=True),
        (votes_norm_raw * 1.0).rename('Votes Norm').reset_index(drop=True),
        (runtime_norm   * 1.0).rename('Runtime Norm').reset_index(drop=True),
        (year_norm      * 1.0).rename('Year Norm').reset_index(drop=True),
        (genre_dummies  * 3.0).reset_index(drop=True),
        embedding_df.reset_index(drop=True),
    ], axis=1)

    feature_matrix2    = feature_df2.drop(columns=['Title']).values
    similarity_matrix2 = cosine_similarity(feature_matrix2)

    # ── K-Means clustering ────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(feature_matrix2)

    return df_clean, feature_df2, similarity_matrix2


def recommend(title, df_clean, feature_df2, similarity_matrix2,
              n=10, genre=None, min_votes=50, after=None, before=None,
              popularity_weight=0.1):
    """
    Return the top-n most similar anime to `title`.

    Parameters
    ----------
    title             : anime title (typos tolerated via fuzzy search)
    df_clean          : cleaned DataFrame from load_model()
    feature_df2       : feature DataFrame from load_model()
    similarity_matrix2: cosine similarity matrix from load_model()
    n                 : number of results (default 10)
    genre             : filter to this genre e.g. "Mystery"
    min_votes         : exclude titles with fewer votes (default 50)
    after             : only titles released after this year
    before            : only titles released before this year
    popularity_weight : blend similarity with rating quality (0.0–1.0, default 0.1)

    Returns
    -------
    (pd.DataFrame, matched_title) on success
    (None, error_message) if no match or no results
    """
    all_titles = feature_df2['Title'].tolist()

    exact = [t for t in all_titles if t.lower() == title.lower()]
    if exact:
        matched_title = exact[0]
        fuzzy_note = None
    else:
        result = process.extractOne(title, all_titles)
        if result is None:
            return None, f"No match found for '{title}'."
        matched_title, score, _ = result
        if score < 80:
            return None, (
                f"No confident match found for '{title}'. "
                f"Closest: '{matched_title}' ({score:.0f}/100)."
            )
        fuzzy_note = matched_title

    idx = feature_df2[feature_df2['Title'] == matched_title].index[0]
    all_scores = sorted(
        enumerate(similarity_matrix2[idx]), key=lambda x: x[1], reverse=True
    )
    candidates = [(i, s) for i, s in all_scores if i != idx]

    results = []
    for i, sim_score in candidates:
        row = df_clean.iloc[i]
        if row['Number of Votes'] < min_votes:
            continue
        if genre and genre.lower() not in row['Genre'].lower():
            continue
        if after and row['Year'] < after:
            continue
        if before and row['Year'] > before:
            continue
        # Skip sequels/seasons — result title contains the query title as a substring
        if matched_title.lower() in row['Title'].lower():
            continue

        # Hybrid score: blend content similarity with rating quality.
        # Uses Rating Norm (not Votes Norm) to avoid audience mismatch —
        # IMDb vote counts are skewed toward mainstream Western films.
        popularity    = row['Rating Norm']
        hybrid_score  = (1 - popularity_weight) * sim_score + popularity_weight * popularity

        results.append({
            'Title':      row['Title'],
            'Score':      round(hybrid_score, 3),
            'Similarity': round(sim_score, 3),
            'Rating':     round(row['User Rating'], 1),
            'Votes':      int(row['Number of Votes']),
            'Year':       int(row['Year']),
            'Genres':     row['Genre'],
        })

    results.sort(key=lambda x: x['Score'], reverse=True)
    results = results[:n]

    if not results:
        return None, "No results matched your filters. Try relaxing genre, year, or min_votes."

    return pd.DataFrame(results), fuzzy_note


def profile_recommend(user_ratings, df_clean, feature_df2, similarity_matrix2,
                      n=10, min_votes=50):
    """
    Personalised content-based recommendations from a watch history.

    Parameters
    ----------
    user_ratings : dict of {title: rating} where rating is 1–10
    Returns (pd.DataFrame, unmatched_titles) on success, (None, error_message) on failure.
    """
    all_titles = feature_df2['Title'].tolist()
    max_rating = max(user_ratings.values()) if user_ratings else 1

    resolved   = {}   # matched_title → (row_index, weight)
    unmatched  = []

    for title, rating in user_ratings.items():
        exact = [t for t in all_titles if t.lower() == title.lower()]
        if exact:
            matched = exact[0]
        else:
            result = process.extractOne(title, all_titles)
            if result is None or result[1] < 80:
                unmatched.append(title)
                continue
            matched = result[0]
        idx    = feature_df2[feature_df2['Title'] == matched].index[0]
        weight = rating / max_rating   # normalise so best-rated show = 1.0
        resolved[matched] = (idx, weight)

    if not resolved:
        return None, "None of the titles could be matched. Check for typos."

    # Build a weighted aggregate of each watched anime's similarity row.
    # Higher-rated shows in your history pull the profile harder.
    watched_indices = {idx for idx, _ in resolved.values()}
    agg = np.zeros(len(df_clean))
    for _, (idx, weight) in resolved.items():
        agg += similarity_matrix2[idx] * weight

    results = []
    for i in agg.argsort()[::-1]:
        if i in watched_indices:
            continue
        row = df_clean.iloc[i]
        if row['Number of Votes'] < min_votes:
            continue
        results.append({
            'Title':  row['Title'],
            'Score':  round(float(agg[i]), 3),
            'Rating': round(row['User Rating'], 1),
            'Votes':  int(row['Number of Votes']),
            'Year':   int(row['Year']),
            'Genres': row['Genre'],
        })
        if len(results) >= n:
            break

    if not results:
        return None, "No results found. Try lowering min_votes."

    return pd.DataFrame(results), unmatched


def cf_profile_recommend(user_ratings, cf_sim, anime_enc, idx_to_anime,
                         id_to_meta, title_to_id, n=10):
    """
    Personalised CF recommendations from a watch history.

    Parameters
    ----------
    user_ratings : dict of {title: rating} where rating is 1–10
    Returns (pd.DataFrame, unmatched_titles) on success, (None, error_message) on failure.
    """
    all_titles = list(title_to_id.keys())
    max_rating = max(user_ratings.values()) if user_ratings else 1

    resolved  = {}   # matched_title → (anime_id, row_index, weight)
    unmatched = []

    for title, rating in user_ratings.items():
        exact = [t for t in all_titles if t.lower() == title.lower()]
        if exact:
            matched = exact[0]
        else:
            result = process.extractOne(title, all_titles)
            if result is None or result[1] < 80:
                unmatched.append(title)
                continue
            matched = result[0]
        anime_id = title_to_id[matched]
        if anime_id not in anime_enc:
            unmatched.append(title)
            continue
        idx    = anime_enc[anime_id]
        weight = rating / max_rating
        resolved[matched] = (anime_id, idx, weight)

    if not resolved:
        return None, "None of the titles could be matched. Check for typos."

    watched_ids = {aid for aid, _, _ in resolved.values()}
    agg = np.zeros(cf_sim.shape[0])
    for _, (_, idx, weight) in resolved.items():
        agg += cf_sim[idx] * weight

    results = []
    for i in agg.argsort()[::-1]:
        aid = idx_to_anime[i]
        if aid in watched_ids:
            continue
        if aid not in id_to_meta.index:
            continue
        meta     = id_to_meta.loc[aid]
        en_title = meta['title_english'] if pd.notna(meta['title_english']) else ''
        results.append({
            'Title JP': meta['title'],
            'Title EN': en_title,
            'CF Score': round(float(agg[i]), 3),
            'MAL Score': meta['score'],
            'Genre':    meta['genre'],
        })
        if len(results) >= n:
            break

    if not results:
        return None, "No results found."

    return pd.DataFrame(results), unmatched


def load_cf_model(meta_path='cf_anime_meta.csv',
                  sim_path='cf_similarity.npy',
                  enc_path='cf_anime_enc.json'):
    """
    Load pre-computed CF artifacts from disk.
    Returns (cf_sim, anime_enc, idx_to_anime, id_to_meta, title_to_id).
    Call once at startup and cache the result.
    """
    cf_meta = pd.read_csv(meta_path)
    cf_sim  = np.load(sim_path)

    with open(enc_path) as f:
        anime_enc = {int(k): v for k, v in json.load(f).items()}
    idx_to_anime = {v: k for k, v in anime_enc.items()}

    id_to_meta = cf_meta.set_index('anime_id')

    # Build title → anime_id lookup covering both JP and EN titles
    title_to_id = cf_meta.set_index('title')['anime_id'].to_dict()
    en = cf_meta.dropna(subset=['title_english'])
    title_to_id.update(en.set_index('title_english')['anime_id'].to_dict())

    return cf_sim, anime_enc, idx_to_anime, id_to_meta, title_to_id


def cf_recommend(title, cf_sim, anime_enc, idx_to_anime, id_to_meta, title_to_id, n=10):
    """
    Return the top-n anime whose audiences overlap most with `title`.
    Searches both Japanese (romaji) and English titles.

    Returns
    -------
    (pd.DataFrame, matched_title) on success
    (None, error_message) on failure
    """
    all_titles = list(title_to_id.keys())

    exact = [t for t in all_titles if t.lower() == title.lower()]
    if exact:
        matched_title = exact[0]
        fuzzy_note = None
    else:
        result = process.extractOne(title, all_titles)
        if result is None:
            return None, f"No match found for '{title}'."
        matched_title, score, _ = result
        if score < 80:
            return None, (
                f"No confident match for '{title}'. "
                f"Closest: '{matched_title}' ({score:.0f}/100)."
            )
        fuzzy_note = matched_title

    anime_id = title_to_id[matched_title]
    if anime_id not in anime_enc:
        return None, f"'{matched_title}' has no ratings data."

    idx    = anime_enc[anime_id]
    scores = cf_sim[idx]
    top_idx = scores.argsort()[::-1][1:n + 1]

    results = []
    for i in top_idx:
        aid = idx_to_anime[i]
        if aid not in id_to_meta.index:
            continue
        meta = id_to_meta.loc[aid]
        en_title = meta['title_english'] if pd.notna(meta['title_english']) else ''
        results.append({
            'Title JP':      meta['title'],
            'Title EN':      en_title,
            'CF Similarity': round(float(scores[i]), 3),
            'MAL Score':     meta['score'],
            'Genre':         meta['genre'],
        })

    if not results:
        return None, "No results found."

    return pd.DataFrame(results), fuzzy_note

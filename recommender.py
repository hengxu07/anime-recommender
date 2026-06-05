import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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

    # ── TF-IDF on plot summaries ──────────────────────────────────────────────
    # Note: summaries are short (15-35 words), so TF-IDF adds minimal signal.
    # Architecture is ready for Phase 5 where sentence embeddings replace it.
    summary_map = df.drop_duplicates(subset='Title', keep='first').set_index('Title')['Summary']
    df_clean['Summary'] = df_clean['Title'].map(summary_map).fillna('')

    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_clean['Summary']).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_matrix * 0.5,
        columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    )

    # ── Build feature matrix ──────────────────────────────────────────────────
    # Weights: genres 3x (dominant signal), numeric features 1x, TF-IDF 0.5x
    feature_df2 = pd.concat([
        df_clean[['Title']].reset_index(drop=True),
        (rating_norm    * 1.0).rename('Rating Norm').reset_index(drop=True),
        (votes_norm_raw * 1.0).rename('Votes Norm').reset_index(drop=True),
        (runtime_norm   * 1.0).rename('Runtime Norm').reset_index(drop=True),
        (year_norm      * 1.0).rename('Year Norm').reset_index(drop=True),
        (genre_dummies  * 3.0).reset_index(drop=True),
        tfidf_df.reset_index(drop=True),
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

"""
Unit tests for recommend().

Uses a tiny in-memory fixture (5 anime) so tests run in milliseconds
without loading the real dataset.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from recommender import recommend


@pytest.fixture
def small_model():
    """Minimal in-memory model — no file I/O, deterministic."""
    np.random.seed(42)

    df_clean = pd.DataFrame({
        'Title': [
            'Attack on Titan',
            'Attack on Titan Season 2',
            'Fullmetal Alchemist: Brotherhood',
            'Spirited Away',
            'One Piece',
        ],
        'Genre': [
            'Animation, Action, Drama',
            'Animation, Action, Drama',
            'Animation, Action, Drama',
            'Animation, Adventure, Family',
            'Animation, Action, Adventure',
        ],
        'User Rating':     [9.0,     8.8,     9.1,     8.6,     8.7],
        'Number of Votes': [500_000, 100_000, 400_000, 200_000, 150_000],
        'Year':            [2013.0,  2017.0,  2009.0,  2001.0,  1999.0],
        'Rating Norm':     [0.90,    0.80,    1.00,    0.70,    0.75],
    })

    features = np.random.rand(5, 10)
    feature_df2 = pd.DataFrame(features, columns=[f'f{i}' for i in range(10)])
    feature_df2.insert(0, 'Title', df_clean['Title'].values)
    sim = cosine_similarity(features)

    return df_clean, feature_df2, sim


# ── Core behaviour ────────────────────────────────────────────────────────────

def test_returns_requested_n(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim, n=2)
    assert len(result) == 2


def test_result_columns_present(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim)
    assert {'Title', 'Score', 'Similarity', 'Rating', 'Votes', 'Year', 'Genres'}.issubset(result.columns)


def test_scores_descending(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim)
    assert result['Score'].is_monotonic_decreasing


# ── Sequel filter ─────────────────────────────────────────────────────────────

def test_excludes_sequels(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim)
    assert 'Attack on Titan Season 2' not in result['Title'].values


# ── Filters ───────────────────────────────────────────────────────────────────

def test_genre_filter(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim, genre='Family')
    assert all('Family' in g for g in result['Genres'])


def test_min_votes_filter(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim, min_votes=300_000)
    assert all(result['Votes'] >= 300_000)


def test_year_after_filter(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim, after=2005)
    assert all(result['Year'] >= 2005)


def test_year_before_filter(small_model):
    df_clean, feature_df2, sim = small_model
    result, _ = recommend('Attack on Titan', df_clean, feature_df2, sim, before=2010)
    assert all(result['Year'] <= 2010)


# ── Fuzzy matching and error cases ────────────────────────────────────────────

def test_fuzzy_match_returns_results(small_model):
    df_clean, feature_df2, sim = small_model
    result, note = recommend('Attack on Titann', df_clean, feature_df2, sim)
    assert result is not None
    assert note == 'Attack on Titan'


def test_no_match_returns_none(small_model):
    df_clean, feature_df2, sim = small_model
    result, msg = recommend('xyzzy_no_such_anime', df_clean, feature_df2, sim)
    assert result is None
    assert isinstance(msg, str)


def test_impossible_filter_returns_none(small_model):
    # No anime in the fixture was released after 2020
    df_clean, feature_df2, sim = small_model
    result, msg = recommend('Attack on Titan', df_clean, feature_df2, sim, after=2020)
    assert result is None
    assert isinstance(msg, str)

# Anime Recommendation System

A content-based anime recommender built with Python and pandas.

Given an anime title, it returns the 10 most similar anime based on genre, rating, and popularity.

## How it works

1. Loads and cleans an IMDb anime dataset (~7,700 series)
2. Encodes genres as numeric features
3. Normalizes ratings and vote counts
4. Uses cosine similarity to find the closest matches

## Usage

Open `anime_project.ipynb` in Jupyter and run all cells. At the bottom, call:

```python
recommend("Attack on Titan")
recommend("Spirited Away")
```

## Dataset

Data sourced from Kaggle: [Japanese Anime: An In-Depth IMDb Data Set](https://www.kaggle.com/datasets/lorentzyeung/all-japanese-anime-titles-in-imdb) by Lorentz Yeung.

The CSV is not included in this repo. Download it from Kaggle and place it in the project root as `imdb_anime.csv`.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install with: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`

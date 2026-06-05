"""
One-time script: compute sentence embeddings for all anime summaries and save to disk.

Run this once (or whenever imdb_anime.csv changes):
    python build_embeddings.py

Output: embeddings.npy  — shape (6808, 384), one 384-dim vector per anime
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load and clean data (same steps as recommender.py)
df = pd.read_csv('imdb_anime.csv')
df_clean = df[df['Episode'] == '0'].copy()
df_clean['Number of Votes'] = df_clean['Number of Votes'].str.replace(',', '').astype(float)
df_clean['User Rating'] = pd.to_numeric(df_clean['User Rating'], errors='coerce')
df_clean = df_clean.dropna(subset=['User Rating'])
df_clean = df_clean.sort_values('Number of Votes', ascending=False)
df_clean = df_clean.drop_duplicates(subset='Title', keep='first')
df_clean = df_clean.reset_index(drop=True)

summary_map = df.drop_duplicates(subset='Title', keep='first').set_index('Title')['Summary']
df_clean['Summary'] = df_clean['Title'].map(summary_map).fillna('')

summaries = df_clean['Summary'].tolist()
print(f"Encoding {len(summaries)} summaries...")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(summaries, show_progress_bar=True, batch_size=64)

np.save('embeddings.npy', embeddings)
print(f"Saved embeddings.npy — shape: {embeddings.shape}")

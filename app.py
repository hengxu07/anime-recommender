import streamlit as st
from recommender import load_model, recommend, load_cf_model, cf_recommend

st.set_page_config(page_title="Anime Recommender",
                   page_icon="🎌", layout="wide")

st.title("Anime Recommender")
st.caption("Find anime similar to one you already love.")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_cf_model():
    try:
        return load_cf_model()
    except FileNotFoundError:
        return None


with st.spinner("Loading model (one-time, ~20 seconds)..."):
    df_clean, feature_df2, similarity_matrix2 = get_model()

cf_model = get_cf_model()
cf_available = cf_model is not None
if cf_available:
    cf_sim, cf_anime_enc, cf_idx_to_anime, cf_id_to_meta, cf_title_to_id = cf_model

all_genres = sorted({
    g.strip()
    for genres in df_clean['Genre'].dropna()
    for g in genres.split(', ')
    if g.strip()
})


# ── Sidebar: filters (apply to content tab only) ──────────────────────────────
with st.sidebar:
    st.header("Filters")
    st.caption("Applied to Content Match tab only.")
    n = st.slider("Number of results", min_value=5,
                  max_value=25, value=10, step=5)
    genre_choice = st.selectbox("Genre filter", options=["(any)"] + all_genres)
    genre = None if genre_choice == "(any)" else genre_choice
    min_votes = st.number_input(
        "Minimum votes", min_value=0, value=50, step=50)
    col1, col2 = st.columns(2)
    after = col1.number_input(
        "Released after",  min_value=1900, max_value=2024, value=1900, step=1)
    before = col2.number_input(
        "Released before", min_value=1900, max_value=2024, value=2024, step=1)
    popularity_weight = st.slider(
        "Popularity weight",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="0 = pure content similarity, 1 = pure rating quality"
    )


# ── Main: search ──────────────────────────────────────────────────────────────
with st.form("search_form"):
    title = st.text_input("Enter an anime title",
                          placeholder="e.g. Attack on Titan")
    search = st.form_submit_button("Find recommendations", type="primary")

if search:
    if not title.strip():
        st.warning("Please enter an anime title.")
    else:
        tab1, tab2 = st.tabs(["Content Match", "Fans Also Watched"])

        # ── Tab 1: content-based ──────────────────────────────────────────────
        with tab1:
            result, note = recommend(
                title,
                df_clean, feature_df2, similarity_matrix2,
                n=n,
                genre=genre,
                min_votes=int(min_votes),
                after=int(after) if after > 1900 else None,
                before=int(before) if before < 2024 else None,
                popularity_weight=float(popularity_weight),
            )

            if result is None:
                st.error(note)
            else:
                if note:
                    st.info(
                        f'Showing results for **{note}** (fuzzy match for "{title}")')
                else:
                    st.success(f'Showing results for **{title}**')

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Score":      st.column_config.NumberColumn(format="%.3f"),
                        "Similarity": st.column_config.NumberColumn(format="%.3f"),
                        "Rating":     st.column_config.NumberColumn(format="%.1f"),
                        "Votes":      st.column_config.NumberColumn(format="%d"),
                    }
                )

        # ── Tab 2: collaborative filtering ────────────────────────────────────
        with tab2:
            if not cf_available:
                st.info(
                    "Collaborative filtering requires the MAL dataset. "
                    "Run `python build_cf.py` locally to enable this tab."
                )
            else:
                cf_result, cf_note = cf_recommend(
                    title,
                    cf_sim, cf_anime_enc, cf_idx_to_anime,
                    cf_id_to_meta, cf_title_to_id,
                    n=n,
                )

                if cf_result is None:
                    st.error(cf_note)
                else:
                    if cf_note:
                        st.info(
                            f'Showing results for **{cf_note}** (fuzzy match for "{title}")')
                    else:
                        st.success(f'Showing results for **{title}**')

                    st.dataframe(
                        cf_result,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "CF Similarity": st.column_config.NumberColumn(format="%.3f"),
                            "MAL Score":     st.column_config.NumberColumn(format="%.2f"),
                        }
                    )

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
from collections import defaultdict


# ----------------------------
# TMDB API Key
# ----------------------------
TMDB_API_KEY = "c00fe4d0268482539a6645050108d7e2"


# ----------------------------
# Load PKL Files
# ----------------------------
@st.cache_resource
def load_all_pkls():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def load_pkl(filename):
        path = os.path.join(BASE_DIR, filename)
        with open(path, "rb") as f:
            return pickle.load(f)

    movies = load_pkl("movies.pkl")
    ratings = load_pkl("train_ratings.pkl")
    user_item_matrix = load_pkl("user_item_matrix.pkl")
    item_sim = load_pkl("item_similarity_reduced.pkl")
    content_sim = load_pkl("content_similarity_reduced.pkl")
    popularity = load_pkl("movie_popularity.pkl")

    return movies, ratings, user_item_matrix, item_sim, content_sim, popularity


# ----------------------------
# Helper: Similarity List/Dict Fix
# ----------------------------
def get_similar_movies(sim_matrix, movie_id):
    """
    sim_matrix can store:
    dict -> {movieId: {simId: score}}
    dict -> {movieId: [(simId, score), (simId, score)]}
    """

    if movie_id not in sim_matrix:
        return []

    sims = sim_matrix[movie_id]

    if isinstance(sims, dict):
        return list(sims.items())

    if isinstance(sims, list):
        return sims

    return []


# ----------------------------
# Get Poster from TMDB
# ----------------------------
@st.cache_data
def get_movie_poster(title):
    try:
        # remove year like (1999)
        clean_title = title.split("(")[0].strip()

        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
        res = requests.get(url).json()

        if "results" in res and len(res["results"]) > 0:
            poster_path = res["results"][0].get("poster_path", None)
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None

    return None



# ----------------------------
# Collaborative Filtering
# ----------------------------
def recommend_cf(user_id, user_item_matrix, item_sim_matrix):
    if user_id not in user_item_matrix.index:
        return {}

    user_ratings = user_item_matrix.loc[user_id].fillna(0)
    rated_movies = user_ratings[user_ratings > 0].index.tolist()

    if len(rated_movies) == 0:
        return {}

    scores = defaultdict(float)

    for movie in rated_movies:
        similar_list = get_similar_movies(item_sim_matrix, movie)

        for sim_movie_id, sim_score in similar_list:
            sim_movie_id = int(sim_movie_id)
            sim_score = float(sim_score)

            if sim_movie_id not in rated_movies:
                scores[sim_movie_id] += sim_score

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ----------------------------
# Content Based Filtering
# ----------------------------
def recommend_content(user_id, ratings, content_sim_matrix, threshold=3.5):
    liked_movies = ratings[
        (ratings["userId"] == user_id) & (ratings["rating"] >= threshold)
    ]["movieId"].tolist()

    if len(liked_movies) == 0:
        return {}

    content_scores = defaultdict(float)

    for movie in liked_movies:
        similar_list = get_similar_movies(content_sim_matrix, movie)

        for sim_movie_id, sim_score in similar_list:
            sim_movie_id = int(sim_movie_id)
            sim_score = float(sim_score)

            if sim_movie_id not in liked_movies:
                content_scores[sim_movie_id] += sim_score

    return dict(sorted(content_scores.items(), key=lambda x: x[1], reverse=True))


# ----------------------------
# Popularity Fallback
# ----------------------------
def recommend_popular(popularity, k=10):
    if isinstance(popularity, pd.DataFrame):
        if "movieId" in popularity.columns:
            return popularity["movieId"].head(k).tolist()

    if isinstance(popularity, list):
        return popularity[:k]

    if isinstance(popularity, dict):
        return list(popularity.keys())[:k]

    return []


# ----------------------------
# Hybrid Recommendation
# ----------------------------
def hybrid_recommend(user_id, user_item_matrix, item_sim, content_sim, ratings, popularity,
                     k=10, alpha=0.7, min_rating=3.5):

    cf_scores = recommend_cf(user_id, user_item_matrix, item_sim)
    content_scores = recommend_content(user_id, ratings, content_sim, threshold=min_rating)

    if len(cf_scores) == 0 and len(content_scores) == 0:
        return recommend_popular(popularity, k=k)

    max_cf = max(cf_scores.values()) if len(cf_scores) > 0 else 1
    max_content = max(content_scores.values()) if len(content_scores) > 0 else 1

    final_scores = defaultdict(float)

    all_movies = set(cf_scores.keys()).union(set(content_scores.keys()))

    for movie_id in all_movies:
        cf_part = cf_scores.get(movie_id, 0) / max_cf
        cont_part = content_scores.get(movie_id, 0) / max_content
        final_scores[movie_id] = alpha * cf_part + (1 - alpha) * cont_part

    final_scores = dict(sorted(final_scores.items(), key=lambda x: x[1], reverse=True))

    return list(final_scores.keys())[:k]


# ----------------------------
# Explainability
# ----------------------------
def explain_recommendation(user_id, rec_movie_id, movies, ratings, top_n=2):

    user_history = ratings[ratings["userId"] == user_id]

    if user_history.empty:
        return "Recommended because it is popular among users."

    # take user top rated movies
    top_movies = user_history.sort_values("rating", ascending=False).head(10)["movieId"].tolist()

    rec_row = movies[movies["movieId"] == rec_movie_id]
    if rec_row.empty:
        return "Recommended based on similarity with your preferences."

    rec_row = rec_row.iloc[0]

    rec_tags = set(str(rec_row.get("all_tags", "")).lower().split())
    rec_genres = set(str(rec_row.get("genres", "")).lower().split("|"))

    best_matches = []

    for mid in top_movies:
        row = movies[movies["movieId"] == mid]
        if row.empty:
            continue

        row = row.iloc[0]

        title = row.get("title", "Unknown Movie")
        tags = set(str(row.get("all_tags", "")).lower().split())
        genres = set(str(row.get("genres", "")).lower().split("|"))

        common_tags = rec_tags.intersection(tags)
        common_genres = rec_genres.intersection(genres)

        score = len(common_tags) + len(common_genres)

        if score > 0:
            best_matches.append((title, common_tags, common_genres, score))

    if len(best_matches) == 0:
        return "Recommended because it matches your viewing profile."

    # sort by max overlap
    best_matches.sort(key=lambda x: x[3], reverse=True)

    # take top N movies for explanation
    chosen = best_matches[:top_n]

    liked_titles = [x[0] for x in chosen]

    # merge tags and genres
    combined_tags = set()
    combined_genres = set()

    for _, tags, genres, _ in chosen:
        combined_tags.update(tags)
        combined_genres.update(genres)

    # keep only few tags for readability
    combined_tags = list(combined_tags)[:4]
    combined_genres = list(combined_genres)[:3]

    tags_text = ", ".join([f"'{t}'" for t in combined_tags]) if combined_tags else ""
    genres_text = ", ".join([f"'{g}'" for g in combined_genres]) if combined_genres else ""

    movies_text = " and ".join(liked_titles[:2])

    if tags_text and genres_text:
        return f"Because you liked {movies_text}, which share the genres {genres_text} and tags {tags_text}."
    elif genres_text:
        return f"Because you liked {movies_text}, which share the genres {genres_text}."
    elif tags_text:
        return f"Because you liked {movies_text}, which share the tags {tags_text}."

    return f"Because you liked {movies_text}, and this movie matches similar preferences."



# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="ReelSense | Premium Movie Discovery", layout="wide", page_icon="🎬")

    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        
        <style>
        :root {
            --primary: #FF4B4B;
            --primary-glow: rgba(255, 75, 75, 0.4);
            --bg-dark: #0E1117;
            --card-bg: rgba(255, 255, 255, 0.04);
            --text-main: #FFFFFF;
            --text-dim: #A0A0A0;
        }

        .stApp {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at 20% 20%, #1a1e2e 0%, #0e1117 100%);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] {
            background-color: rgba(15, 18, 25, 0.8) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        .hero-container {
            text-align: center;
            padding: 80px 0px 60px 0px;
        }
        
        .main-title {
            font-size: 7rem;
            font-weight: 900;
            background: linear-gradient(to right, #FFFFFF, #FF4B4B, #FF8E8E, #FFFFFF);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            letter-spacing: -5px;
            line-height: 0.9;
            filter: drop-shadow(0 0 20px rgba(255, 75, 75, 0.25));
            animation: titleGradient 6s linear infinite;
        }

        @keyframes titleGradient {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }
        
        .sub-title {
            font-size: 1.1rem;
            color: var(--text-dim);
            font-weight: 600;
            letter-spacing: 6px;
            text-transform: uppercase;
            margin-bottom: 40px;
            opacity: 0.9;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            padding: 10px 0;
        }

        .movie-card {
            position: relative;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 24px;
            padding: 16px;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            margin-bottom: 25px;
        }

        .movie-card:hover {
            transform: translateY(-12px) scale(1.02);
            border: 1px solid rgba(255, 75, 75, 0.4);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 20px var(--primary-glow);
            background: rgba(255, 255, 255, 0.08);
        }

        .rank-badge {
            position: absolute;
            top: 25px;
            left: 25px;
            background: var(--primary);
            color: white;
            padding: 6px 14px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 800;
            z-index: 10;
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
            border: 2px solid rgba(255,255,255,0.2);
        }

        .poster-img {
            width: 100%;
            aspect-ratio: 2/3;
            object-fit: cover;
            border-radius: 18px;
            transition: transform 0.5s ease;
        }

        .movie-card:hover .poster-img {
            transform: scale(1.03);
        }

        .movie-title {
            font-weight: 700;
            font-size: 1.15rem;
            margin-top: 18px;
            margin-bottom: 6px;
            line-height: 1.2;
            color: white;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            height: 2.8rem;
        }

        .movie-genre {
            font-size: 0.75rem;
            color: var(--primary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            opacity: 0.9;
        }

        .explain-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 14px;
            padding: 12px;
            font-size: 0.8rem;
            color: #D1D5DB;
            margin-top: auto;
            border: 1px solid rgba(255, 255, 255, 0.04);
            line-height: 1.4;
        }
        
        .explain-box-label {
            display: block;
            font-size: 0.6rem;
            font-weight: 800;
            letter-spacing: 1.2px;
            color: var(--text-dim);
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        .no-poster {
            width: 100%;
            aspect-ratio: 2/3;
            border-radius: 18px;
            background: linear-gradient(135deg, #2A2F3A, #1C1F26);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-size: 2rem;
            color: rgba(255, 255, 255, 0.1);
            text-align: center;
            border: 1px dashed rgba(255,255,255,0.1);
        }

        div.stButton > button {
            background: linear-gradient(135deg, #FF4B4B 0%, #D42D2D 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100% !important;
            box-shadow: 0 10px 20px rgba(255, 75, 75, 0.2) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 30px rgba(255, 75, 75, 0.4) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero-container">
            <h1 class="main-title">ReelSense</h1>
            <p class="sub-title">Premium Movie Discovery Engine</p>
        </div>
    """, unsafe_allow_html=True)

    movies, ratings, user_item_matrix, item_sim, content_sim, popularity = load_all_pkls()

    st.sidebar.title("⚙️ Personalize")
    user_id = st.sidebar.number_input("Experience for User ID", min_value=1, value=1)
    top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    alpha = st.sidebar.slider("Hybrid Weight (CF vs Content)", 0.0, 1.0, 0.7)
    min_rating = st.sidebar.slider("Min Rating Considered as Liked", 1.0, 5.0, 3.5)
    
    if st.sidebar.button("✨ CURATE MY LIST"):
        with st.spinner("🚀 Curating your personalized movie experience..."):
            recs = hybrid_recommend(
                user_id=user_id,
                user_item_matrix=user_item_matrix,
                item_sim=item_sim,
                content_sim=content_sim,
                ratings=ratings,
                popularity=popularity,
                k=top_k,
                alpha=alpha,
                min_rating=min_rating
            )

        if len(recs) == 0:
            st.error("❌ No recommendations found.")
            return

        st.markdown(f"<h3 style='margin-bottom: 30px; border-left: 4px solid #FF4B4B; padding-left: 15px;'>Top {top_k} Selected for You</h3>", unsafe_allow_html=True)
        cols = st.columns(5)

        for idx, movie_id in enumerate(recs):
            movie_row = movies[movies["movieId"] == movie_id]
            if movie_row.empty: continue
            movie_row = movie_row.iloc[0]
            title = movie_row.get("title", "Unknown Title")
            genres = movie_row.get("genres", "Unknown Genres").replace("|", " • ")
            poster_url = get_movie_poster(title)
            explanation = explain_recommendation(user_id, movie_id, movies, ratings)

            with cols[idx % 5]:
                poster_html = f'<img src="{poster_url}" class="poster-img">' if poster_url else '<div class="no-poster">🎬</div>'
                st.markdown(f"""
                <div class="movie-card">
                    <div class="rank-badge">#{idx+1}</div>
                    {poster_html}
                    <div class="movie-title">{title}</div>
                    <div class="movie-genre">{genres}</div>
                    <div class="explain-box">
                        <span class="explain-box-label">REELSENSE INTELLIGENCE</span>
                        {explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()

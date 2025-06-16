import streamlit as st
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from functions import (
    score_movies_item_based,
    score_movies_svd,
    recommend_hybrid,
    get_weighted_recommendations,
)

# Load precomputed assets
with open("streamlit_files/recommender_assets.pkl", "rb") as f:
    assets = pickle.load(f)

train_matrix = assets["train_matrix"]
sim_item = assets["sim_item"]
sim_user = assets["sim_user"]
user_factors = assets["user_factors"]
item_factors = assets["item_factors"]
df = assets["df"]
genres = assets["genres"]

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.markdown("# 🍿 Welcome to CineMatch")

auth_choice = st.radio("Are you logging in or signing up?", ["Login", "Sign Up"], horizontal=True)

top_n = 3 

if auth_choice == "Login":
    user_id = st.number_input("Enter your User ID (0–610)", min_value=0, max_value=610, step=1)
    selected_genre = st.selectbox("Choose a Genre (optional)", ["None"] + genres)
    if st.button("Get My Recommendations"):
        genre = None if selected_genre == "None" else selected_genre
        recs = recommend_hybrid(
            userId=user_id,
            train_matrix=train_matrix,
            sim_item=sim_item,
            user_factors=user_factors,
            item_factors=item_factors,
            dataframe=df,
            top_n=top_n,
            filter_genre=genre
        )

        # Fallback logic: top off with weighted recommendations if needed
        if len(recs) < top_n:
            st.warning("Personalized list is short — adding top-rated picks.")
            fallback_recs = get_weighted_recommendations(
            df,
            genre=genre,
            top_n=top_n
        )
            
            # Avoid duplicates
            recs_set = set(recs)
            for movie in fallback_recs:
                if len(recs) >= top_n:
                    break
                if movie not in recs_set:
                    recs.append(movie)
                    recs_set.add(movie)

        st.success("Here are your recommendations:")

        st.markdown("""
            <style>
            .movie-box {
                background-color: #f0f0f0;
                padding: 16px;
                margin: 10px 0;
                border-radius: 12px;
                transition: transform 0.2s ease-in-out;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                font-size: 18px;
                font-weight: 500;
            }
            .movie-box:hover {
                transform: scale(1.03);
                box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
            }
            </style>
        """, unsafe_allow_html=True)

        for movie in recs:
            st.markdown(f'<div class="movie-box">{movie}</div>', unsafe_allow_html=True)


else:
    st.markdown("### New here? No problem!")
    selected_genre = st.selectbox("Choose a Genre (optional)", ["None"] + genres)
    if st.button("Show Me Some Movies"):
        genre = None if selected_genre == "None" else selected_genre
        recs = get_weighted_recommendations(
            df,
            genre=genre,
            top_n=3
        )
        st.success("Here are some top-rated picks for you:")
        st.markdown("""
                    <style>.movie-box {
                    background-color: #f0f0f0;
                    padding: 16px;
                    margin: 10px 0;
                    border-radius: 12px;
                    transition: transform 0.2s ease-in-out;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    font-size: 18px;
                    font-weight: 500;
                    }
                    .movie-box:hover {
                    transform: scale(1.03);
                    box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
                    }
                    </style>""", unsafe_allow_html=True)

        for movie in recs:
            st.markdown(f'<div class="movie-box">{movie}</div>', unsafe_allow_html=True)

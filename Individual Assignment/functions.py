from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

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

def score_movies_item_based(userId, train_matrix, sim_item):
    if userId not in train_matrix.index:
        return {}

    user_row = train_matrix.loc[userId]
    watched = set(user_row[user_row > 0].index)
    scores = defaultdict(float)

    for movie in watched:
        if movie not in sim_item:
            continue
        similar = sim_item[movie].drop(index=movie, errors='ignore').nlargest(50)
        for sim_movie, score in similar.items():
            if sim_movie not in watched:
                scores[sim_movie] += score

    return scores



def score_movies_svd(userId, train_matrix, user_factors, item_factors):
    if userId not in train_matrix.index:
        return {}

    user_idx = train_matrix.index.get_loc(userId)
    user_vector = user_factors[user_idx]
    scores = np.dot(item_factors, user_vector)
    movie_ids = train_matrix.columns
    score_series = pd.Series(scores, index=movie_ids)

    rated = train_matrix.loc[userId]
    return score_series[rated == 0].to_dict()

def recommend_hybrid(
    userId,
    train_matrix,
    sim_item,
    user_factors,
    item_factors,
    dataframe,
    top_n=5,
    weights=(0.4, 0.6),     # defined through the grid search ahead
    filter_genre=None
):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Score from each model
    item_scores = score_movies_item_based(userId, train_matrix, sim_item)
    svd_scores  = score_movies_svd(userId, train_matrix, user_factors, item_factors)

    all_movie_ids = set(item_scores) | set(svd_scores)
    combined_scores = {}

    for movie_id in all_movie_ids:
        i = item_scores.get(movie_id, 0)
        s = svd_scores.get(movie_id, 0)
        combined_scores[movie_id] = (
            weights[0] * i + weights[1] * s
        )

    # Normalize scores
    if combined_scores:
        score_df = pd.DataFrame.from_dict(combined_scores, orient='index', columns=['score'])
        score_df['score'] = scaler.fit_transform(score_df[['score']])
        top_ids = score_df.sort_values('score', ascending=False).head(top_n * 3).index
    else:
        return f"No recommendations found for User {userId}."

    # Apply genre filter
    if filter_genre and filter_genre in dataframe.columns:
        filtered_df = dataframe[(dataframe['movieId'].isin(top_ids)) & (dataframe[filter_genre] == 1)]
    else:
        filtered_df = dataframe[dataframe['movieId'].isin(top_ids)]

    return filtered_df['title'].unique().tolist()[:top_n]


def get_weighted_recommendations(df, genre=None, m_threshold=100, top_n=10):
    if genre:
        if genre not in df.columns:
            raise ValueError(f"Genre '{genre}' not found in the dataset.")
        df = df[df[genre] == 1]

    # Calculate the global mean rating (C)
    C = df['rating'].mean()

    # Group by movie and compute v (count) and U(j) (mean)
    movie_stats = df.groupby('title').agg(
        v=('rating', 'count'),
        U=('rating', 'mean')
    ).reset_index()

    # Filter for movies with enough votes
    qualified = movie_stats[movie_stats['v'] >= m_threshold].copy()

    # Use m as the threshold
    m = m_threshold

    # Compute WR(j)
    qualified['WR'] = (qualified['v'] / (qualified['v'] + m)) * qualified['U'] + \
                      (m / (qualified['v'] + m)) * C
    sorted_df = qualified.sort_values("WR", ascending=False)
    
    return sorted_df['title'].head(top_n).tolist()
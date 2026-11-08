
import streamlit as st
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Load saved files
movies = pd.read_csv("movies_recommender.csv")
tfidf_matrix = load_npz("tfidf_matrix.npz")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #10121b;
}
h1 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movies "
    "using Machine Learning and NLP."
)

st.divider()

# Movie selection
st.subheader("Find Your Next Movie")

movie_list = sorted(movies["title"].dropna().unique())

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=15,
    value=10
)

# Recommendation function
def recommend_movies(movie_title, top_n):

    movie_index = movies.index[
        movies["title"] == movie_title
    ][0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    similar_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    similar_movies = [
        item for item in similar_movies
        if item[0] != movie_index
    ][:top_n]

    recommendations = movies.iloc[
        [item[0] for item in similar_movies]
    ][["title", "genres"]].copy()

    recommendations["Similarity Score"] = [
        round(item[1], 3)
        for item in similar_movies
    ]

    return recommendations.reset_index(drop=True)


# Recommendation button
if st.button("🍿 Recommend Movies", type="primary"):

    results = recommend_movies(
        selected_movie,
        number_of_recommendations
    )

    st.success(
        f"Movies similar to {selected_movie}"
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.caption(
    "Movie Recommendation System | "
    "Python | Pandas | Scikit-learn | TF-IDF"
)

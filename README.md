
# 🎬 Movie Recommendation System

A Content-Based Movie Recommendation System built using Python, Machine Learning, and NLP. It recommends similar movies based on movie genres and tags using TF-IDF Vectorization and Cosine Similarity.

## 🚀 Project Overview

This project helps users discover movies similar to their favourite movies. It analyzes movie genres and user-generated tags to identify similarities and generate recommendations.

## ✨ Features

- Search and select movies from the dataset.
- Generate up to 10 similar movie recommendations.
- Content-based recommendation approach.
- TF-IDF feature extraction.
- Cosine Similarity for measuring movie similarity.
- Interactive Streamlit web application.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Jupyter Notebook

## 📂 Project Files

- `app.py` – Streamlit application.
- `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer.
- `tfidf_matrix.npz` – Saved TF-IDF feature matrix.
- `movies_recommender.csv` – Processed movie dataset.
- `requirements.txt` – Project dependencies.
- `README.md` – Project documentation.

## ⚙️ How to Run

1. Clone this repository.
2. Install the required dependencies:

   `pip install -r requirements.txt`

3. Run the Streamlit application:

   `streamlit run app.py`

4. Open the local URL displayed in your browser.

## 🧠 Recommendation Methodology

1. Load and preprocess movie data.
2. Combine movie genres and tags into text features.
3. Convert text into numerical vectors using TF-IDF.
4. Calculate similarity using Cosine Similarity.
5. Recommend movies with the highest similarity scores.

## 📊 Dataset

Dataset: MovieLens Latest Small

Source: https://grouplens.org/datasets/movielens/latest/

## 👨‍💻 Project Information

**Project:** Movie Recommendation System

**Approach:** Content-Based Filtering

**Developer:** Aman Husain

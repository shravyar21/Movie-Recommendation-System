import streamlit as st
import pandas as pd
import numpy as np

#load dataset
df = pd.read_csv('movies_for_app.csv')
similarity_matrix = np.load("similarity_matrix.npy")

#recommendation function
def recommend(item_name, top_n = 5):
    titles = df['title'].fillna('').str.strip().str.lower()
    matches = df.index[titles == item_name.strip().lower()].tolist()

    if not matches:
        return []

    index = matches[0]

    scores = similarity_matrix[index]
    similar_indices = scores.argsort()[::-1]

    recommendations = []

    for i in similar_indices:
        if i != index:
            recommendations.append(
                (df.iloc[i]['title'], round(float(scores[i]), 3))
            )

        if len(recommendations) == top_n:
            break

    return recommendations


# Streamlit UI
st.title("Movie Recommendation System")

st.write("Select a movie to discover similar movies!")

movie_name = st.selectbox(
    "Choose a Movie",
    df['title'].dropna().unique()
)

if st.button("Recommend Movies"):

    results = recommend(movie_name)

    st.subheader("Recommended Movies")

    if results:
        for movie, score in results:
            st.write(f"🎥 {movie} | Similarity Score: {score}")
    else:
        st.warning("No recommendations found!")
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="centered"
)

st.title("Movie Recommendation System")
st.write("Discover movies similar to your favourite movies!")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_movies_small.csv")

    df = df.dropna(subset=['title'])
    df = df.drop_duplicates(subset=['title'])
    df = df.reset_index(drop=True)

    return df

df = load_data()

# Prepare movie features
def extract_names(value):
    try:
        data = ast.literal_eval(value) if isinstance(value, str) else []
        return " ".join(
            item.get('name', '') for item in data
            if isinstance(item, dict)
        )
    except (ValueError, SyntaxError, TypeError):
        return ""

feature_columns = [
    col for col in ['genres', 'keywords']
    if col in df.columns
]

features = []

for _, row in df.iterrows():

    parts = []

    for col in feature_columns:
        parts.append(extract_names(row[col]))

    if 'overview' in df.columns:
        parts.append(str(row['overview']))

    features.append(" ".join(parts))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(features)

similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(item_name, top_n=5):

    titles = df['title'].fillna('').str.strip().str.lower()

    matches = df.index[
        titles == item_name.strip().lower()
    ].tolist()

    if not matches:
        return []

    index = matches[0]

    scores = list(enumerate(similarity_matrix[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in scores:

        if i != index:

            recommendations.append(
                (df.iloc[i]['title'], round(float(score), 3))
            )

        if len(recommendations) == top_n:
            break

    return recommendations

# User Interface
movie_list = sorted(df['title'].dropna().unique())

selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button("Get Recommendations"):

    results = recommend(selected_movie, 5)

    if results:

        st.subheader("Recommended Movies")

        for movie, score in results:

            st.write(f"🎥 *{movie}*")
            st.write(f"Similarity Score: {score}")

    else:
        st.warning("No recommendations found.")

st.success(f"Loaded {len(df)} movies successfully!")
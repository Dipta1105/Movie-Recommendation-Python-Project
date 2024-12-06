import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
def load_data(file_path="data/movies.csv"):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(df):
    df['description'] = df['description'].fillna('')
    return df

# Build a recommendation system using TF-IDF
def build_recommendation_system(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies
def recommend_movies(title, similarity_matrix, df, top_n=5):
    if title not in df['title'].values:
        return f"Movie '{title}' not found in the database."
    
    idx = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in similarity_scores[1:top_n+1]]  # Exclude self-match
    return df['title'].iloc[top_indices].tolist()

# Main
if __name__ == "__main__":
    file_path = "data/movies.csv"
    data = load_data(file_path)
    data = preprocess_data(data)
    
    similarity_matrix = build_recommendation_system(data)
    print("Enter a movie title for recommendations:")
    movie_title = input().strip()
    
    recommendations = recommend_movies(movie_title, similarity_matrix, data)
    if isinstance(recommendations, list):
        print(f"Movies similar to '{movie_title}':")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print(recommendations)

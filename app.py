from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import logging

# Configure logging to see print statements in the console
logging.basicConfig(level=logging.DEBUG)

movies = pickle.load(open("movie_list.pkl", "rb"))  # Should be a Pandas DataFrame
similarity = pickle.load(open("similarity.pkl", "rb"))  # Similarity matrix (NumPy array)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        movie_name = request.form.get("movie")

        if not movie_name:
            return jsonify({'error': 'No movie name provided'}), 400

        movie_name = movie_name.lower().strip()
        logging.debug(f"Received movie name: {movie_name}")

        # Check if movie exists in dataset
        matched_movies = movies[movies['title'].str.lower() == movie_name]

        if matched_movies.empty:
            return jsonify({'error': 'Movie not found'}), 404

        movie_index = matched_movies.index[0]
        logging.debug(f"Movie index found: {movie_index}")

        # Get similarity scores and sort them
        similar_movies = list(enumerate(similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

        # Fetch recommended movie titles
        recommendations = movies.iloc[[i[0] for i in similar_movies]]['title'].tolist()
        logging.debug(f"Recommendations: {recommendations}")

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import re

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///movies.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "your_secret_key"
db = SQLAlchemy(app)

# User Authentication
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"

# TMDB API
API_KEY = "e0d439be479ba676a102b95ebdeebfa4"
BASE_URL = "https://api.themoviedb.org/3"

# DATABASE MODELS
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Integer, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def fetch_movies():
    try:
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.RequestException as e:
        print(f"Error fetching movies: {e}")
        return []

# Initialize movies globally
movies_df = pd.DataFrame()
cosine_sim = None

def initialize_movies():
    global movies_df, cosine_sim
    movies = fetch_movies()
    
    if not movies:
        movies = [{"id": 1, "title": "Sample Movie", "overview": "This is a sample movie.", "poster_path": None}]
    
    movies_df = pd.DataFrame(movies)
    movies_df.fillna("", inplace=True)
    
    tfidf = TfidfVectorizer(stop_words="english")
    if not movies_df.empty and "overview" in movies_df.columns:
        tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    else:
        cosine_sim = np.array([[1]])

def get_recommendations(movie_id):
    try:
        movie_id = int(movie_id)
        idx = movies_df[movies_df['id'] == movie_id].index
        if len(idx) == 0:
            return []
        idx = idx[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        
        return movies_df.iloc[movie_indices]['title'].tolist()
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return []

# Simple Matrix Factorization for Collaborative Filtering
class MatrixFactorization:
    def __init__(self, n_factors=20, n_iterations=100, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        
    def fit(self, ratings_matrix):
        """
        Train a matrix factorization model on the provided ratings matrix
        """
        n_users, n_items = ratings_matrix.shape
        
        # Initialize user and item factors with small random values
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        
        # Get indices of observed ratings
        mask = ~np.isnan(ratings_matrix)
        user_indices, item_indices = np.where(mask)
        
        # Training loop
        for _ in range(self.n_iterations):
            for u, i in zip(user_indices, item_indices):
                # Prediction
                prediction = self.user_factors[u].dot(self.item_factors[i])
                
                # Compute error
                error = ratings_matrix[u, i] - prediction
                
                # Update factors with gradient descent
                self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.regularization * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - self.regularization * self.item_factors[i])
                
        return self
    
    def predict(self, user_idx, item_idx=None):
        """
        Make rating predictions for a specific user, or for a user-item pair
        """
        if item_idx is None:
            # Predict all items for this user
            return self.user_factors[user_idx].dot(self.item_factors.T)
        else:
            # Predict specific user-item rating
            return self.user_factors[user_idx].dot(self.item_factors[item_idx])

# Content-based recommendation routes
@app.route("/recommend/content/<int:movie_id>")
def recommend_content_based(movie_id):
    recommendations = get_recommendations(movie_id)
    return jsonify({"recommendations": recommendations})

# Collaborative filtering recommendation route
@app.route("/recommend/collaborative")
@login_required
def recommend_collaborative():
    try:
        user_id = current_user.id
        
        # Get all reviews from the database
        reviews = pd.read_sql(Review.query.statement, db.session.bind)
        
        if len(reviews) < 10:
            return jsonify({"message": "Not enough review data for recommendations", "recommendations": []})

        # Create a user-movie rating matrix
        user_movie_df = reviews.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating').fillna(np.nan)
            
        # Train our matrix factorization model
        model = MatrixFactorization(n_factors=10, n_iterations=50)
        model.fit(user_movie_df.values)
        
        # Generate recommendations for the current user
        user_idx = user_movie_df.index.get_loc(user_id)
        
        # Get user's already-rated movies
        rated_movies = set(reviews[reviews['user_id'] == user_id]['movie_id'].values)
        
        # Predict ratings for all movies for this user
        all_predictions = model.predict(user_idx)
        
        # Create a list of (movie_id, predicted_rating) tuples
        movie_ids = user_movie_df.columns.tolist()
        predictions = [(movie_id, score) for movie_id, score in zip(movie_ids, all_predictions) 
                       if movie_id not in rated_movies]
        
        # Sort by predicted rating and take top 5
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:5]
        
        # Format the recommendations
        recommended_movies = []
        for movie_id, score in top_predictions:
            # Find the movie in our movies_df
            movie = movies_df[movies_df['id'] == movie_id]
            if not movie.empty:
                recommended_movies.append({
                    "id": int(movie_id),
                    "title": movie.iloc[0]['title'],
                    "predicted_rating": round(float(score), 1)
                })
            else:
                # Try to fetch the movie details from the API
                try:
                    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
                    response = requests.get(url)
                    if response.status_code == 200:
                        movie_data = response.json()
                        recommended_movies.append({
                            "id": movie_id,
                            "title": movie_data.get('title', f"Movie {movie_id}"),
                            "predicted_rating": round(float(score), 1)
                        })
                except Exception as e:
                    print(f"Error fetching movie data: {e}")
        
        return jsonify({'recommendations': recommended_movies})
    except Exception as e:
        print(f"Error in collaborative recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    movies = fetch_movies()
    return render_template("index.html", movies=movies)

@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    user = User.query.filter_by(username=username).first()
    
    if user and check_password_hash(user.password, password):
        login_user(user)
        flash("Logged in successfully!")
        return redirect(url_for("home"))
    else:
        flash("Invalid username or password")
        return redirect(url_for("login_page"))

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username")
    password = request.form.get("password")

    # Password validation
    password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()])[A-Za-z\d!@#$%^&*()]{9,}$'
    if not re.match(password_pattern, password):
        flash("Password must be at least 9 characters long and include at least one uppercase letter, one lowercase letter, one number, and one special character (!@#$%^&*())")
        return redirect(url_for("signup_page"))

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash("Username already exists")
        return redirect(url_for("signup_page"))

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    flash("Account created successfully!")
    return redirect(url_for("login_page"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully")
    return redirect(url_for("home"))

@app.route("/review", methods=["POST"])
@login_required
def review():
    data = request.get_json()
    movie_id = data.get("movie_id")
    rating = data.get("rating")
    
    if not movie_id or not rating:
        return jsonify({"error": "Missing data"}), 400
    
    # Check if the user has already reviewed this movie
    existing_review = Review.query.filter_by(
        user_id=current_user.id, movie_id=movie_id
    ).first()
    
    if existing_review:
        existing_review.rating = rating
        db.session.commit()
        return jsonify({"message": "Rating updated!"})
    else:
        new_review = Review(
            user_id=current_user.id,
            movie_id=movie_id,
            rating=rating
        )
        db.session.add(new_review)
        db.session.commit()
        return jsonify({"message": "Rating submitted!"})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        initialize_movies()
    app.run(debug=True, host="0.0.0.0", port=5000)
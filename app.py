from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

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
    if not movies_df.empty:
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

@app.route("/")
def home():
    return render_template("index.html", movies=movies_df.to_dict(orient="records"))

@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route("/signup_page")
def signup_page():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.form
    if User.query.filter_by(username=data["username"]).first():
        flash("User already exists")
        return redirect(url_for("signup_page"))
    
    hashed_password = generate_password_hash(data["password"], method='pbkdf2:sha256')
    user = User(username=data["username"], password=hashed_password)
    db.session.add(user)
    db.session.commit()
    flash("Signup successful! Please login.")
    return redirect(url_for("login_page"))

@app.route("/login", methods=["POST"])
def login():
    data = request.form
    user = User.query.filter_by(username=data["username"]).first()
    
    if user and check_password_hash(user.password, data["password"]):
        login_user(user)
        return redirect(url_for("home"))
    
    flash("Invalid credentials")
    return redirect(url_for("login_page"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/review", methods=["POST"])
@login_required
def submit_review():
    try:
        data = request.json
        movie_id = int(data["movie_id"])
        rating = int(data["rating"])
        
        if rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be between 1 and 5"}), 400
            
        if movie_id not in movies_df['id'].values:
            return jsonify({"error": "Movie not found"}), 404
            
        existing_review = Review.query.filter_by(
            user_id=current_user.id, 
            movie_id=movie_id
        ).first()
        
        if existing_review:
            existing_review.rating = rating
        else:
            review = Review(user_id=current_user.id, movie_id=movie_id, rating=rating)
            db.session.add(review)
            
        db.session.commit()
        return jsonify({"message": "Review submitted!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/content/<int:movie_id>")
def recommend_content_based(movie_id):
    try:
        recommendations = get_recommendations(movie_id)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/collaborative")
@login_required
def recommend_collaborative():
    try:
        user_id = current_user.id
        reviews = pd.read_sql(Review.query.statement, db.session.bind)
        
        if len(reviews) < 10:
            return jsonify({"message": "Not enough review data for recommendations", "recommendations": []})

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(reviews[['user_id', 'movie_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        model = SVD()
        model.fit(trainset)
        
        user_reviewed = reviews[reviews['user_id'] == user_id]['movie_id'].tolist()
        movies_to_predict = [movie_id for movie_id in movies_df['id'].tolist() if movie_id not in user_reviewed]
        
        predictions = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movies_to_predict]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommended_movies = [{"id": pred[0], "title": movies_df[movies_df['id'] == pred[0]]['title'].values[0], "predicted_rating": round(pred[1], 1)} for pred in predictions[:5]]

        return jsonify({'recommendations': recommended_movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        initialize_movies()
    app.run(debug=True)

import numpy as np
import pandas as pd
import json
import time

class SimpleMatrixFactorization:
    """
    A simple matrix factorization implementation that doesn't require 
    external libraries like LightFM or Surprise.
    """
    
    def __init__(self, n_factors=20, n_iterations=100, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        
    def fit(self, ratings_matrix):
        """Train the model on the provided ratings matrix"""
        # Get dimensions
        n_users, n_items = ratings_matrix.shape
        
        # Initialize user and item factors with small random values
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        
        # Get indices of observed ratings
        mask = ~np.isnan(ratings_matrix)
        user_indices, item_indices = np.where(mask)
        ratings = ratings_matrix[mask]
        
        # Training loop
        for iteration in range(self.n_iterations):
            start_time = time.time()
            
            # Shuffle the training data
            indices = np.arange(len(user_indices))
            np.random.shuffle(indices)
            
            # Batch update
            for idx in indices:
                u, i = user_indices[idx], item_indices[idx]
                r = ratings_matrix[u, i]
                
                # Compute prediction and error
                prediction = self.user_factors[u].dot(self.item_factors[i])
                error = r - prediction
                
                # Update factors
                self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.regularization * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - self.regularization * self.item_factors[i])
            
            # Compute training error
            if iteration % 10 == 0:
                predictions = np.zeros_like(ratings)
                for idx in range(len(ratings)):
                    u, i = user_indices[idx], item_indices[idx]
                    predictions[idx] = self.user_factors[u].dot(self.item_factors[i])
                
                mse = np.mean((ratings - predictions) ** 2)
                print(f"Iteration {iteration}: MSE = {mse:.4f}, Time: {time.time() - start_time:.2f} seconds")
        
        return self
    
    def predict(self, user_idx, item_indices=None):
        """Make predictions for a user on specific items or all items"""
        if item_indices is None:
            # Predict all items
            return self.user_factors[user_idx].dot(self.item_factors.T)
        else:
            # Predict specific items
            return np.array([self.user_factors[user_idx].dot(self.item_factors[i]) for i in item_indices])

def create_sample_data():
    """Create a sample user-item ratings matrix"""
    # Create 10 users and 20 movies
    n_users = 10
    n_items = 20
    
    # Create a mostly sparse matrix with some ratings
    ratings = np.full((n_users, n_items), np.nan)
    
    # Add some sample ratings (1-5 scale)
    for user_id in range(n_users):
        # Each user rates between 3 and 10 random movies
        n_ratings = np.random.randint(3, 11)
        rated_items = np.random.choice(n_items, n_ratings, replace=False)
        
        for item_id in rated_items:
            # Random rating between 1 and 5
            ratings[user_id, item_id] = np.random.randint(1, 6)
    
    return ratings

def get_recommendations(model, user_id, ratings_matrix, top_n=5):
    """Get top N recommendations for a user"""
    # Get user's ratings
    user_ratings = ratings_matrix[user_id, :]
    
    # Find items the user hasn't rated
    unrated_items = np.where(np.isnan(user_ratings))[0]
    
    # Predict ratings for unrated items
    predicted_ratings = model.predict(user_id)
    
    # Create a list of (item_id, predicted_rating) tuples for unrated items
    item_ratings = [(item_id, predicted_ratings[item_id]) for item_id in unrated_items]
    
    # Sort by predicted rating and get top N
    item_ratings.sort(key=lambda x: x[1], reverse=True)
    return item_ratings[:top_n]

def main():
    # Create sample data
    print("Creating sample data...")
    ratings_matrix = create_sample_data()
    
    # Print the ratings matrix
    print("\nSample ratings matrix (NaN means no rating):")
    print(pd.DataFrame(ratings_matrix).round(1))
    
    # Train the model
    print("\nTraining the matrix factorization model...")
    model = SimpleMatrixFactorization(n_factors=10, n_iterations=50, learning_rate=0.01)
    model.fit(ratings_matrix)
    
    # Get recommendations for each user
    print("\nGenerating recommendations:")
    for user_id in range(ratings_matrix.shape[0]):
        recommendations = get_recommendations(model, user_id, ratings_matrix, top_n=3)
        
        # Print user's current ratings
        user_ratings = ratings_matrix[user_id, :]
        rated_items = np.where(~np.isnan(user_ratings))[0]
        rated_items_str = ", ".join([f"Item {item_id}: {user_ratings[item_id]}" for item_id in rated_items])
        
        print(f"\nUser {user_id} rated: {rated_items_str}")
        print(f"Recommendations for User {user_id}:")
        for item_id, pred_rating in recommendations:
            print(f"  Item {item_id}: Predicted rating = {pred_rating:.2f}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Example dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'item_id': [101, 102, 103, 101, 104, 102, 103, 101, 104, 103],
    'rating': [5, 4, 3, 5, 4, 4, 5, 4, 5, 3]
}

df = pd.DataFrame(data)

# Create a user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print(user_item_matrix)

# Compute cosine similarity between users
similarity_matrix = cosine_similarity(user_item_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
print("User Similarity Matrix:")
print(similarity_df)

def predict_ratings(user_item_matrix, similarity_matrix):
    # Convert to numpy arrays for computation
    user_mean = np.mean(user_item_matrix, axis=1).reshape(-1, 1)
    user_item_matrix_centered = user_item_matrix - user_mean

    predictions = user_mean + similarity_matrix.dot(user_item_matrix_centered) / np.abs(similarity_matrix).sum(axis=1, keepdims=True)
    return predictions

# Predict ratings
predicted_ratings = predict_ratings(user_item_matrix.values, similarity_matrix)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
print("Predicted Ratings:")
print(predicted_ratings_df)

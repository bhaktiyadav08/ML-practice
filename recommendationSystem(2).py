import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "Item1": [5, 0, 3, 0],
    "Item2": [4, 0, 3, 2],
    "Item3": [1, 1, 0, 5],
    "Item4": [0, 0, 4, 4],
}

user_item_matrix = pd.DataFrame(data, index=["User1", "User2", "User3", "User4"])

user_item_matrix

item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("Item-Item Similarity Matrix:")
print(item_similarity_df)

# Step 3: Predict ratings
def predict_rating(user, item, user_item_matrix, similarity_matrix):
    if user_item_matrix.loc[user, item] > 0:  # Skip if the user already rated the item
        return user_item_matrix.loc[user, item]

    # Weighted sum of ratings for similar items
    similar_items = similarity_matrix[item]
    user_ratings = user_item_matrix.loc[user]

    numerator = np.dot(similar_items, user_ratings)
    denominator = np.sum(np.abs(similar_items)) if np.sum(similar_items) > 0 else 1

    return numerator / denominator

# Predict ratings for all items for User1
user = "User1"
predicted_ratings1 = {item: predict_rating(user, item, user_item_matrix, item_similarity_df) for item in user_item_matrix.columns}

predicted_ratings1

# Predict ratings for all items for User1
user = "User2"
predicted_ratings2 = {item: predict_rating(user, item, user_item_matrix, item_similarity_df) for item in user_item_matrix.columns}

predicted_ratings2

user = "User3"
predicted_ratings3 = {item: predict_rating(user, item, user_item_matrix, item_similarity_df) for item in user_item_matrix.columns}
predicted_ratings3

user = "User4"
predicted_ratings4 = {item: predict_rating(user, item, user_item_matrix, item_similarity_df) for item in user_item_matrix.columns}
predicted_ratings4

data1 = pd.DataFrame({
    "User1":predicted_ratings1,
    "User2":predicted_ratings2,
    "User3":predicted_ratings3,
    "User4":predicted_ratings4,
    })

data1

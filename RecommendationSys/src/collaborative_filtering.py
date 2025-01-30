import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def collaborative_filtering(data):

    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating'])

    print(data.isnull().sum())

    user_item_matrix = data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Non-zero entries: {np.count_nonzero(user_item_matrix.values)}")

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_matrix = train.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    test_matrix = test.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)

    similarity_matrix = cosine_similarity(train_matrix)

    pred_matrix = np.dot(similarity_matrix, train_matrix) / np.array([np.abs(similarity_matrix).sum(axis=1)]).T

    test_matrix_filled = test_matrix.reindex_like(train_matrix).fillna(0)

    test_flattened = test_matrix_filled.values.flatten()
    pred_flattened = pred_matrix.flatten()

    mask = test_flattened != 0
    if not mask.any():
        print("No valid ratings in the test set to evaluate.")
        return similarity_matrix

    test_flattened = test_flattened[mask]
    pred_flattened = pred_flattened[mask]

    mse = mean_squared_error(test_flattened, pred_flattened)
    print(f"Mean Squared Error: {mse:.2f}")

    return similarity_matrix

def precision_at_k(predictions, actual_ratings, k):
    top_k_pred = np.argsort(predictions)[:k]
    relevant_items = actual_ratings[top_k_pred] > 0
    precision = np.sum(relevant_items) / k
    return precision

def calculate_precision_at_k_for_all_users(predictions_matrix, actual_ratings_matrix, max_k=10):
    precision_values = []
    for k in range(1, max_k + 1):
        precisions = []
        for user_idx in range(predictions_matrix.shape[0]):
            predictions = predictions_matrix[user_idx]
            actual_ratings = actual_ratings_matrix[user_idx]
            precision = precision_at_k(predictions, actual_ratings, k)
            precisions.append(precision)
        avg_precision = np.mean(precisions)
        precision_values.append(avg_precision)
    return precision_values

def plot_precision_at_k(precision_values):
    plt.plot(range(1, len(precision_values) + 1), precision_values)
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.title('Precision at K')
    plt.show()
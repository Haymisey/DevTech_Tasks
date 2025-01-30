from src.data_preprocessing import load_data
from src.collaborative_filtering import collaborative_filtering,calculate_precision_at_k_for_all_users, plot_precision_at_k
from src.content_based_filtering import content_based_filtering
import numpy as np

data = load_data('data/amazon.csv')

collaborative_filtering(data)

content_based_filtering(data)

predictions_matrix = np.random.rand(100, 50)  
actual_ratings_matrix = np.random.randint(0, 2, (100, 50))  
precision_values = calculate_precision_at_k_for_all_users(predictions_matrix, actual_ratings_matrix, max_k=10)

plot_precision_at_k(precision_values)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import random
from implicit.als import AlternatingLeastSquares

# Load and prepare data
def load_data(books_path, reviews_path):
    """
    Load books and reviews datasets, and filter reviews based on the top 200 books.
    """
    books = pd.read_csv(books_path)
    reviews = pd.read_csv(reviews_path)
    books = books.head(200)  # Limit to top 200 books for simplicity
    reviews = reviews[reviews['book_id'].isin(books['book_id'])]
    return books, reviews

def create_user_item_matrix(reviews):
    """
    Create a user-item matrix from reviews.
    Returns:
        - csr_matrix: User-item matrix
        - book_id_to_idx: Mapping from book IDs to matrix indices
        - idx_to_book_id: Mapping from matrix indices to book IDs
    """
    user_item_pivot = reviews.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    book_id_to_idx = {book_id: i for i, book_id in enumerate(user_item_pivot.columns)}
    idx_to_book_id = {i: book_id for book_id, i in book_id_to_idx.items()}
    return csr_matrix(user_item_pivot), book_id_to_idx, idx_to_book_id

# Recommendation Algorithms
def user_knn_recommendations(user_item_matrix, user_index, num_recommendations, idx_to_book_id):
    """
    Generate recommendations using User-based KNN.
    """
    model = NearestNeighbors(n_neighbors=50, metric='cosine')
    model.fit(user_item_matrix)
    distances, indices = model.kneighbors(user_item_matrix[user_index], n_neighbors=21)
    similar_users_indices = indices[0][1:]  # Exclude the user themselves
    similar_users_ratings = user_item_matrix[similar_users_indices].toarray()
    recommended_item_indices = np.argsort(-similar_users_ratings.sum(axis=0))[:num_recommendations]
    return [idx_to_book_id[i] for i in recommended_item_indices]

def als_recommendations(user_item_matrix, user_index, num_recommendations, idx_to_book_id):
    """
    Generate recommendations using Alternating Least Squares (ALS).
    """
    model = AlternatingLeastSquares(factors=50, iterations=15, regularization=0.1)
    model.fit(user_item_matrix.T)  # ALS requires transposed user-item matrix
    scores = model.recommend(user_index, user_item_matrix[user_index], N=num_recommendations)
    return [idx_to_book_id[i] for i, _ in scores]

def most_popular_recommendations(user_item_matrix, num_recommendations, idx_to_book_id):
    """
    Generate recommendations based on item popularity.
    """
    item_popularity = np.array(user_item_matrix.sum(axis=0)).flatten()
    recommended_item_indices = np.argsort(-item_popularity)[:num_recommendations]
    return [idx_to_book_id[i] for i in recommended_item_indices]

# Feedback Loop
def simulate_feedback_loop(reviews, books, iterations, selection_policy, algorithm):
    """
    Simulate a feedback loop to observe the evolution of popularity and diversity over time.
    """
    user_item_matrix, book_id_to_idx, idx_to_book_id = create_user_item_matrix(reviews)
    avg_popularity_over_time = []
    aggregate_diversity_over_time = []
    total_books = len(books['book_id'].unique())

    for t in range(iterations):
        print(f"Iteration {t + 1} with {selection_policy} policy using {algorithm}")
        train_data, _ = train_test_split(reviews, test_size=0.2)
        train_matrix, book_id_to_idx, idx_to_book_id = create_user_item_matrix(train_data)
        train_matrix = train_matrix.tolil()  # Convert to sparse format for efficient updates
        recommended_items_set = set()

        for user_index in range(train_matrix.shape[0]):
            # Generate recommendations for each user
            recommendations = generate_recommendations(algorithm, train_matrix, user_index, 10, idx_to_book_id)
            # Select items based on the selection policy
            selected_items = select_item(recommendations, selection_policy)
            recommended_items_set.update(selected_items)

        # Update metrics
        avg_popularity, diversity = update_popularity_and_diversity(recommended_items_set, train_matrix)
        avg_popularity_over_time.append(avg_popularity)
        aggregate_diversity_over_time.append(diversity / total_books)

        print(f"Average popularity at iteration {t + 1}: {avg_popularity}")
        print(f"Aggregate diversity at iteration {t + 1}: {diversity / total_books}")

    return avg_popularity_over_time, aggregate_diversity_over_time

# Helper Functions
def select_item(recommendations, policy, top_n=10):
    """
    Select items from recommendations based on the specified policy.
    """
    if policy == 'random':
        return [random.choice(recommendations)]
    elif policy == 'top_n':
        return recommendations[:top_n]
    elif policy == 'rank_based':
        probabilities = [np.exp(-0.5 * i) for i in range(len(recommendations))]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        return [np.random.choice(recommendations, p=probabilities)]
    else:
        raise ValueError("Unknown selection policy")

def update_popularity_and_diversity(recommended_items_set, train_matrix):
    """
    Calculate the average popularity and diversity of recommended items.
    """
    diversity = len(recommended_items_set)
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    avg_popularity = np.mean(item_popularity)
    return avg_popularity, diversity

# Main Execution
books_path = "/kaggle/input/goodreads-comic-books-dataset/comic_books.csv"
reviews_path = "/kaggle/input/goodreads-comic-books-dataset/comic_books_reviews.csv"
books, reviews = load_data(books_path, reviews_path)

iterations = 10
selection_policy = 'random'
algorithm = 'ALS'

# Simulate feedback loop
avg_popularity_over_time, aggregate_diversity_over_time = simulate_feedback_loop(reviews, books, iterations, selection_policy, algorithm)

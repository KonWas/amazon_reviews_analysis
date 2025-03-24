import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import os

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = "visualizations"
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)


def get_viz_path(filename):
    """Get the full path for a visualization file"""
    return os.path.join(VISUALIZATIONS_DIR, filename)


def create_user_item_matrix(df):
    """
    Create a user-item matrix for collaborative filtering

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        tuple: (matrix, user_indices, item_indices) -
               matrix of user ratings, mappings of user IDs to indices and product IDs to indices
    """
    # Filter out rows with missing user_id or product_id
    filtered_df = df.dropna(subset=['user_id', 'product_id', 'score'])

    # Create user and product ID dictionaries
    user_ids = filtered_df['user_id'].unique()
    product_ids = filtered_df['product_id'].unique()

    user_indices = {user_id: i for i, user_id in enumerate(user_ids)}
    item_indices = {product_id: i for i, product_id in enumerate(product_ids)}

    # Create the user-item matrix
    user_item_matrix = np.zeros((len(user_ids), len(product_ids)))

    for _, row in filtered_df.iterrows():
        user_idx = user_indices[row['user_id']]
        item_idx = item_indices[row['product_id']]
        user_item_matrix[user_idx, item_idx] = row['score']

    return user_item_matrix, user_indices, item_indices


def analyze_rating_distribution(df):
    """
    Analyze the distribution of ratings

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Visualize rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='score', data=df)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
    plt.savefig(get_viz_path('rating_distribution.png'))
    print("Rating distribution visualization saved as 'rating_distribution.png'")

    # Calculate average rating per product
    avg_ratings = df.groupby('product_id')['score'].agg(['mean', 'count']).reset_index()
    avg_ratings = avg_ratings.rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'})

    # Filter products with at least 5 ratings
    popular_products = avg_ratings[avg_ratings['num_ratings'] >= 5].sort_values('avg_rating', ascending=False)

    print("\nTop 10 highest rated products (with at least 5 ratings):")
    top_products = popular_products.head(10)

    # Get product names
    product_names = {}
    for _, row in df.drop_duplicates('product_id').iterrows():
        product_names[row['product_id']] = row['product_title']

    for _, row in top_products.iterrows():
        product_id = row['product_id']
        product_name = product_names.get(product_id, 'Unknown')
        print(f"Product ID: {product_id}")
        print(f"Product Name: {product_name}")
        print(f"Average Rating: {row['avg_rating']:.2f}")
        print(f"Number of Ratings: {row['num_ratings']}")
        print("-" * 50)


def item_based_collaborative_filtering(df):
    """
    Implement item-based collaborative filtering

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        pandas.DataFrame: DataFrame with item-item similarity scores
    """
    # Filter out rows with missing data
    filtered_df = df.dropna(subset=['user_id', 'product_id', 'score'])

    # Create a pivot table: rows are products, columns are users
    pivot_table = filtered_df.pivot_table(
        index='product_id',
        columns='user_id',
        values='score',
        fill_value=0
    )

    # Calculate item-item similarity using cosine similarity
    item_similarity = cosine_similarity(pivot_table)

    # Convert to DataFrame
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=pivot_table.index,
        columns=pivot_table.index
    )

    return item_similarity_df


def get_similar_products(item_similarity_df, product_id, n=5):
    """
    Get top N similar products for a given product

    Args:
        item_similarity_df (pandas.DataFrame): DataFrame with item-item similarity scores
        product_id (str): ID of the product to find similar items for
        n (int): Number of similar products to return

    Returns:
        pandas.Series: Series with similarity scores for the most similar products
    """
    if product_id not in item_similarity_df.index:
        return pd.Series([])

    # Get similarity scores for the product
    similar_scores = item_similarity_df[product_id]

    # Sort by similarity (descending) and get top N (excluding the product itself)
    similar_products = similar_scores.sort_values(ascending=False)[1:n + 1]

    return similar_products


def matrix_factorization_recommender(df, n_components=20):
    """
    Implement matrix factorization for collaborative filtering using SVD

    Args:
        df (pandas.DataFrame): DataFrame with review data
        n_components (int): Number of latent factors

    Returns:
        tuple: (product_features, reconstructed_matrix) -
               product features matrix and reconstructed rating matrix
    """
    # Filter out rows with missing data
    filtered_df = df.dropna(subset=['user_id', 'product_id', 'score'])

    # Create a pivot table: rows are users, columns are products
    pivot_table = filtered_df.pivot_table(
        index='user_id',
        columns='product_id',
        values='score',
        fill_value=0
    )

    # Convert to sparse matrix
    sparse_matrix = csr_matrix(pivot_table.values)

    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(sparse_matrix)

    # Get the product features
    product_features = svd.components_

    # Reconstruct the rating matrix
    reconstructed_matrix = svd.inverse_transform(svd.transform(sparse_matrix))

    return product_features, reconstructed_matrix, pivot_table.index, pivot_table.columns


def recommend_products_for_user(user_id, reconstructed_matrix, users_index, products_index, top_n=5):
    """
    Recommend products for a user based on matrix factorization

    Args:
        user_id (str): ID of the user to recommend products for
        reconstructed_matrix (numpy.ndarray): Reconstructed rating matrix
        users_index (pandas.Index): Index of user IDs
        products_index (pandas.Index): Index of product IDs
        top_n (int): Number of products to recommend

    Returns:
        list: List of (product_id, predicted_rating) tuples
    """
    if user_id not in users_index:
        return []

    # Get the user's index
    user_idx = users_index.get_loc(user_id)

    # Get the user's predicted ratings
    user_ratings = reconstructed_matrix[user_idx]

    # Create a dictionary of product IDs and predicted ratings
    product_ratings = dict(zip(products_index, user_ratings))

    # Sort by predicted rating (descending) and get top N
    top_products = sorted(product_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_products


def evaluate_recommendations(df):
    """
    Evaluate the recommendation system using a simple train-test split

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    from sklearn.model_selection import train_test_split

    # Filter out rows with missing data
    filtered_df = df.dropna(subset=['user_id', 'product_id', 'score'])

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

    # Train the recommender on the training set
    product_features, reconstructed_matrix, users_index, products_index = matrix_factorization_recommender(train_df)

    # Evaluate on the test set
    # Calculate RMSE (Root Mean Square Error)
    rmse_sum = 0
    count = 0

    for _, row in test_df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']
        actual_rating = row['score']

        if user_id not in users_index or product_id not in products_index:
            continue

        user_idx = users_index.get_loc(user_id)
        product_idx = products_index.get_loc(product_id)

        predicted_rating = reconstructed_matrix[user_idx, product_idx]
        rmse_sum += (predicted_rating - actual_rating) ** 2
        count += 1

    if count > 0:
        rmse = np.sqrt(rmse_sum / count)
        print(f"\nRecommendation System Evaluation:")
        print(f"RMSE on test set: {rmse:.4f}")
    else:
        print("\nCould not evaluate RMSE due to missing user/product pairs in test set")


def generate_example_recommendations(df):
    """
    Generate example recommendations for a few products

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Implement item-based collaborative filtering
    item_similarity_df = item_based_collaborative_filtering(df)

    # Get 5 product IDs with the most reviews
    popular_products = df.groupby('product_id').size().sort_values(ascending=False).head(5).index.tolist()

    print("\nExample Product Recommendations:")

    # Get product names
    product_names = {}
    for _, row in df.drop_duplicates('product_id').iterrows():
        product_names[row['product_id']] = row['product_title']

    for product_id in popular_products:
        product_name = product_names.get(product_id, 'Unknown')
        print(f"\nSimilar products to {product_id}: {product_name}")

        similar_products = get_similar_products(item_similarity_df, product_id)

        if not similar_products.empty:
            for similar_id, similarity in similar_products.items():
                similar_name = product_names.get(similar_id, 'Unknown')
                print(f"  - {similar_id}: {similar_name} (Similarity: {similarity:.4f})")
        else:
            print("  No similar products found")


if __name__ == "__main__":
    # Example usage
    input_file = "results\\cell_phones_reviews.csv"

    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews")

    # Analyze rating distribution
    analyze_rating_distribution(df)

    # Generate example recommendations
    generate_example_recommendations(df)

    # Evaluate the recommendation system
    evaluate_recommendations(df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = "visualizations"
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)


def get_viz_path(filename):
    """Get the full path for a visualization file"""
    return os.path.join(VISUALIZATIONS_DIR, filename)


def preprocess_text(text):
    """
    Preprocess text data for clustering

    Args:
        text (str): Text to preprocess

    Returns:
        str: Preprocessed text
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    return ' '.join(tokens)


def download_nltk_resources():
    """Download necessary NLTK resources"""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")


def cluster_reviews_by_text(df, n_clusters=5):
    """
    Cluster reviews based on their text content using K-means

    Args:
        df (pandas.DataFrame): DataFrame with review data
        n_clusters (int): Number of clusters to create

    Returns:
        pandas.DataFrame: DataFrame with cluster assignments
    """
    # Preprocess the text
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Remove empty texts
    filtered_df = df[df['processed_text'] != ""]

    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(filtered_df['processed_text'])

    # Determine optimal number of clusters using silhouette score if not specified
    if n_clusters is None:
        print("Determining optimal number of clusters...")
        silhouette_scores = []
        range_n_clusters = range(2, 10)
        for num_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_tfidf)
            silhouette_avg = silhouette_score(X_tfidf, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters = {num_clusters}, the silhouette score is {silhouette_avg:.4f}")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.title('Silhouette Score for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.savefig(get_viz_path('cl_silhouette_scores.png'))
        print("Silhouette scores visualization saved as 'cl_silhouette_scores.png'")

        # Choose the number of clusters that gives the highest silhouette score
        n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {n_clusters}")

    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    filtered_df['cluster'] = kmeans.fit_predict(X_tfidf)

    # For visualization, reduce dimensions using PCA
    print("Reducing dimensions for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Add PCA components to DataFrame
    filtered_df['pca_1'] = X_pca[:, 0]
    filtered_df['pca_2'] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', data=filtered_df, palette='viridis')
    plt.title('Clusters of Reviews (PCA)')
    plt.legend(title='Cluster')
    plt.savefig(get_viz_path('cl_review_clusters_pca.png'))
    print("PCA clusters visualization saved as 'cl_review_clusters_pca.png'")

    # Try t-SNE for better visualization
    print("Applying t-SNE for better visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_tfidf.toarray())

    # Add t-SNE components to DataFrame
    filtered_df['tsne_1'] = X_tsne[:, 0]
    filtered_df['tsne_2'] = X_tsne[:, 1]

    # Visualize clusters with t-SNE
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='cluster', data=filtered_df, palette='viridis')
    plt.title('Clusters of Reviews (t-SNE)')
    plt.legend(title='Cluster')
    plt.savefig(get_viz_path('cl_review_clusters_tsne.png'))
    print("t-SNE clusters visualization saved as 'cl_review_clusters_tsne.png'")

    return filtered_df


def analyze_clusters(df_with_clusters):
    """
    Analyze the content of each cluster

    Args:
        df_with_clusters (pandas.DataFrame): DataFrame with cluster assignments
    """
    n_clusters = df_with_clusters['cluster'].nunique()

    print(f"\nAnalyzing {n_clusters} clusters:")

    # For each cluster, find the most common words
    for cluster in range(n_clusters):
        cluster_texts = df_with_clusters[df_with_clusters['cluster'] == cluster]['processed_text']

        # Count words
        all_words = ' '.join(cluster_texts).split()
        word_counts = Counter(all_words).most_common(10)

        # Get average rating for this cluster
        avg_rating = df_with_clusters[df_with_clusters['cluster'] == cluster]['score'].mean()

        # Get example reviews
        example_reviews = df_with_clusters[df_with_clusters['cluster'] == cluster]['text'].sample(
            min(3, len(cluster_texts))).tolist()

        print(f"\nCluster {cluster}:")
        print(f"Number of reviews: {len(cluster_texts)}")
        print(f"Average rating: {avg_rating:.2f}")
        print("Top 10 words:")
        for word, count in word_counts:
            print(f"  - {word}: {count}")

        print("Example reviews:")
        for i, review in enumerate(example_reviews):
            print(f"  {i + 1}. {review[:100]}..." if len(review) > 100 else f"  {i + 1}. {review}")

    # Visualize cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    plt.title('Number of Reviews in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.savefig(get_viz_path('cl_cluster_sizes.png'))
    print("\nCluster sizes visualization saved as 'cl_cluster_sizes.png'")

    # Visualize average ratings per cluster
    plt.figure(figsize=(10, 6))
    cluster_ratings = df_with_clusters.groupby('cluster')['score'].mean().sort_index()
    sns.barplot(x=cluster_ratings.index, y=cluster_ratings.values)
    plt.title('Average Rating per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Rating')
    plt.savefig(get_viz_path('cl_cluster_ratings.png'))
    print("Cluster ratings visualization saved as 'cl_cluster_ratings.png'")


def cluster_products_by_reviews(df):
    """
    Cluster products based on their review characteristics

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        pandas.DataFrame: DataFrame with product clusters
    """
    # Aggregate review data by product
    print("Aggregating review data by product...")
    product_stats = df.groupby('product_id').agg({
        'score': ['mean', 'count', 'std'],
        'helpful_votes': ['sum', 'mean'],
        'total_votes': ['sum', 'mean']
    }).reset_index()

    # Flatten the column names
    product_stats.columns = ['product_id'] + [f'{col[0]}_{col[1]}' for col in product_stats.columns[1:]]

    # Remove products with too few reviews
    product_stats = product_stats[product_stats['score_count'] >= 3]

    # Handle missing values
    product_stats = product_stats.fillna(0)

    # Standardize the features
    print("Standardizing features...")
    features = product_stats.columns[1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(product_stats[features])

    # Determine optimal number of clusters
    print("Determining optimal number of clusters...")
    silhouette_scores = []
    range_n_clusters = range(2, 10)
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {num_clusters}, the silhouette score is {silhouette_avg:.4f}")

    # Choose the number of clusters that gives the highest silhouette score
    n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {n_clusters}")

    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    product_stats['cluster'] = kmeans.fit_predict(X_scaled)

    # For visualization, reduce dimensions using PCA
    print("Reducing dimensions for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA components to DataFrame
    product_stats['pca_1'] = X_pca[:, 0]
    product_stats['pca_2'] = X_pca[:, 1]

    # Visualize clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='pca_1', y='pca_2', hue='cluster', data=product_stats, palette='viridis', s=100)
    plt.title('Clusters of Products (PCA)')
    plt.legend(title='Cluster')
    plt.savefig(get_viz_path('cl_product_clusters_pca.png'))
    print("PCA product clusters visualization saved as 'cl_product_clusters_pca.png'")

    # Analyze clusters
    print("\nAnalyzing product clusters:")
    for cluster in range(n_clusters):
        cluster_products = product_stats[product_stats['cluster'] == cluster]

        print(f"\nProduct Cluster {cluster}:")
        print(f"Number of products: {len(cluster_products)}")
        print("Average characteristics:")
        for feature in features:
            avg_value = cluster_products[feature].mean()
            print(f"  - {feature}: {avg_value:.2f}")

    return product_stats


if __name__ == "__main__":
    # Example usage
    input_file = "results\\cell_phones_reviews.csv"

    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews")

    # Download NLTK resources
    download_nltk_resources()

    # Cluster reviews by text content
    df_with_clusters = cluster_reviews_by_text(df)

    # Analyze the clusters
    analyze_clusters(df_with_clusters)

    # Cluster products by review characteristics
    product_clusters = cluster_products_by_reviews(df)

    # Save results
    df_with_clusters.to_csv('review_clusters.csv', index=False)
    product_clusters.to_csv('product_clusters.csv', index=False)
    print("\nClustering results saved to 'review_clusters.csv' and 'product_clusters.csv'")
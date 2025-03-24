import os
import pandas as pd
from datetime import datetime

# Create reports directory if it doesn't exist
REPORTS_DIR = "reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)


def get_report_path(filename):
    """Get the full path for a report file"""
    return os.path.join(REPORTS_DIR, filename)


def generate_header(title):
    """Generate a formatted header for reports"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""
{'-' * 80}
{title.upper()}
{'-' * 80}
Generated: {now}
{'-' * 80}
"""
    return header


def generate_basic_stats_report(df):
    """Generate a report with basic statistics about the dataset"""
    report_path = get_report_path("basic_stats_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Basic Statistics Report"))

        # Dataset overview
        f.write("\n1. DATASET OVERVIEW\n")
        f.write(f"Number of reviews: {len(df)}\n")
        f.write(f"Number of unique products: {df['product_id'].nunique()}\n")
        f.write(f"Number of unique users: {df['user_id'].nunique()}\n")

        # Check for missing values
        f.write("\n2. MISSING VALUES\n")
        missing_values = df.isnull().sum()
        for column, count in missing_values.items():
            if count > 0:
                f.write(f"{column}: {count} ({count / len(df) * 100:.2f}%)\n")

        # Rating distribution
        f.write("\n3. RATING DISTRIBUTION\n")
        rating_counts = df['score'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            f.write(f"Rating {rating}: {count} reviews ({count / len(df) * 100:.2f}%)\n")

        f.write(f"\nAverage rating: {df['score'].mean():.2f}\n")

        # Review length statistics
        if 'review_length' not in df.columns:
            df['review_length'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

        f.write("\n4. REVIEW LENGTH STATISTICS\n")
        f.write(f"Average review length (words): {df['review_length'].mean():.2f}\n")
        f.write(f"Median review length (words): {df['review_length'].median()}\n")
        f.write(f"Shortest review (words): {df['review_length'].min()}\n")
        f.write(f"Longest review (words): {df['review_length'].max()}\n")

        f.write("\n5. REVIEW LENGTH BY RATING\n")
        avg_length_by_rating = df.groupby('score')['review_length'].mean().reset_index()
        for _, row in avg_length_by_rating.iterrows():
            f.write(f"Rating {row['score']}: {row['review_length']:.2f} words\n")

    print(f"Basic statistics report saved to {report_path}")
    return report_path


def generate_sentiment_analysis_report(df):
    """Generate a report about sentiment analysis results"""
    report_path = get_report_path("sentiment_analysis_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Sentiment Analysis Report"))

        # Check if sentiment columns exist
        if 'sentiment' not in df.columns or 'rating_sentiment' not in df.columns:
            f.write("Error: Sentiment analysis columns not found in the DataFrame.\n")
            return report_path

        # Rating-based sentiment
        f.write("\n1. RATING-BASED SENTIMENT\n")
        rating_sentiment_counts = df['rating_sentiment'].value_counts()
        for sentiment, count in rating_sentiment_counts.items():
            f.write(f"{sentiment.capitalize()}: {count} reviews ({count / len(df) * 100:.2f}%)\n")

        # NLTK-based sentiment
        f.write("\n2. NLTK-BASED SENTIMENT\n")
        nltk_sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in nltk_sentiment_counts.items():
            f.write(f"{sentiment.capitalize()}: {count} reviews ({count / len(df) * 100:.2f}%)\n")

        # Confusion matrix between methods
        f.write("\n3. COMPARISON BETWEEN METHODS\n")
        confusion = pd.crosstab(df['rating_sentiment'], df['sentiment'],
                                rownames=['Rating Sentiment'],
                                colnames=['NLTK Sentiment'])
        f.write(confusion.to_string())

        # Average compound score by rating
        if 'compound_score' in df.columns:
            f.write("\n\n4. AVERAGE COMPOUND SCORE BY RATING\n")
            avg_compound_by_rating = df.groupby('score')['compound_score'].mean().reset_index()
            for _, row in avg_compound_by_rating.iterrows():
                f.write(f"Rating {row['score']}: {row['compound_score']:.4f}\n")

        # Correlation between score and compound
        if 'compound_score' in df.columns:
            correlation = df['score'].corr(df['compound_score'])
            f.write(f"\nCorrelation between rating and sentiment compound score: {correlation:.4f}\n")

    print(f"Sentiment analysis report saved to {report_path}")
    return report_path


def generate_recommender_system_report(df):
    """Generate a report about the recommender system"""
    report_path = get_report_path("recommender_system_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Recommender System Report"))

        # Top rated products
        f.write("\n1. TOP RATED PRODUCTS (with at least 5 reviews)\n")
        # Filter products with at least 5 reviews
        products_with_min_reviews = df.groupby('product_id').filter(lambda x: len(x) >= 5)
        top_products = products_with_min_reviews.groupby(['product_id', 'product_title'])['score'].mean().reset_index()
        top_products = top_products.sort_values('score', ascending=False).head(20)

        for i, (_, row) in enumerate(top_products.iterrows(), 1):
            f.write(f"{i}. {row['product_title']} (ID: {row['product_id']}): {row['score']:.2f}\n")

        # Most reviewed products
        f.write("\n2. MOST REVIEWED PRODUCTS\n")
        most_reviewed = df.groupby(['product_id', 'product_title']).size().reset_index(name='num_reviews')
        most_reviewed = most_reviewed.sort_values('num_reviews', ascending=False).head(20)

        for i, (_, row) in enumerate(most_reviewed.iterrows(), 1):
            f.write(f"{i}. {row['product_title']} (ID: {row['product_id']}): {row['num_reviews']} reviews\n")

        # User activity
        f.write("\n3. USER ACTIVITY\n")
        user_activity = df.groupby('user_id').size().reset_index(name='num_reviews')
        user_activity = user_activity.sort_values('num_reviews', ascending=False)

        f.write(f"Most active user: {user_activity.iloc[0]['num_reviews']} reviews\n")
        f.write(f"Average reviews per user: {user_activity['num_reviews'].mean():.2f}\n")
        f.write(f"Median reviews per user: {user_activity['num_reviews'].median()}\n")

        # Users with more than 10 reviews
        active_users = user_activity[user_activity['num_reviews'] > 10]
        f.write(f"Users with more than 10 reviews: {len(active_users)}\n")

        # Rating distribution (statistics)
        f.write("\n4. RATING DISTRIBUTION STATISTICS\n")
        f.write(f"Average rating: {df['score'].mean():.2f}\n")
        f.write(f"Median rating: {df['score'].median()}\n")
        f.write(f"Standard deviation: {df['score'].std():.2f}\n")
        f.write(f"Mode (most common rating): {df['score'].mode()[0]}\n")

    print(f"Recommender system report saved to {report_path}")
    return report_path


def generate_clustering_report(df_with_clusters, product_clusters):
    """Generate a report about clustering results"""
    report_path = get_report_path("clustering_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Clustering Report"))

        # Review clusters
        if 'cluster' in df_with_clusters.columns:
            f.write("\n1. REVIEW CLUSTERS\n")
            # Number of clusters
            n_clusters = df_with_clusters['cluster'].nunique()
            f.write(f"Number of review clusters: {n_clusters}\n")

            # Cluster sizes
            f.write("\nCluster sizes:\n")
            cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
            for cluster, size in cluster_sizes.items():
                f.write(f"Cluster {cluster}: {size} reviews ({size / len(df_with_clusters) * 100:.2f}%)\n")

            # Average ratings per cluster
            f.write("\nAverage ratings per cluster:\n")
            avg_ratings = df_with_clusters.groupby('cluster')['score'].mean().sort_index()
            for cluster, avg_rating in avg_ratings.items():
                f.write(f"Cluster {cluster}: {avg_rating:.2f}\n")
        else:
            f.write("No review clustering information available.\n")

        # Product clusters
        if 'cluster' in product_clusters.columns:
            f.write("\n2. PRODUCT CLUSTERS\n")
            # Number of clusters
            n_clusters = product_clusters['cluster'].nunique()
            f.write(f"Number of product clusters: {n_clusters}\n")

            # Cluster sizes
            f.write("\nCluster sizes:\n")
            cluster_sizes = product_clusters['cluster'].value_counts().sort_index()
            for cluster, size in cluster_sizes.items():
                f.write(f"Cluster {cluster}: {size} products ({size / len(product_clusters) * 100:.2f}%)\n")

            # Cluster characteristics
            f.write("\nCluster characteristics:\n")
            for cluster in range(n_clusters):
                f.write(f"\nProduct Cluster {cluster}:\n")
                cluster_products = product_clusters[product_clusters['cluster'] == cluster]

                # Get all numeric features (not product_id or cluster)
                features = [col for col in product_clusters.columns if
                            col not in ['product_id', 'cluster'] and pd.api.types.is_numeric_dtype(
                                product_clusters[col])]

                if features:
                    for feature in features:
                        avg_value = cluster_products[feature].mean()
                        f.write(f"  - {feature}: {avg_value:.2f}\n")
                else:
                    f.write("  No numeric features found for analysis.\n")
        else:
            f.write("No product clustering information available.\n")

    print(f"Clustering report saved to {report_path}")
    return report_path


def generate_product_categories_report(df):
    """Generate a report about product categories"""
    report_path = get_report_path("product_categories_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Product Categories Report"))

        if 'product_category' not in df.columns:
            f.write("No product category information available.\n")
            return report_path

        # Category distribution
        f.write("\n1. CATEGORY DISTRIBUTION\n")
        category_counts = df['product_category'].value_counts()
        for category, count in category_counts.items():
            f.write(f"{category}: {count} reviews ({count / len(df) * 100:.2f}%)\n")

        # Average rating by category
        f.write("\n2. AVERAGE RATING BY CATEGORY\n")
        avg_rating_by_category = df.groupby('product_category')['score'].mean().sort_values(ascending=False)
        for category, avg_rating in avg_rating_by_category.items():
            f.write(f"{category}: {avg_rating:.2f}\n")

        # Top products per category
        f.write("\n3. TOP PRODUCTS BY CATEGORY\n")
        for category in category_counts.index:
            f.write(f"\n{category.upper()}\n")

            # Filter products in this category with at least 3 reviews
            category_df = df[df['product_category'] == category]
            products_with_min_reviews = category_df.groupby('product_id').filter(lambda x: len(x) >= 3)

            if len(products_with_min_reviews) == 0:
                f.write("  No products with at least 3 reviews in this category.\n")
                continue

            top_products = products_with_min_reviews.groupby(['product_id', 'product_title'])[
                'score'].mean().reset_index()
            top_products = top_products.sort_values('score', ascending=False).head(5)

            for i, (_, row) in enumerate(top_products.iterrows(), 1):
                f.write(f"  {i}. {row['product_title']} - {row['score']:.2f}\n")

    print(f"Product categories report saved to {report_path}")
    return report_path


def generate_full_report(df, df_with_clusters=None, product_clusters=None):
    """Generate a comprehensive report combining all aspects of the analysis"""
    report_path = get_report_path("full_analysis_report.txt")

    # Generate individual reports
    basic_stats_report = generate_basic_stats_report(df)
    sentiment_report = generate_sentiment_analysis_report(df)
    recommender_report = generate_recommender_system_report(df)

    if df_with_clusters is not None and product_clusters is not None:
        clustering_report = generate_clustering_report(df_with_clusters, product_clusters)

    if 'product_category' in df.columns:
        categories_report = generate_product_categories_report(df)

    # Combine all reports into one
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(generate_header("Complete Analysis Report"))

        # Include content from each individual report
        for report_file in [basic_stats_report, sentiment_report, recommender_report]:
            with open(report_file, 'r', encoding='utf-8') as source:
                # Skip the header from individual reports
                lines = source.readlines()
                start_line = 0
                for i, line in enumerate(lines):
                    if line.strip() == '-' * 80 and i > 5:
                        start_line = i + 1
                        break

                f.write('\n\n')
                f.write(''.join(lines[start_line:]))

        # Add clustering report if available
        if df_with_clusters is not None and product_clusters is not None:
            with open(clustering_report, 'r', encoding='utf-8') as source:
                lines = source.readlines()
                start_line = 0
                for i, line in enumerate(lines):
                    if line.strip() == '-' * 80 and i > 5:
                        start_line = i + 1
                        break

                f.write('\n\n')
                f.write(''.join(lines[start_line:]))

        # Add categories report if available
        if 'product_category' in df.columns:
            with open(categories_report, 'r', encoding='utf-8') as source:
                lines = source.readlines()
                start_line = 0
                for i, line in enumerate(lines):
                    if line.strip() == '-' * 80 and i > 5:
                        start_line = i + 1
                        break

                f.write('\n\n')
                f.write(''.join(lines[start_line:]))

    print(f"Complete analysis report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    # Example usage
    input_file = "results\\cell_phones_reviews.csv"

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        # Generate basic report as an example
        generate_basic_stats_report(df)
    else:
        print(f"Input file '{input_file}' not found. Please run the full analysis first.")
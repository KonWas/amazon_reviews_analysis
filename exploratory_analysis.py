import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import matplotlib.dates as mdates
import re
import os

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = "visualizations"
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)


def get_viz_path(filename):
    """Get the full path for a visualization file"""
    return os.path.join(VISUALIZATIONS_DIR, filename)


def basic_stats(df):
    """
    Calculate basic statistics for the dataset

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    print("Dataset Information:")
    print(f"Number of reviews: {len(df)}")
    print(f"Number of unique products: {df['product_id'].nunique()}")
    print(f"Number of unique users: {df['user_id'].nunique()}")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values:")
    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count} ({count / len(df) * 100:.2f}%)")

    # Rating distribution
    rating_counts = df['score'].value_counts().sort_index()
    print("\nRating distribution:")
    for rating, count in rating_counts.items():
        print(f"Rating {rating}: {count} reviews ({count / len(df) * 100:.2f}%)")

    # Average rating
    print(f"\nAverage rating: {df['score'].mean():.2f}")

    # Review length statistics
    df['review_length'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    print(f"\nAverage review length (words): {df['review_length'].mean():.2f}")
    print(f"Median review length (words): {df['review_length'].median():.2f}")
    print(f"Shortest review (words): {df['review_length'].min()}")
    print(f"Longest review (words): {df['review_length'].max()}")


def visualize_rating_distribution(df):
    """
    Visualize the distribution of ratings

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='score', data=df, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')

    # Add count and percentage annotations
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 0.1,
                f'{height}\n({height / total * 100:.1f}%)',
                ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(get_viz_path('ea_rating_distribution.png'))
    print("Rating distribution visualization saved as 'ea_rating_distribution.png'")


def analyze_review_length(df):
    """
    Analyze the length of reviews

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Add review length column if it doesn't exist
    if 'review_length' not in df.columns:
        df['review_length'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

    # Visualize review length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['review_length'], bins=50, kde=True)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Review Length (words)')
    plt.ylabel('Count')

    # Add a vertical line for the mean
    plt.axvline(df['review_length'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["review_length"].mean():.2f}')

    # Add a vertical line for the median
    plt.axvline(df['review_length'].median(), color='green', linestyle='--',
                label=f'Median: {df["review_length"].median():.2f}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_review_length_distribution.png'))
    print("Review length distribution visualization saved as 'ea_review_length_distribution.png'")

    # Analyze review length by rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='score', y='review_length', data=df)
    plt.title('Review Length by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Review Length (words)')
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_review_length_by_rating.png'))
    print("Review length by rating visualization saved as 'ea_review_length_by_rating.png'")

    # Calculate average review length by rating
    avg_length_by_rating = df.groupby('score')['review_length'].mean().reset_index()
    print("\nAverage review length by rating:")
    for _, row in avg_length_by_rating.iterrows():
        print(f"Rating {row['score']}: {row['review_length']:.2f} words")


def analyze_review_helpfulness(df):
    """
    Analyze the helpfulness of reviews

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Calculate helpfulness ratio
    df['helpfulness_ratio'] = df.apply(
        lambda row: row['helpful_votes'] / row['total_votes'] if row['total_votes'] > 0 else np.nan,
        axis=1
    )

    # Filter reviews with at least 1 vote
    voted_reviews = df[df['total_votes'] > 0]

    print(f"\nReviews with at least one vote: {len(voted_reviews)} ({len(voted_reviews) / len(df) * 100:.2f}%)")
    print(f"Average helpfulness ratio: {voted_reviews['helpfulness_ratio'].mean():.2f}")

    # Visualize helpfulness ratio distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(voted_reviews['helpfulness_ratio'], bins=20, kde=True)
    plt.title('Distribution of Helpfulness Ratio')
    plt.xlabel('Helpfulness Ratio (helpful votes / total votes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_helpfulness_ratio_distribution.png'))
    print("Helpfulness ratio distribution visualization saved as 'ea_helpfulness_ratio_distribution.png'")

    # Analyze helpfulness by rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='score', y='helpfulness_ratio', data=voted_reviews)
    plt.title('Helpfulness Ratio by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Helpfulness Ratio')
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_helpfulness_by_rating.png'))
    print("Helpfulness by rating visualization saved as 'ea_helpfulness_by_rating.png'")

    # Calculate average helpfulness by rating
    avg_helpfulness_by_rating = voted_reviews.groupby('score')['helpfulness_ratio'].mean().reset_index()
    print("\nAverage helpfulness ratio by rating:")
    for _, row in avg_helpfulness_by_rating.iterrows():
        print(f"Rating {row['score']}: {row['helpfulness_ratio']:.2f}")


def analyze_temporal_trends(df):
    """
    Analyze temporal trends in the reviews

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Convert review_date to datetime
    df['review_date'] = pd.to_datetime(df['review_date'])

    # Group by date and calculate average rating and count
    daily_reviews = df.groupby('review_date').agg({
        'score': ['mean', 'count']
    }).reset_index()

    # Flatten the column names
    daily_reviews.columns = ['review_date', 'avg_rating', 'num_reviews']

    # Sort by date
    daily_reviews = daily_reviews.sort_values('review_date')

    # Visualize trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot average rating over time
    ax1.plot(daily_reviews['review_date'], daily_reviews['avg_rating'], marker='o', markersize=3)
    ax1.set_title('Average Rating Over Time')
    ax1.set_ylabel('Average Rating')
    ax1.set_ylim(1, 5)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot number of reviews over time
    ax2.plot(daily_reviews['review_date'], daily_reviews['num_reviews'], marker='o', markersize=3, color='green')
    ax2.set_title('Number of Reviews Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Reviews')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Format the date axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(get_viz_path('ea_temporal_trends.png'))
    print("Temporal trends visualization saved as 'ea_temporal_trends.png'")

    # Calculate monthly trends
    df['year_month'] = df['review_date'].dt.strftime('%Y-%m')
    monthly_reviews = df.groupby('year_month').agg({
        'score': ['mean', 'count']
    }).reset_index()

    # Flatten the column names
    monthly_reviews.columns = ['year_month', 'avg_rating', 'num_reviews']

    # Sort by year_month
    monthly_reviews = monthly_reviews.sort_values('year_month')

    print("\nMonthly trends (first 5 months):")
    print(monthly_reviews.head())


def create_wordcloud(df):
    """
    Create a word cloud for positive and negative reviews

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Filter positive (4-5 stars) and negative (1-2 stars) reviews
    positive_reviews = df[df['score'] >= 4]['text'].dropna()
    negative_reviews = df[df['score'] <= 2]['text'].dropna()

    # Function to create and save a word cloud
    def generate_wordcloud(text_series, title, filename):
        text = ' '.join(text_series.astype(str))

        # Clean the text - remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, contour_width=3).generate(
            text)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(get_viz_path(filename))
        print(f"Word cloud visualization saved as '{filename}'")

    # Generate word clouds
    generate_wordcloud(positive_reviews, 'Word Cloud - Positive Reviews', 'ea_positive_wordcloud.png')
    generate_wordcloud(negative_reviews, 'Word Cloud - Negative Reviews', 'ea_negative_wordcloud.png')


def analyze_top_products_and_users(df):
    """
    Analyze top products and users in the dataset

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """
    # Top products by number of reviews
    top_products_by_reviews = df.groupby(['product_id', 'product_title']).size().reset_index(name='num_reviews')
    top_products_by_reviews = top_products_by_reviews.sort_values('num_reviews', ascending=False).head(10)

    print("\nTop 10 products by number of reviews:")
    for i, (_, row) in enumerate(top_products_by_reviews.iterrows(), 1):
        print(f"{i}. {row['product_title']} (ID: {row['product_id']}): {row['num_reviews']} reviews")

    # Top products by average rating (with at least 5 reviews)
    products_with_min_reviews = df.groupby('product_id').filter(lambda x: len(x) >= 5)
    top_products_by_rating = products_with_min_reviews.groupby(['product_id', 'product_title'])[
        'score'].mean().reset_index(name='avg_rating')
    top_products_by_rating = top_products_by_rating.sort_values('avg_rating', ascending=False).head(10)

    print("\nTop 10 products by average rating (with at least 5 reviews):")
    for i, (_, row) in enumerate(top_products_by_rating.iterrows(), 1):
        print(f"{i}. {row['product_title']} (ID: {row['product_id']}): {row['avg_rating']:.2f} average rating")

    # Top users by number of reviews
    top_users_by_reviews = df.groupby(['user_id', 'profile_name']).size().reset_index(name='num_reviews')
    top_users_by_reviews = top_users_by_reviews.sort_values('num_reviews', ascending=False).head(10)

    print("\nTop 10 users by number of reviews:")
    for i, (_, row) in enumerate(top_users_by_reviews.iterrows(), 1):
        print(f"{i}. {row['profile_name']} (ID: {row['user_id']}): {row['num_reviews']} reviews")


def analyze_product_categories(df):
    """
    Try to identify product categories from product titles

    Args:
        df (pandas.DataFrame): DataFrame with review data
    """

    # Function to extract product category
    def extract_category(title):
        if pd.isna(title):
            return 'Unknown'

        title = str(title).lower()

        categories = [
            ('case', 'Case/Cover'),
            ('cover', 'Case/Cover'),
            ('bluetooth', 'Bluetooth Accessory'),
            ('headset', 'Bluetooth Accessory'),
            ('charger', 'Charger/Cable'),
            ('cable', 'Charger/Cable'),
            ('screen protector', 'Screen Protector'),
            ('protector', 'Screen Protector'),
            ('car', 'Car Accessory'),
            ('earphone', 'Earphone/Headphone'),
            ('headphone', 'Earphone/Headphone'),
            ('earbud', 'Earphone/Headphone'),
            ('phone', 'Phone'),
            ('battery', 'Battery'),
            ('holder', 'Holder/Mount'),
            ('mount', 'Holder/Mount'),
            ('adapter', 'Adapter'),
            ('memory', 'Memory/Storage')
        ]

        for keyword, category in categories:
            if keyword in title:
                return category

        return 'Other'

    # Add category column
    df['product_category'] = df['product_title'].apply(extract_category)

    # Count reviews by category
    category_counts = df['product_category'].value_counts()

    print("\nProduct categories by number of reviews:")
    for category, count in category_counts.items():
        print(f"{category}: {count} reviews ({count / len(df) * 100:.2f}%)")

    # Visualize categories
    plt.figure(figsize=(12, 6))
    sns.countplot(y='product_category', data=df, order=category_counts.index)
    plt.title('Number of Reviews by Product Category')
    plt.xlabel('Count')
    plt.ylabel('Product Category')
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_category_distribution.png'))
    print("Category distribution visualization saved as 'ea_category_distribution.png'")

    # Average rating by category
    avg_rating_by_category = df.groupby('product_category')['score'].mean().sort_values(ascending=False)

    print("\nAverage rating by category:")
    for category, avg_rating in avg_rating_by_category.items():
        print(f"{category}: {avg_rating:.2f}")

    # Visualize average rating by category
    plt.figure(figsize=(12, 6))
    sns.barplot(x='score', y='product_category', data=df, estimator=np.mean, order=avg_rating_by_category.index)
    plt.title('Average Rating by Product Category')
    plt.xlabel('Average Rating')
    plt.ylabel('Product Category')
    plt.tight_layout()
    plt.savefig(get_viz_path('ea_category_ratings.png'))
    print("Category ratings visualization saved as 'ea_category_ratings.png'")


if __name__ == "__main__":
    # Example usage
    input_file = "results\\cell_phones_reviews.csv"

    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews")

    # Basic statistics
    basic_stats(df)

    # Visualize rating distribution
    visualize_rating_distribution(df)

    # Analyze review length
    analyze_review_length(df)

    # Analyze review helpfulness
    analyze_review_helpfulness(df)

    # Analyze temporal trends
    analyze_temporal_trends(df)

    # Create word clouds
    create_wordcloud(df)

    # Analyze top products and users
    analyze_top_products_and_users(df)

    # Analyze product categories
    analyze_product_categories(df)

    print("\nExploratory analysis complete. Check the generated visualizations for detailed insights.")
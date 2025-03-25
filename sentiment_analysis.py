import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Create visualizations directory if it doesn't exist
VISUALIZATIONS_DIR = "visualizations"
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)


def get_viz_path(filename):
    """Get the full path for a visualization file"""
    return os.path.join(VISUALIZATIONS_DIR, filename)


def download_nltk_resources():
    """Download necessary NLTK resources"""
    try:
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")


def analyze_sentiment_nltk(df):
    """
    Analyze sentiment using NLTK's VADER SentimentIntensityAnalyzer

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        pandas.DataFrame: DataFrame with sentiment scores added
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to review text
    result_df['sentiment_scores'] = result_df['text'].apply(
        lambda text: sid.polarity_scores(str(text)) if pd.notnull(text) else None
    )

    # Extract compound score
    result_df['compound_score'] = result_df['sentiment_scores'].apply(
        lambda score: score['compound'] if score else None
    )

    # Classify sentiment
    result_df['sentiment'] = result_df['compound_score'].apply(
        lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
    )

    return result_df


def analyze_sentiment_based_on_rating(df):
    """
    Classify sentiment based on the rating/score

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        pandas.DataFrame: DataFrame with sentiment based on rating
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # Classify sentiment based on rating
    result_df['rating_sentiment'] = result_df['score'].apply(
        lambda score: 'positive' if score >= 4 else ('negative' if score <= 2 else 'neutral')
    )

    return result_df


def compare_sentiment_methods(df):
    """
    Compare sentiment analysis methods

    Args:
        df (pandas.DataFrame): DataFrame with sentiment analysis results
    """
    # Count ratings-based sentiment
    rating_sentiment_counts = df['rating_sentiment'].value_counts()
    print("\nSentiment distribution based on ratings:")
    print(rating_sentiment_counts)

    # Count NLTK-based sentiment
    nltk_sentiment_counts = df['sentiment'].value_counts()
    print("\nSentiment distribution based on NLTK analysis:")
    print(nltk_sentiment_counts)

    # Create a comparison table
    comparison_df = pd.DataFrame({
        'rating_sentiment': rating_sentiment_counts,
        'nltk_sentiment': nltk_sentiment_counts
    }).fillna(0).astype(int)

    print("\nComparison of sentiment analysis methods:")
    print(comparison_df)

    # Confusion matrix between methods
    confusion = pd.crosstab(df['rating_sentiment'], df['sentiment'],
                            rownames=['Rating Sentiment'],
                            colnames=['NLTK Sentiment'])
    print("\nConfusion matrix between methods:")
    print(confusion)


def visualize_sentiment_distribution(df):
    """
    Visualize sentiment distribution

    Args:
        df (pandas.DataFrame): DataFrame with sentiment analysis results
    """
    plt.figure(figsize=(12, 6))

    # Plot 1: Sentiment distribution based on ratings
    plt.subplot(1, 2, 1)
    sns.countplot(x='rating_sentiment', data=df)
    plt.title('Sentiment Based on Ratings')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Plot 2: Sentiment distribution based on NLTK analysis
    plt.subplot(1, 2, 2)
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Based on NLTK Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(get_viz_path('sa_sentiment_distribution.png'))
    print("Sentiment distribution visualization saved as 'sa_sentiment_distribution.png'")

    # Plot the relationship between score and compound sentiment
    plt.figure(figsize=(10, 6))
    plt.scatter(df['score'], df['compound_score'], alpha=0.5)
    plt.title('Relationship Between Rating Score and Sentiment Compound Score')
    plt.xlabel('Rating Score')
    plt.ylabel('Sentiment Compound Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(get_viz_path('sa_score_vs_sentiment.png'))
    print("Score vs sentiment visualization saved as 'sa_score_vs_sentiment.png'")


def analyze_positive_negative_terms(df):
    """
    Analyze most common terms in positive and negative reviews

    Args:
        df (pandas.DataFrame): DataFrame with sentiment analysis results
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter

    # Filter for positive and negative reviews
    positive_reviews = df[df['rating_sentiment'] == 'positive']['text'].dropna()
    negative_reviews = df[df['rating_sentiment'] == 'negative']['text'].dropna()

    # Get stopwords
    stop_words = set(stopwords.words('english'))

    # Function to get most common words
    def get_common_words(texts, n=20):
        all_words = []
        for text in texts:
            words = word_tokenize(str(text).lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            all_words.extend(words)
        return Counter(all_words).most_common(n)

    # Get most common words in positive and negative reviews
    positive_common = get_common_words(positive_reviews)
    negative_common = get_common_words(negative_reviews)

    print("\nMost common words in positive reviews:")
    for word, count in positive_common:
        print(f"{word}: {count}")

    print("\nMost common words in negative reviews:")
    for word, count in negative_common:
        print(f"{word}: {count}")

    # Visualize most common words
    plt.figure(figsize=(12, 10))

    # Plot 1: Most common words in positive reviews
    plt.subplot(2, 1, 1)
    words, counts = zip(*positive_common)
    plt.barh(words, counts)
    plt.title('Most Common Words in Positive Reviews')
    plt.xlabel('Count')

    # Plot 2: Most common words in negative reviews
    plt.subplot(2, 1, 2)
    words, counts = zip(*negative_common)
    plt.barh(words, counts)
    plt.title('Most Common Words in Negative Reviews')
    plt.xlabel('Count')

    plt.tight_layout()
    plt.savefig(get_viz_path('sa_common_words.png'))
    print("Common words visualization saved as 'sa_common_words.png'")


def train_sentiment_classifier(df):
    """
    Train a machine learning model to classify sentiment

    Args:
        df (pandas.DataFrame): DataFrame with review data

    Returns:
        tuple: (model, vectorizer) - trained model and vectorizer
    """
    # Filter out rows with missing text
    filtered_df = df.dropna(subset=['text', 'score'])

    # Create target variable (positive: >=4, negative: <=2)
    filtered_df['target'] = filtered_df['score'].apply(
        lambda score: 1 if score >= 4 else (0 if score <= 2 else None)
    )

    # Filter out neutral ratings
    filtered_df = filtered_df.dropna(subset=['target'])

    # Convert target to integer
    filtered_df['target'] = filtered_df['target'].astype(int)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(filtered_df['text'].astype(str))
    y = filtered_df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    print("\nSentiment Classification Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(get_viz_path('sa_confusion_matrix.png'))
    print("Confusion matrix visualization saved as 'sa_confusion_matrix.png'")

    return model, vectorizer


if __name__ == "__main__":
    # Example usage
    input_file = "results\\cell_phones_reviews.csv"

    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews")

    # Download NLTK resources
    download_nltk_resources()

    # Analyze sentiment using NLTK
    df = analyze_sentiment_nltk(df)

    # Analyze sentiment based on ratings
    df = analyze_sentiment_based_on_rating(df)

    # Compare sentiment analysis methods
    compare_sentiment_methods(df)

    # Visualize sentiment distribution
    visualize_sentiment_distribution(df)

    # Analyze positive and negative terms
    analyze_positive_negative_terms(df)

    # Train a sentiment classifier
    model, vectorizer = train_sentiment_classifier(df)

    # Save results to CSV
    df.to_csv('sentiment_analysis_results.csv', index=False)
    print("Sentiment analysis results saved to 'sentiment_analysis_results.csv'")
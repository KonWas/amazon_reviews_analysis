import os
import time
import pandas as pd
import nltk

# Import modules
from data_parser import parse_reviews_file, save_to_csv
import sentiment_analysis
import recommender_system
import clustering
import exploratory_analysis
import report_generator


def ensure_nltk_resources():
    """
    Ensure all required NLTK resources are downloaded
    """
    resources = ['punkt', 'vader_lexicon', 'stopwords', 'wordnet']

    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)


# Create folders for organizing outputs
VISUALIZATIONS_DIR = "visualizations"
RESULTS_DIR = "results"


def get_results_path(filename):
    """
    Get the full path for a results file

    Args:
        filename (str): Name of the results file

    Returns:
        str: Full path including the results directory
    """
    return os.path.join(RESULTS_DIR, filename)


def main():
    """
    Main function to run the entire analysis pipeline
    """
    print("=" * 80)
    print("AMAZON CELL PHONES & ACCESSORIES REVIEWS ANALYSIS")
    print("=" * 80)

    # Ensure directories exist
    if not os.path.exists(VISUALIZATIONS_DIR):
        os.makedirs(VISUALIZATIONS_DIR)
        print(f"Created directory '{VISUALIZATIONS_DIR}' for storing visualizations")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory '{RESULTS_DIR}' for storing results")

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    # Check if input file exists
    input_file = "Cell_Phones_&_Accessories.txt"
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure the file is in the current directory.")
        return

    # Step 1: Parse the data
    print("\nStep 1: Parsing the data")
    print("-" * 50)
    start_time = time.time()

    try:
        reviews_df = parse_reviews_file(input_file)
        print(f"Parsed {len(reviews_df)} reviews in {time.time() - start_time:.2f} seconds")

        # Save to CSV
        csv_file = get_results_path("cell_phones_reviews.csv")
        save_to_csv(reviews_df, csv_file)
    except Exception as e:
        print(f"Error parsing data: {e}")
        return

    # Step 2: Exploratory Data Analysis
    print("\nStep 2: Exploratory Data Analysis")
    print("-" * 50)
    start_time = time.time()

    try:
        # Basic statistics
        exploratory_analysis.basic_stats(reviews_df)

        # Visualize rating distribution
        exploratory_analysis.visualize_rating_distribution(reviews_df)

        # Analyze review length
        exploratory_analysis.analyze_review_length(reviews_df)

        # Analyze review helpfulness
        exploratory_analysis.analyze_review_helpfulness(reviews_df)

        # Analyze temporal trends
        exploratory_analysis.analyze_temporal_trends(reviews_df)

        # Create word clouds
        exploratory_analysis.create_wordcloud(reviews_df)

        # Analyze top products and users
        exploratory_analysis.analyze_top_products_and_users(reviews_df)

        # Analyze product categories
        exploratory_analysis.analyze_product_categories(reviews_df)

        print(f"Completed exploratory analysis in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in exploratory analysis: {e}")

    # Step 3: Sentiment Analysis
    print("\nStep 3: Sentiment Analysis")
    print("-" * 50)
    start_time = time.time()

    try:
        # Download NLTK resources
        sentiment_analysis.download_nltk_resources()

        # Analyze sentiment using NLTK
        reviews_df = sentiment_analysis.analyze_sentiment_nltk(reviews_df)

        # Analyze sentiment based on ratings
        reviews_df = sentiment_analysis.analyze_sentiment_based_on_rating(reviews_df)

        # Compare sentiment analysis methods
        sentiment_analysis.compare_sentiment_methods(reviews_df)

        # Visualize sentiment distribution
        sentiment_analysis.visualize_sentiment_distribution(reviews_df)

        # Analyze positive and negative terms
        sentiment_analysis.analyze_positive_negative_terms(reviews_df)

        # Train a sentiment classifier
        model, vectorizer = sentiment_analysis.train_sentiment_classifier(reviews_df)

        # Save results to CSV
        reviews_df.to_csv(get_results_path('sentiment_analysis_results.csv'), index=False)

        print(f"Completed sentiment analysis in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")

    # Step 4: Recommender System
    print("\nStep 4: Recommender System")
    print("-" * 50)
    start_time = time.time()

    try:
        # Analyze rating distribution
        recommender_system.analyze_rating_distribution(reviews_df)

        # Generate example recommendations
        recommender_system.generate_example_recommendations(reviews_df)

        # Evaluate the recommendation system
        recommender_system.evaluate_recommendations(reviews_df)

        print(f"Completed recommender system analysis in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in recommender system: {e}")

    # Step 5: Clustering
    print("\nStep 5: Clustering")
    print("-" * 50)
    start_time = time.time()

    try:
        # Download NLTK resources
        clustering.download_nltk_resources()

        # Cluster reviews by text content
        df_with_clusters = clustering.cluster_reviews_by_text(reviews_df)

        # Analyze the clusters
        clustering.analyze_clusters(df_with_clusters)

        # Cluster products by review characteristics
        product_clusters = clustering.cluster_products_by_reviews(reviews_df)

        # Save results
        df_with_clusters.to_csv(get_results_path('review_clusters.csv'), index=False)
        product_clusters.to_csv(get_results_path('product_clusters.csv'), index=False)

        print(f"Completed clustering analysis in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in clustering: {e}")

    print("\nAnalysis complete! Check the output files and visualizations for results.")

    # Step 6: Generate Reports
    print("\nStep 6: Generating Reports")
    print("-" * 50)
    start_time = time.time()

    try:
        # Generate individual reports
        report_generator.generate_basic_stats_report(reviews_df)
        report_generator.generate_sentiment_analysis_report(reviews_df)
        report_generator.generate_recommender_system_report(reviews_df)

        # If clustering data is available
        try:
            df_with_clusters = pd.read_csv(get_results_path('review_clusters.csv'))
            product_clusters = pd.read_csv(get_results_path('product_clusters.csv'))
            report_generator.generate_clustering_report(df_with_clusters, product_clusters)
        except:
            print("Clustering data not available, skipping clustering report")

        # Generate product categories report if available
        if 'product_category' in reviews_df.columns:
            report_generator.generate_product_categories_report(reviews_df)

        # Generate full report
        try:
            report_generator.generate_full_report(
                reviews_df,
                df_with_clusters if 'df_with_clusters' in locals() else None,
                product_clusters if 'product_clusters' in locals() else None
            )
        except Exception as e:
            print(f"Error generating full report: {e}")

        print(f"Completed report generation in {time.time() - start_time:.2f} seconds")
        print("Reports are available in the 'reports' directory")
    except Exception as e:
        print(f"Error generating reports: {e}")


if __name__ == "__main__":
    main()
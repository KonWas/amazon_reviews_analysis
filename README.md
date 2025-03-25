# Amazon Cell Phones & Accessories Reviews Analysis

This project performs a comprehensive analysis of Amazon product reviews for cell phones and accessories, extracting insights into consumer sentiment, product performance, and market trends.

## Project Overview

The analysis pipeline processes a large dataset of Amazon reviews, applying several analytical techniques:

1. **Data Parsing and Cleaning**: Extracts structured data from the raw review text file
2. **Exploratory Data Analysis**: Investigates basic patterns and statistics in the review data
3. **Sentiment Analysis**: Examines customer sentiment using both rating-based and NLP approaches
4. **Recommender System**: Identifies similar products and makes product recommendations
5. **Clustering Analysis**: Groups reviews and products to identify underlying patterns
6. **Report Generation**: Creates detailed text reports of all analysis results

## Project Structure

```
amazon_reviews_analysis/
│
├── Cell_Phones_&_Accessories.txt     # Original data file
│
├── main.py                           # Main script to run the entire pipeline
├── data_parser.py                    # Parses raw data into structured format
├── exploratory_analysis.py           # Performs basic statistical analysis
├── sentiment_analysis.py             # Analyzes sentiment in reviews
├── recommender_system.py             # Implements recommendation algorithms
├── clustering.py                     # Performs clustering analysis
├── report_generator.py               # Generates text reports
│
├── results/                          # Contains generated CSV files
│   ├── cell_phones_reviews.csv
│   ├── sentiment_analysis_results.csv
│   ├── review_clusters.csv
│   └── product_clusters.csv
│
├── visualizations/                   # Contains generated plots and charts
│   ├── rating_distribution.png
│   ├── sentiment_distribution.png
│   ├── review_clusters_pca.png
│   └── [other visualization files]
│
├── reports/                          # Contains generated text reports
│   ├── basic_stats_report.txt
│   ├── sentiment_analysis_report.txt
│   ├── recommender_system_report.txt
│   ├── clustering_report.txt
│   ├── product_categories_report.txt
│   └── full_analysis_report.txt
│
└── amazon_analysis_summary.md        # Markdown summary of key findings
```

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk
  - scikit-learn
  - scipy
  - wordcloud

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place the `Cell_Phones_&_Accessories.txt` file in the project directory
2. Run the main script to execute the full analysis pipeline:
   ```bash
   python main.py
   ```
3. Results will be organized into three directories:
   - `results/`: CSV files containing processed data
   - `visualizations/`: PNG files with all generated charts and plots
   - `reports/`: Text reports summarizing the analysis findings

Alternatively, you can run individual scripts for specific parts of the analysis:
```bash
python exploratory_analysis.py  # Run only exploratory analysis
python sentiment_analysis.py    # Run only sentiment analysis
```

## Key Findings

The analysis reveals several interesting patterns in consumer behavior and product performance:

- Overall average rating is 3.52/5.0, with a clear J-shaped distribution (many 5-star and 1-star reviews)
- Significant differences in satisfaction across product categories (Holders/Mounts highest at 3.89, Earphones lowest at 3.01)
- Moderate reviews (3-star) contain the most detailed explanations (avg. 117 words vs. 91 words for 1-star and 5-star reviews)
- Bluetooth accessories dominate the market with 32.14% of all reviews
- Most users (68,040 unique users) leave only one review, with just 29 users submitting more than 10 reviews

For a detailed summary of findings, see the reports directory.

import pandas as pd
import re
from datetime import datetime


def parse_reviews_file(file_path):
    """
    Parse the Amazon Cell Phones & Accessories reviews file and convert it to a pandas DataFrame.

    Args:
        file_path (str): Path to the reviews file

    Returns:
        pandas.DataFrame: DataFrame containing structured review data
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into individual reviews
    reviews_raw = re.split(r'\n\s*\n', content)

    reviews_data = []
    for review_raw in reviews_raw:
        if not review_raw.strip():
            continue

        # Dictionary to store review information
        review = {}

        # Extract fields using regex patterns
        review['product_id'] = re.search(r'product/productId: (.*?)(?:\n|$)', review_raw)
        review['product_title'] = re.search(r'product/title: (.*?)(?:\n|$)', review_raw)
        review['product_price'] = re.search(r'product/price: (.*?)(?:\n|$)', review_raw)
        review['user_id'] = re.search(r'review/userId: (.*?)(?:\n|$)', review_raw)
        review['profile_name'] = re.search(r'review/profileName: (.*?)(?:\n|$)', review_raw)
        helpfulness = re.search(r'review/helpfulness: (\d+)/(\d+)(?:\n|$)', review_raw)
        review['score'] = re.search(r'review/score: (.*?)(?:\n|$)', review_raw)
        review['time'] = re.search(r'review/time: (\d+)(?:\n|$)', review_raw)
        review['summary'] = re.search(r'review/summary: (.*?)(?:\n|$)', review_raw)
        review['text'] = re.search(r'review/text: (.*?)(?:\n|$)', review_raw)

        # Process extracted data
        if review['product_id']:
            review['product_id'] = review['product_id'].group(1)
        else:
            continue  # Skip if no product ID

        if review['product_title']:
            review['product_title'] = review['product_title'].group(1)
        else:
            review['product_title'] = None

        if review['product_price']:
            price = review['product_price'].group(1)
            if price.lower() != 'unknown':
                try:
                    review['product_price'] = float(price)
                except ValueError:
                    review['product_price'] = None
            else:
                review['product_price'] = None
        else:
            review['product_price'] = None

        if review['user_id']:
            review['user_id'] = review['user_id'].group(1)
            if review['user_id'].lower() == 'unknown':
                review['user_id'] = None
        else:
            review['user_id'] = None

        if review['profile_name']:
            review['profile_name'] = review['profile_name'].group(1)
            if review['profile_name'].lower() == 'unknown':
                review['profile_name'] = None
        else:
            review['profile_name'] = None

        if helpfulness:
            review['helpful_votes'] = int(helpfulness.group(1))
            review['total_votes'] = int(helpfulness.group(2))
        else:
            review['helpful_votes'] = None
            review['total_votes'] = None

        if review['score']:
            try:
                review['score'] = float(review['score'].group(1))
            except ValueError:
                review['score'] = None
        else:
            review['score'] = None

        if review['time']:
            timestamp = int(review['time'].group(1))
            review['review_date'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        else:
            review['review_date'] = None

        if review['summary']:
            review['summary'] = review['summary'].group(1)
        else:
            review['summary'] = None

        if review['text']:
            review['text'] = review['text'].group(1)
        else:
            review['text'] = None

        reviews_data.append(review)

    # Create DataFrame
    df = pd.DataFrame(reviews_data)

    # Remove unnecessary columns
    if 'time' in df.columns:
        df = df.drop('time', axis=1)

    return df


def save_to_csv(df, output_path):
    """
    Save DataFrame to CSV file

    Args:
        df (pandas.DataFrame): DataFrame to save
        output_path (str): Path where CSV file will be saved
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_file = "Cell_Phones_&_Accessories.txt"
    output_file = "results\\cell_phones_reviews.csv"

    # Parse the file and create DataFrame
    reviews_df = parse_reviews_file(input_file)

    # Display information about the DataFrame
    print(f"Parsed {len(reviews_df)} reviews")
    print("\nDataFrame columns:")
    print(reviews_df.columns.tolist())

    print("\nDataFrame sample:")
    print(reviews_df.head())

    # Save to CSV
    save_to_csv(reviews_df, output_file)
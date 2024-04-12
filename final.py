from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import time
from FINAL_scrape_scoring_sentiment import get_product_details, get_scraping_link, scrape_amazon_reviews, initialize_classifier ,insert_product_info_to_mongodb
from transformers import pipeline
import plotly.graph_objs as go
import numpy as np
import random
import plotly.io as pio
from word_cloud import generate_word_cloud
import pymongo
from bson.objectid import ObjectId

app = Flask(__name__)

def scrape_amazon_and_save_to_excel(product_url):
    start_time = time.time()

    # Initializing the classifier
    model_name = "Recognai/zeroshot_selectra_medium"
    classifier = initialize_classifier(model_name)

    # Initializing the sentiment model
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    # Scraping and processing Amazon reviews
    excel_file_product_details = '.\\scrapping\\product_details.xlsx'
    excel_file_reviews = '.\\scrapping\\amazon_reviews.xlsx'

    product_details = get_product_details(product_url, excel_file_product_details)
    modified_url = get_scraping_link(product_url)
    print(product_details)

    if modified_url:
        star_ratings = ['one', 'two', 'three', 'four', 'five']
        candidate_labels = product_details["Features"]
        candidate_labels = candidate_labels[0].replace(" ", "").split(",")
        all_reviews = scrape_amazon_reviews(modified_url, star_ratings, candidate_labels, classifier, sentiment_model)

        # Save reviews to Excel
        df = pd.DataFrame(all_reviews)
        df.to_excel(excel_file_reviews, index=False)
        print('Excel is Ready!')

        # Insert product info to MongoDB
        product_id = insert_product_info_to_mongodb(product_url, product_details, all_reviews)

        end_time = time.time()
        execution_time = end_time - start_time
        print('Execution time:', execution_time, 'seconds')

        return product_id
    else:
        print('Invalid URL or pattern not found.')
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_url = request.form['product_url']
        product_id = scrape_amazon_and_save_to_excel(product_url)
        if product_id:
            return redirect(url_for('display', product_id=product_id))
        else:
            return render_template('failure.html')
    return render_template('index.html')

@app.route('/display/<product_id>', methods=['GET'])
def display(product_id):
    # Connect to the MongoDB cluster
    client = pymongo.MongoClient("mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net")

    # Access the database
    db = client["Full_Stack_Project"]

    # Access the collection
    collection = db["Amazon_Reviews"]

    # Query the collection for the document with the specified ID
    document = collection.find_one({"_id":  ObjectId(product_id)})

    if document:
        print('Document fetched')
        # Extract product details (features) and reviews
        product_details1 = document.get('Product_Details', {})
        reviews_data = document.get('Reviews', [])

    else:
        print("Document not found with the specified ID.")

    window_size = 30
    # Split the 'Features' string into a list
    features_list = product_details1['Features'].split(', ')
    # Convert reviews data to DataFrame
    review_df = pd.DataFrame(reviews_data)

    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df.dropna(subset=['date'], inplace=True)
    review_df.set_index('date', inplace=True)
    
    # Convert 'rating' column to numeric
    review_df['rating'] = pd.to_numeric(review_df['rating'], errors='coerce')

    # Convert 'sentiment' column to numeric
    review_df['sentiment'] = pd.to_numeric(review_df['sentiment'], errors='coerce')
    

    # Sort DataFrame by date index
    review_df.sort_index(inplace=True)

    # Calculate 2-week moving average of ratings
    ratings_2_weeks_ma = review_df['rating'].rolling(window=f'{window_size}D').mean()

    Default_pros_cons = {}
    Default_pros_cons['Pros'] = "Default pros1"
    Default_pros_cons['Cons'] = "Default Cons1"

    pros_dict = {}
    cons_dict = {}

    for feature in features_list:
        pros_dict[feature] = f"{feature} pros"
        cons_dict[feature] = f"{feature} cons"

    # Calculate 1-month rolling sum of positive sentiment (1) and total sentiment
    sentiment_1_month_sum = review_df['sentiment'].rolling(window=f'{window_size}D').sum()
    total_sentiment_1_month = review_df['sentiment'].rolling(window=f'{window_size}D').count()

    # Resample sentiment_1_month_sum to the end of each month and take the last value
    last_sentiment_monthly_sum = sentiment_1_month_sum.resample('M').last()

    # Resample total_sentiment_1_month to the end of each month and take the last value
    last_total_sentiment_monthly_count = total_sentiment_1_month.resample('M').last()

    # Calculate the percentage of positive sentiment within each 1-month window
    percentage_positive_sentiment = (last_sentiment_monthly_sum / last_total_sentiment_monthly_count) * 100

    # Create Plotly figure for the 2-week moving average ratings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratings_2_weeks_ma.index, y=ratings_2_weeks_ma.values, mode='lines', name=f'{window_size}-Day Moving Average Ratings'))

    # Add shaded regions for sentiment percentages
    for month_end_date, percent in percentage_positive_sentiment.items():
        start_of_month = month_end_date - pd.offsets.MonthBegin(1)
        color = 'green' if percent >= 50 else 'yellow' if percent >= 40 else 'red'
        fig.add_shape(type="rect", x0=start_of_month, y0=0, x1=month_end_date, y1=5, fillcolor=color, opacity=0.3)

    # Update layout
    fig.update_layout(title=f'{window_size}-Day Moving Average Ratings Over Time',
                      xaxis_title='Date',
                      yaxis_title=f'{window_size}-Day Moving Average Rating',
                      xaxis_tickangle=-45)

    # Save the plot as an HTML file
    plot_html1 = f"static/ratings_plot.html"
    fig.write_html(plot_html1)

    # Generate word cloud
    wordcloud_image_path1 = generate_word_cloud(review_df)

    # Render the template with product details and plot HTML
    return render_template('index copy.html', products=product_details1, features_list1=features_list, pros=pros_dict, cons=cons_dict, dft=Default_pros_cons)

if __name__ == '__main__':
    app.run(debug=True)

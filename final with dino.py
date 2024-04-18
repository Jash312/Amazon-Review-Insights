import asyncio
import os
import threading
import time
import uuid
from functools import lru_cache

import pandas as pd
import plotly.graph_objs as go
import redis
from bson.objectid import ObjectId
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline

import summarize_review
from FINAL_scrape_scoring_sentiment import get_Title, get_product_details, get_scraping_link, initialize_classifier, \
    insert_product_info_to_mongodb, scrape_amazon_reviews_async
from summarize_review import get_db
from word_cloud import generate_word_cloud

redis_messages = redis.Redis(
    host='redis-18614.c73.us-east-1-2.ec2.cloud.redislabs.com',
    port=18614,
    password='ntXzvQ3cVyPFkQB3lK0IIAkMu8RKu8mc', decode_responses=True)

# Load environment variables from .env file
load_dotenv()
mongo_uri = "mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net"
# Access the API key using os.environ.get()
api_key = os.environ.get("API_KEY")

app = Flask(__name__)


def red_msg(req_id, status, Messages, product_id=None):
    message1 = {
        "Status": status,
        "Message": Messages
    }

    if product_id is not None:
        message1['Product_id'] = str(product_id)

    redis_messages.hset(req_id, mapping=message1)


def check_exist(product_url):
    Title = get_Title(product_url)
    db = get_db(mongo_uri)

    # Access the collection
    collection = db["Amazon_Reviews"]
    result = collection.find_one({"Product_Details.Title": Title})

    if result:
        print('Already Existed', str(result.get('_id')))
        return str(result.get('_id'))


@lru_cache
def models_intializer():
    # Initializing the classifier
    model_name = "Recognai/zeroshot_selectra_medium"
    classifier = initialize_classifier(model_name)

    # Initializing the sentiment model
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, max_length=512,
                               truncation=True)

    return classifier, sentiment_model


async def scrape_amazon_and_save_to_excel(product_url, req_id):
    product_id = check_exist(product_url)
    if product_id:
        red_msg(req_id, "Completed", "99% Done -  Generating Insights", product_id)
        return product_id

    start_time = time.time()

    classifier, sentiment_model = models_intializer()

    # Scraping and processing Amazon reviews
    excel_file_product_details = '.\\scrapping\\product_details.xlsx'
    excel_file_reviews = '.\\scrapping\\amazon_reviews.xlsx'

    red_msg(req_id, "Incompleted", "5% Done - Scraping Started")

    product_details = get_product_details(product_url, excel_file_product_details)
    counter = 0
    while product_details["Features"] == [""]:
        product_details = get_product_details(product_url, excel_file_product_details)
        counter += 1
        if counter == 3:
            print('Invalid URL or pattern not found.')
            return None
    red_msg(req_id, "Incompleted", "15% Done - Product Details Retrived")

    modified_url = get_scraping_link(product_url)
    print(product_details)

    if modified_url:
        star_ratings = ['one', 'two', 'three', 'four', 'five']
        candidate_labels = product_details["Features"]
        candidate_labels = candidate_labels[0].replace(" ", "").split(",")
        red_msg(req_id, "Incompleted", "25% Done - Scrapping In Progress")
        all_reviews = await scrape_amazon_reviews_async(modified_url, star_ratings, candidate_labels, classifier,
                                                        sentiment_model)

        # Save reviews to Excel
        df = pd.DataFrame(all_reviews)
        df.to_excel(excel_file_reviews, index=False)
        print('Excel is Ready!')
        red_msg(req_id, "Incompleted", "45% Done - Scrapping Completed")

        # Insert product info to MongoDB
        product_id = insert_product_info_to_mongodb(product_url, product_details, all_reviews)
        red_msg(req_id, "Incompleted", "55% Done - Generating Action Items ")
        await summarize_review.main_async(product_id, "True", openai_key=api_key)
        red_msg(req_id, "Incompleted", "85% Done - Generating Pros and Cons")
        end_time = time.time()
        execution_time = end_time - start_time
        print('Execution time:', execution_time, 'seconds')
        red_msg(req_id, "Completed", f"99% Done - Generating Insights, it took {execution_time} seconds", product_id)
        return product_id
    else:
        print('Invalid URL or pattern not found.')
        return None


def scrape_and_redirect(product_url, req_id):
    try:
        loop = asyncio.new_event_loop()  # loop = asyncio.get_event_loop()
        loop.run_until_complete(scrape_amazon_and_save_to_excel(product_url, req_id))
        loop.close()
    except Exception as ex:
        print(f"Problem - {product_url} - {ex}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        req_id = str(uuid.uuid4())
        product_url = request.form['product_url']
        # Start a new thread to run the scraping task
        threading.Thread(target=scrape_and_redirect, args=(product_url, req_id)).start()
        # time.sleep(5000)
        return redirect(url_for('dino', req_id=req_id))
    return render_template('index.html')


@app.route('/status', methods=['GET'])
def get_status():
    req_id = request.args.get('request_id')
    if req_id is None:
        return {}
    return redis_messages.hgetall(req_id)


@app.route('/dino')
def dino():
    return render_template('dino.html')


@app.route('/display/<product_id>', methods=['GET'])
def display(product_id):
    # Access the database
    db = get_db(mongo_uri)

    # Access the collection
    collection = db["Amazon_Reviews"]

    # Query the collection for the document with the specified ID
    document = collection.find_one({"_id": ObjectId(product_id)})

    if document:
        print('Document fetched')
        # Extract product details (features) and reviews
        product_details1 = document.get('Product_Details', {})
        reviews_data = document.get('Reviews', [])
        summarized = document.get('Summary', {})
        action_items = document.get('ActionItems', {})

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

    Default_action_items = {}
    Default_action_items['Action Items'] = action_items

    pros_dict = {}
    cons_dict = {}

    for feature in features_list:
        pros_dict[feature] = summarized['feature_summary'][feature.replace(" ", "")]['pros']
        cons_dict[feature] = summarized['feature_summary'][feature.replace(" ", "")]['cons']

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
    fig.add_trace(go.Scatter(x=ratings_2_weeks_ma.index, y=ratings_2_weeks_ma.values, mode='lines',
                             name=f'{window_size}-Day Moving Average Ratings'))

    # Add shaded regions for sentiment percentages
    for month_end_date, percent in percentage_positive_sentiment.items():
        start_of_month = month_end_date - pd.offsets.MonthBegin(1)
        color = 'green' if percent >= 50 else 'yellow' if percent >= 40 else 'red'
        fig.add_shape(type="rect", x0=start_of_month, y0=0, x1=month_end_date, y1=5, fillcolor=color, opacity=0.3)

    # Update layout
    fig.update_layout(
        title=f'Average Rating and Monthly Sentiment over Time',
        xaxis_title='Date',
        yaxis_title=f'{window_size}-Day Moving Average Rating',
        xaxis_tickangle=-45,
        showlegend=False  # Set to True if you want to show Plotly's default legend
    )

    # Save the plot as an HTML file
    plot_html1 = f"static/rating_plot_{product_id}.html"
    fig.write_html(plot_html1)

    # Generate word cloud
    wordcloud_image_path1, word_freq = generate_word_cloud(review_df, product_id)

    # Render the template with product details and plot HTML
    return render_template('index_bootstrap.html', rating=plot_html1, word_cloud=wordcloud_image_path1,
                           products=product_details1, features_list1=features_list,
                           pros=pros_dict, cons=cons_dict, dft=Default_action_items, word_freq=list(word_freq.items()))


@app.route('/health')
def health():
    return {"message": "server is up"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

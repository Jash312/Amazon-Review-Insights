from flask import Flask, render_template, request
import pandas as pd
import time
from FINAL_scrape_scoring_sentiment import get_product_details, get_scraping_link, scrape_amazon_reviews, initialize_classifier ,insert_product_info_to_mongodb
from transformers import pipeline


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
    excel_file_product_details = 'product_details.xlsx'
    excel_file_reviews = 'amazon_reviews.xlsx'

    product_details = get_product_details(product_url, excel_file_product_details)
    modified_url = get_scraping_link(product_url)

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
        insert_product_info_to_mongodb(product_url, product_details, all_reviews)

        end_time = time.time()
        execution_time = end_time - start_time
        print('Execution time:', execution_time, 'seconds')

        return True, execution_time
    else:
        print('Invalid URL or pattern not found.')
        return False, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_url = request.form['product_url']
        success, execution_time = scrape_amazon_and_save_to_excel(product_url)
        if success:
            return render_template('success.html', execution_time=execution_time)
        else:
            return render_template('failure.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

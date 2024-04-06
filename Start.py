from flask import Flask, render_template, request
import random
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

app = Flask(__name__)

# Function to generate a random date within a specific range
def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + datetime.timedelta(days=random_days)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    # Get form inputs
    start_date = datetime.datetime.strptime(request.form['start_date'], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(request.form['end_date'], '%Y-%m-%d').date()
    num_reviews = int(request.form['num_reviews'])
    window_size = int(request.form['window_size'])

    # Generate random reviews
    reviews = []
    for _ in range(num_reviews):
        date = random_date(start_date, end_date)
        rating = random.randint(1, 5)
        review = "This is a sample review."
        sentiment = random.choice([1, 0])
        reviews.append({'date': date, 'rating': rating, 'review': review, 'sentiment': sentiment})

    # Convert reviews to DataFrame
    review_df = pd.DataFrame(reviews)
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df.set_index('date', inplace=True)

    # Sort DataFrame by date index
    review_df.sort_index(inplace=True)

    # Calculate 2-week moving average of ratings
    ratings_2_weeks_ma = review_df['rating'].rolling(window=f'{window_size}D').mean()

    # Calculate 1-month rolling sum of positive sentiment (1) and total sentiment
    sentiment_1_month_sum = review_df['sentiment'].rolling(window='30D').sum()
    total_sentiment_1_month = review_df['sentiment'].rolling(window='30D').count()

    # Resample sentiment_1_month_sum to the end of each month and take the last value
    last_sentiment_monthly_sum = sentiment_1_month_sum.resample('M').last()

    # Resample total_sentiment_1_month to the end of each month and take the last value
    last_total_sentiment_monthly_count = total_sentiment_1_month.resample('M').last()

    # Calculate the percentage of positive sentiment within each 1-month window
    percentage_positive_sentiment = (last_sentiment_monthly_sum / last_total_sentiment_monthly_count) * 100

    # Plot the 2-week moving average ratings over time
    plt.figure(figsize=(10, 6))
    ratings_2_weeks_ma.plot(linestyle='-')
    plt.title(f'{window_size}-Day Moving Average Ratings Over Time')
    plt.xlabel('Date')
    plt.ylabel(f'{window_size}-Day Moving Average Rating')
    plt.xticks(rotation=45)
    plt.xlim(review_df.index.min(), review_df.index.max())

    legend_patches = [
        Patch(color='green', alpha=0.3, label='Positive Sentiment (>= 50%)'),
        Patch(color='yellow', alpha=0.3, label='Neutral Sentiment (>= 40% and < 50%)'),
        Patch(color='red', alpha=0.3, label='Negative Sentiment (< 40%)')
    ]

    for month_end_date, percent in percentage_positive_sentiment.items():
        start_of_month = month_end_date - pd.offsets.MonthBegin(1)
        plt.axvspan(start_of_month, month_end_date , color='green' if percent >= 50 else ('yellow' if percent >= 40 else 'red'), alpha=0.3)

    plt.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    
    # Save the plot as an image
    plt.savefig('static/ratings_plot.png')
    plt.close()

    # Render the template with the plot
    return render_template('index.html', plot_img='ratings_plot.png')

if __name__ == '__main__':
    app.run(debug=True)

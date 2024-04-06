from flask import Flask, render_template
import  random
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

# Define start and end dates for the reviews
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2023, 12, 31)

# Generate 1000 reviews
reviews = []
for _ in range(1000):
    # Generate random date within the defined range
    date = random_date(start_date, end_date)
    
    # Generate random rating (between 1 and 5)
    rating = random.randint(1, 5)
    
    # Generate random review text
    review = "This is a sample review."
    
    # Assign positive or negative sentiment randomly
    sentiment = random.choice([1, 0])
    
    # Append review to the list
    reviews.append({'date': date, 'rating': rating, 'review': review, 'sentiment': sentiment})

# Assuming 'reviews' is your dataset containing reviews
review = pd.DataFrame(reviews)
review['date'] = pd.to_datetime(review['date'])
review.set_index('date', inplace=True)

# Sort the DataFrame by the date index in ascending order
review.sort_index(inplace=True)

# Calculate the 2-week moving average of ratings
ratings_2_weeks_ma = review['rating'].rolling(window='30D').mean()

# Calculate the 1-month rolling sum of positive sentiment (1) and total sentiment
sentiment_1_month_sum = review['sentiment'].rolling(window='30D').sum()
total_sentiment_1_month = review['sentiment'].rolling(window='30D').count()

# Resample sentiment_1_month_sum to the end of each month and take the last value
last_sentiment_monthly_sum = sentiment_1_month_sum.resample('M').last()

# Resample total_sentiment_1_month to the end of each month and take the last value
last_total_sentiment_monthly_count = total_sentiment_1_month.resample('M').last()

# Calculate the percentage of positive sentiment within each 1-month window
percentage_positive_sentiment = (last_sentiment_monthly_sum / last_total_sentiment_monthly_count) * 100
@app.route('/')
def index():
    # Plot the 2-week moving average ratings over time
    plt.figure(figsize=(10, 6))
    ratings_2_weeks_ma.plot(linestyle='-')
    plt.title('30-Day Moving Average Ratings Over Time')
    plt.xlabel('Date')
    plt.ylabel('30-Day Moving Average Rating')
    plt.xticks(rotation=45)
    plt.xlim(review.index.min(), review.index.max())  # Set x-axis limits to the minimum and maximum dates

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
    plt.savefig('static/ratings_plot.png')  # Save the plot as a static image file
    plt.close()  # Close the plot to free memory
    
    # Render the template with the plot
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

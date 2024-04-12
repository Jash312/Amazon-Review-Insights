from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import random
import plotly.io as pio
from word_cloud import generate_word_cloud

app = Flask(__name__)

@app.route('/')
def index():
    # Read product details from a DataFrame or any other data source
    product_details = pd.read_excel('.\\scrapping\\product_details.xlsx').to_dict(orient='records')
    # Get the 'Features' from the first record
    features_string = product_details[0]['Features']

    # Split the 'Features' string into a list
    features_list = features_string.split(', ')

    print(features_list)
    # Convert reviews to DataFrame
    review_df = pd.read_excel(".\\scrapping\\amazon_reviews.xlsx")
    window_size = 120
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_df.set_index('date', inplace=True)


    
    # Sort DataFrame by date index
    review_df.sort_index(inplace=True)

    # Calculate 2-week moving average of ratings
    ratings_2_weeks_ma = review_df['rating'].rolling(window=f'{window_size}D').mean()

    review_df['sentiment'] = np.random.randint(2, size=len(review_df))
    review_df['feature'] = np.random.choice(features_list,size=len(review_df))

    review_df.to_csv('check.csv')

    Default_pros_cons = {}
    Default_pros_cons['Pros'] = "Default pros1"
    Default_pros_cons['Cons'] = "Default Cons1"

    pros_dict = {}
    cons_dict = {}

    for feature in features_list:
        pros_dict[feature] = f"{feature} pros"
        cons_dict[feature] = f"{feature} cons"

    # review_df.to_csv('check.csv')

    # Calculate 1-month rolling sum of positive sentiment (1) and total sentiment
    sentiment_1_month_sum = review_df['sentiment'].rolling(window='120D').sum()
    total_sentiment_1_month = review_df['sentiment'].rolling(window='120D').count()

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


    # # Convert Plotly figure to HTML content
    # plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False)

    # Save the plot as an HTML file
    plot_html = f"static/ratings_plot.html"
    fig.write_html(plot_html)

    # Generate word cloud
    wordcloud_image_path = generate_word_cloud(review_df)

    # Render the template with product details and plot HTML
    return render_template('index copy.html', products=product_details, plot_html=plot_html,features_list1=features_list,pros= pros_dict, cons= cons_dict,dft = Default_pros_cons , wordcloud_image_path=wordcloud_image_path)

if __name__ == '__main__':
    app.run(debug=True)

# Amazon Review Insights

This repository contains code for a Flask web application that scrapes Amazon product reviews, performs sentiment analysis, generates insights, and displays them in a user-friendly interface. The application is designed to help sellers on Amazon gain actionable insights from customer reviews to improve their products and services.

## Features

- **Amazon Review Scraping**: The application scrapes product details and reviews from Amazon using a provided product URL.
- **Sentiment Analysis**: Utilizes machine learning models to perform sentiment analysis on the scraped reviews.
- **Action Items Generation**: Generates action items for sellers based on the insights extracted from the reviews.
- **Pros and Cons Identification**: Identifies and presents the pros and cons of the product features based on customer reviews.
- **Word Cloud Visualization**: Displays a word cloud visualization of the most frequently used words in the reviews.
- **Rating Trends**: Shows the trend of average ratings over time along with monthly sentiment analysis.

## Technologies Used

- **Flask**: Python web framework used for developing the application.
- **MongoDB**: NoSQL database used for storing product details and reviews.
- **Redis**: Key-value store used for caching and storing temporary data.
- **Transformers**: Library for natural language processing (NLP) tasks, used for sentiment analysis.
- **Plotly**: Python graphing library used for creating interactive plots.
- **Bootstrap**: Front-end framework used for styling the web interface.
- **Docker**: Containerization technology used for packaging the application.
- **BeautifulSoup** : Scraping Packages used to scrape the reviews from Amazon
- **AWS**: Cloud infrastructure provider utilized for deployment.

## Deployment

The application is deployed on Amazon ECS (Elastic Container Service) using the provided deployment YAML file (`deploy-to-ecs.yml`). It utilizes Amazon ECR (Elastic Container Registry) for storing Docker images.

To deploy the application:

1. Configure AWS credentials.
2. Login to Amazon ECR.
3. Build and push the Docker image to Amazon ECR.
4. Deploy the Docker image to Amazon ECS

## Contributions
Jai Vigneshwar Aiyyappan
Vignesh Baskaran
Ali Guneysel
Madhoumithaa Veerasethu 

## Usage

1. Clone the repository to your local machine.
2. Install the necessary dependencies specified in `requirements.txt`.
3. Run the Flask application using `python final_with_dino.py`.
4. Access the application through your web browser.


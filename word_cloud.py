from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import inflect
import re
import os

# Download NLTK stopwords (run this line once)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Function to convert plural words to singular form
def plural_to_singular(word):
    p = inflect.engine()
    return p.singular_noun(word) or word

# Function to generate word cloud
def generate_word_cloud(reviews_df):
    # Join all reviews into a single string
    reviews = ' '.join(reviews_df['review'])

    # Tokenize the reviews
    tokens = word_tokenize(reviews)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Convert plural words to singular form
    filtered_tokens = [plural_to_singular(word) for word in filtered_tokens]

    # Remove '.', ',', n't, and pronunciations
    filtered_tokens = [re.sub(r'[.,!?’\'"\(\)]', '', word) for word in filtered_tokens if word.lower() != "n't"]

    # Join the filtered tokens back into a single string
    filtered_reviews = ' '.join(filtered_tokens)

    # Generate word frequencies
    word_freq = pd.Series(filtered_tokens).value_counts()

    # Get top 20 words
<<<<<<< HEAD
    top_words = word_freq.head(50)
=======
    top_words = word_freq.head(20)
>>>>>>> 900891a (fixed the error related to empty reviews)

    # Convert to dictionary for WordCloud
    wordcloud_data = top_words.to_dict()

    # Generate word cloud
<<<<<<< HEAD
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(wordcloud_data)
=======
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
>>>>>>> 900891a (fixed the error related to empty reviews)

    # Save the word cloud as an image
    image_path = f"static/wordcloud.png"
    wordcloud.to_file(image_path)
    print('image_path')
    return image_path



if __name__ =='__main__':
    df_review = pd.read_excel(".\\scrapping\\amazon_reviews.xlsx")
    generate_word_cloud(df_review)
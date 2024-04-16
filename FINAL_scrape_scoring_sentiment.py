import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
import pymongo
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pymongo import MongoClient

client = pymongo.MongoClient("mongodb+srv://Admin:Admin1234@cluster0.lhuhlns.mongodb.net")
db = client["Full_Stack_Project"]
collection = db["Amazon_Reviews"]

#This should be ran once the docker model initialized in order to save time before receiving the url.
def initialize_classifier(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("zero-shot-classification",
                          model=model,
                          tokenizer=tokenizer)
    return classifier



def get_scraping_link(url):
    match = re.search(r'(https://www.amazon.com/.+?/dp/[\w-]+)', url)
    if match:
        extracted_url = match.group(1)
        replaced_url = re.sub(r'/dp/', '/product-reviews/', extracted_url)
        return replaced_url
    else:
        return None

def get_soup(url):
    r = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
    #r=requests.get(product_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

def get_product_details(product_url, excel_file):
    soup = get_soup(product_url)

    # Title
    title_elem = soup.select_one('#productTitle')
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Overall Rating
    rating_elem = soup.select_one('#acrPopover')
    rating = rating_elem.get('title', '').replace(" out of 5 stars", "") if rating_elem else ""

    # Final price
    price_elem = soup.select_one('span.a-price span.a-offscreen')
    price = price_elem.get_text(strip=True) if price_elem else ""

    # Link to the product image
    image_elem = soup.select_one('#landingImage')
    image = image_elem.get('src', '') if image_elem else ""

    # Product description
    description_elem = soup.select_one('#feature-bullets')
    description = description_elem.get_text(strip=True) if description_elem else ""

    # Features
    features = []
    for element in soup.find_all('div', {'data-hook': 'cr-insights-widget-aspects'}):
       for span in element.find_all("span", {"class": "a-size-base"}):
         features.append(span.get_text(strip=True))

    product_details = {
        "Title": [title],
        "Rating": [rating],
        "Price": [price],
        "Image_URL": [image],  
        "Description": [description],
        "Features": [", ".join(features)]
    }
    
    df = pd.DataFrame([product_details])

    df.to_excel(excel_file, index=False)
    print('Product details saved to', excel_file)

    return product_details

def get_reviews(soup,  candidate_labels, classifier, sentiment_model, product_url=None, star_rating=None):
    reviewlist = []
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            title_text_elem = item.find('a', {'data-hook': 'review-title'})
            title_text = title_text_elem.text.strip() if title_text_elem else ""
            title_parts = title_text.split("stars")
            title = title_parts[1].strip() if len(title_parts) > 1 else ""
            
            rating_elem = item.find('i', {'data-hook': 'review-star-rating'})
            rating = float(rating_elem.text.replace('out of 5 stars', '').strip()) if rating_elem else ""
    
            
            date_text_elem = item.find('span', {'data-hook': 'review-date'})
            date_text = date_text_elem.text.strip().split("on")[-1].strip() if date_text_elem else ""
            date = datetime.strptime(date_text, "%B %d, %Y").strftime("%d %B %Y") if date_text else ""
    
            review_text_elem = item.find('span', {'data-hook': 'review-body'})
            review_text = review_text_elem.text.strip() if review_text_elem  else ""

            sentiment = ""
            if review_text != "": 
                # print(review_text) #### CHANGES
                # print(candidate_labels) #### CHANGES
                sentiment_result = sentiment_model(review_text)[0]['label']
                if sentiment_result == 'positive':
                    sentiment = 1
                elif sentiment_result == 'neutral':
                    sentiment = 1
                elif sentiment_result == 'negative':
                    sentiment = 0
                
                result = classifier(review_text,candidate_labels) #Getting scores for each feature
                labels = result['labels']
                scores = result['scores']
    
                num_feats = len(scores)
    
                a = 0
                for i in range(num_feats - 1):
                    if scores[i+1] / scores[i] >= 0.75:
                        a = i
                    else:
                        break
                labels = labels[:a+1]
                # scores = [str(x) for x in scores]
                review = {
                    'title': title,
                    'rating': rating,
                    'date': date,
                    'review': review_text,
                    'features': labels,
                    'sentiment' : sentiment
                }
                reviewlist.append(review)
    except Exception as e:
        print("Error occurred while parsing review:", e)
    return reviewlist

def scrape_amazon_reviews(product_url, star_ratings, candidate_labels, classifier, sentiment_model):
    reviewlist = []

    for rating in star_ratings:
        for x in range(1, 11): 
            full_url = f'{product_url}/ref=cm_cr_getr_d_paging_btm_next_{x}?ie=UTF8&reviewerType=all_reviews&filterByStar={rating}_star&pageNumber={x}&sortBy=recent'
            print('Current URL:', full_url)
            
            soup = get_soup(full_url)
            
            print(f'Getting page: {x} for {rating} star(s)')
            
            reviews = get_reviews(soup, candidate_labels, classifier, sentiment_model, product_url, rating)
            print('Number of reviews on this page:', len(reviews))
            
            reviewlist.extend(reviews)
            print('Total reviews collected so far:', len(reviewlist))
            
            if not soup.find('li', {'class': 'a-disabled a-last'}):
                pass
            else:
                break

    return reviewlist

def insert_product_info_to_mongodb(product_url, product_details, all_reviews):
    
    product_info = {
        "Product_Details": {
            "Product_URL": product_url,
            "Title": product_details["Title"][0],  
            "Overall_Rating": product_details["Rating"][0], 
            "Final_Price": product_details["Price"][0], 
            "Image_URL": product_details["Image_URL"][0],
            "Description": product_details["Description"][0],  
            "Features": product_details["Features"][0],
        },
        "Reviews": all_reviews,  
    }

    result = collection.insert_one(product_info)
    print("Document inserted successfully with ID:", result.inserted_id)
    print(product_details)

    return result.inserted_id



if __name__ == '__main__':
    
    start_time = time.time()

    #Initializing the classifier
    model_name = "Recognai/zeroshot_selectra_medium" #This zero-shot classifier is a lightweight classification model that will help us to speed up 
    classifier = initialize_classifier(model_name)

    #Initializing the sentiment model
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, max_length=512, truncation=True)
    # sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    
    product_url = input('Enter the Amazon URL: ')
    
    excel_file_product_details = 'product_details.xlsx'
    excel_file_reviews = 'amazon_reviews.xlsx'

    
    product_details = get_product_details(product_url, excel_file_product_details)
    while product_details["Features"] == [""]:
        product_details = get_product_details(product_url, excel_file_product_details)
        
    candidate_labels = product_details["Features"]

    
    modified_url = get_scraping_link(product_url)

    if modified_url:
        print('Modified URL:', modified_url)
    else:
        print('Invalid URL or pattern not found.')

    star_ratings = ['one', 'two', 'three', 'four', 'five']
    #star_ratings = ['two']
    
    candidate_labels = product_details["Features"]
    candidate_labels = candidate_labels[0].replace(" ", "").split(",")
    all_reviews = scrape_amazon_reviews(modified_url, star_ratings, candidate_labels, classifier, sentiment_model)
    
    df = pd.DataFrame(all_reviews)
    df.to_excel(excel_file_reviews, index=False)
    print('Excel is Ready!')
    
    insert_product_info_to_mongodb(product_url, product_details, all_reviews)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print('Execution time:', execution_time, 'seconds')

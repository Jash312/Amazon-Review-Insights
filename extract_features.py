import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time

#Extracting AI generated features
def feature_extractor(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    candidate_labels = []

    for element in soup.find_all('div', {'data-hook' :'cr-insights-widget-aspects'}):
        span_tags = element.find_all('span', {'class': 'a-size-base'})
        for span_tag in span_tags:
            candidate_labels.append(span_tag.text)
    return candidate_labels

#This should be ran once the docker model initialized in order to save time before receiving the url.
def initialize_classifier(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("zero-shot-classification",
                          model=model,
                          tokenizer=tokenizer)
    return classifier

#This function takes candidate labels and classifier, then returns scored dataframe with reviews.
def feature_scorer(candidate_labels, classifier, reviews_df):
    features_df = pd.DataFrame(columns = ['review'] + candidate_labels) #The table that we are going to fill with feature scores belong to the each review.
    #Looping through each review
    for i,review in enumerate(reviews_df['review']):
        result = classifier(review,candidate_labels) #Getting scores for each feature
        row_values = (dict(zip(result['labels'],result['scores']))) #Creating a dictionary to fill values into table
        row_values['review'] = result['sequence'] #Sequence is our review
        row_df = pd.DataFrame([row_values])
        features_df = pd.concat([features_df, row_df]) #Adding each result into the table
    return features_df

if __name__ == "__main__":
    url = input('Input the product url')
    start = time.time()
    model_name = "Recognai/zeroshot_selectra_medium" #This zero-shot classifier is a lightweight classification model that will help us to speed up the process.
    reviews_df = pd.read_excel('amazon_reviews.xlsx')
    classifier = initialize_classifier(model_name)
    candidate_labels = feature_extractor(url)
    features_df = feature_scorer(candidate_labels, classifier, reviews_df)

    features_df.to_excel('feature_scoring.xlsx')
    end = time.time()
    print('Scores successfully saved into feature_scoring.xlsx')
    print(f'Elaped time: {(end - start):.2f} seconds.')
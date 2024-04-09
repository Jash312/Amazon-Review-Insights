import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re

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
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

def get_product_details(product_url, excel_file):
    soup = get_soup(product_url)

    # Title
    title = soup.select_one("#productTitle").get_text(strip=True)

    # Overall Rating
    rating_elem = soup.select_one("#acrPopover")
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

    # Getting the features
    features = [span.get_text(strip=True) for element in soup.find_all('div', {'data-hook': 'cr-insights-widget-aspects'}) for span in element.find_all("span", {"class": "a-size-base"})]

    # Create DataFrame for product details
    product_details = {
        "Title": [title],
        "Rating": [rating],
        "Price": [price],
        "Image URL": [image],
        "Description": [description],
        "Features": [", ".join(features)]
    }
    df = pd.DataFrame(product_details)

    # Write DataFrame to Excel
    df.to_excel(excel_file, index=False)
    print("Product details saved to", excel_file)

def get_reviews(soup, product_url=None, star_rating=None):
    reviewlist = []
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            title_text = item.find('a', {'data-hook': 'review-title'}).text.strip() if item.find('a', {'data-hook': 'review-title'}) else ""
            title_parts = title_text.split("stars")
            title = title_parts[1].strip() if len(title_parts) > 1 else ""
            rating = float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()) if item.find('i', {'data-hook': 'review-star-rating'}) else ""
            date_text = item.find("span", {"data-hook": "review-date"}).text.strip().split("on")[-1].strip() if item.find("span", {"data-hook": "review-date"}) else ""
            date = datetime.strptime(date_text, "%B %d, %Y").strftime("%d %B %Y") if date_text else ""
            review_text = item.find('span', {'data-hook': 'review-body'}).text.strip() if item.find('span', {'data-hook': 'review-body'}) else ""
            review = {
                'title': title,
                'rating': rating,
                'date': date,
                'review': review_text
            }
            reviewlist.append(review)
    except Exception as e:
        print("Error occurred while parsing review:", e)
    return reviewlist

def scrape_amazon_reviews(product_url, star_ratings):
    reviewlist = []

    for rating in star_ratings:
        for x in range(1, 11):
            full_url = f'{product_url}/ref=cm_cr_getr_d_paging_btm_next_{x}?ie=UTF8&reviewerType=all_reviews&filterByStar={rating}_star&pageNumber={x}&sortBy=recent'
            print("Current URL:", full_url)
            soup = get_soup(full_url)
            print(f'Getting page: {x} for {rating} star(s)')
            reviews = get_reviews(soup, product_url, rating)
            print("Number of reviews on this page:", len(reviews))
            reviewlist.extend(reviews)
            print("Total reviews collected so far:", len(reviewlist))
            if not soup.find('li', {'class': 'a-disabled a-last'}):
                pass
            else:
                break

    return reviewlist


if __name__ == "__main__":
    start_time = time.time()

    product_url = input("Enter the Amazon URL: ")
    
    excel_file_product_details = "product_details.xlsx"
    excel_file_reviews = "amazon_reviews.xlsx"

    get_product_details(product_url, excel_file_product_details)

    modified_url = get_scraping_link(product_url)

    if modified_url:
        print("Modified URL:", modified_url)
    else:
        print("Invalid URL or pattern not found.")

    star_ratings = ['one', 'two', 'three', 'four', 'five']
    all_reviews = scrape_amazon_reviews(modified_url, star_ratings)

    df = pd.DataFrame(all_reviews)
    df.to_excel(excel_file_reviews, index=False)
    print('Excel is Ready!')
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

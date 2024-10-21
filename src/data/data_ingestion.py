import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import requests
from bs4 import BeautifulSoup
import os
import yaml
import logging

# coz ye tera path folder hai trere pc ka acer likha huya hjai 
# we dont give paths like this coz its troublesome to run on some other machines

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)sss - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def yaml_params(path: str) -> int:
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        pages = config['data_ingestion']['pages']
        page_size = config['data_ingestion']['page_size']
        return (pages, page_size)
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except KeyError as e:
        logger.error(f"Error: Missing key {e} in YAML configuration.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred while reading the YAML file: {e}")
        raise
logger.debug('extraced values of pages and page size succesfully from yaml file')

def scrape_data(base_url : str, pages : int, page_size: int ) -> pd.DataFrame:
    reviews = []
    for i in range(1, pages + 1):
        try:
            print(f"Scraping page {i}")
            url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
            response = requests.get(url)
            response.raise_for_status()  # Raises an error for bad status codes

            content = response.content
            parsed_content = BeautifulSoup(content, 'html.parser')

            for para in parsed_content.find_all("div", {"class": "text_content"}):
                reviews.append(para.get_text())

            print(f"   ---> {len(reviews)} total reviews")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error: Request failed for page {i}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error: An unexpected error occurred during data scraping on page {i}: {e}")
            continue

    df = pd.DataFrame()
    df["reviews"] = reviews
    return df 
logger.debug('df made successfully')

def split(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['reviews'] = df['reviews'].apply(lambda x: x.split('|')[1] if '|' in x else x)
    except Exception as e:
        logger.error(f"Error: An error occurred while splitting the reviews: {e}")
        raise
    return df 
logger.debug('df cleaned successfully')

def store(path : str, file_n : str, df: pd.DataFrame):
    try:
        os.makedirs(path, exist_ok=True)
        file_name = file_n
        full_path = os.path.join(path, file_name)
        df.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")
    except Exception as e:
        logger.error(f"Error: An error occurred while saving data to {path}: {e}")
        raise
logger.debug('data stored sucessfully')

def main():
    try:
        pages, page_size = yaml_params(path='params.yaml')
        
        base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
        df = scrape_data(base_url, pages, page_size)
        
        df = split(df)
        
        data_path = 'data/raw' # we just need inna sa
        file_n = 'raw_data.csv'
        store(data_path, file_n, df)
    
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()


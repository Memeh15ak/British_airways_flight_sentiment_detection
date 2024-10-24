import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
import json
import logging

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

def read_data(path:str, model_path:str, path2:str)->pd.DataFrame:
    try:
        tfidf_test = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Test data file not found at {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error (f"Error parsing the test data CSV file: {e}")
        raise
    except Exception as e:
        logger.error (f"An unexpected error occurred while reading test data: {e}")
        raise
    
    try:
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)
    except FileNotFoundError:
        logger.error (f"Model file not found at {model_path}")
        raise
    except pickle.UnpicklingError as e:
        logger.error (f"Error loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

    try:
        y_pred = rf_model.predict(tfidf_test)
    except Exception as e:
        logger.error (f"An error occurred while making predictions: {e}")
        raise

    try:
        y_test_1 = pd.read_csv(path2)
        y_test = y_test_1.values
    except FileNotFoundError:
        logger.error(f"y_test file not found at {path2}")
        raise
    except pd.errors.ParserError as e:
        logger.error (f"Error parsing the y_test CSV file: {e}")
        raise
    except Exception as e:
        logger.error (f"An unexpected error occurred while reading y_test data: {e}")
        raise

    return tfidf_test, rf_model, y_test, y_pred

logger.debug('file loaded sucessfully')

def metrics(y_test:pd.DataFrame, y_pred:pd.DataFrame)->dict:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        return metrics_dict
    except Exception as e:
        logger.error (f"An error occurred while calculating metrics: {e}")
        raise
logger.debug('metrics created')

def store(path:str, metrics_dict:dict):
    try:
        with open(path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        logger.error (f"An error occurred while storing metrics: {e}")
        raise
logger.debug('metrics created')

def main():
    try:
        path = 'data/interim/tfidf_test.csv'
        path2 = 'data/interim/y_test.csv'
        model_path = 'model.pkl'
        tfidf_test, rf_model, y_test, y_pred = read_data(path, model_path, path2)
        
        metrics_dict = metrics(y_test, y_pred)
        
        path3 = 'metrics.json'
        store(path3, metrics_dict)
    
    except Exception as e:
        logger.error (f"An unexpected error occurred in the main function: {e}")
        raise

if __name__ == "__main__":
    main()

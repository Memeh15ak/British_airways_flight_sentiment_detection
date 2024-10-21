from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import yaml
import os
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

def yaml_params(path:str)-> int:
    try:
        with open(path, 'r') as file:
            n_estimators = yaml.safe_load(file)['model_training']['n_estimators']
        return n_estimators
    except FileNotFoundError:
        logger.error(f"YAML file not found at {path}")
        raise
    except KeyError as e:
        logger.errorKeyError(f"Missing key in the YAML file: {e}")
    except yaml.YAMLError as e:
        logger.error (f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading YAML file: {e}")

def read_data(path:str, path2:str)->pd.DataFrame:
    try:
        tfidf_df_train = pd.read_csv(path)
        tfidf_df_test = pd.read_csv(path2)
        return tfidf_df_train, tfidf_df_test
    except FileNotFoundError:
        logger.error(f"File not found at {path} or {path2}")
        raise
    except pd.errors.ParserError as e:
        logger.error (f"Error while parsing the CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading data: {e}")
        raise
    
logger.debug('')

def train(n_estimators:int, train:pd.DataFrame)-> RandomForestClassifier:
    try:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf_model.fit(train.iloc[:, 0:-1], train['output'])
        return rf_model
    except ValueError as e:
        logger.error (f"Error in training the model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        raise
    
logger.debug('model trained sucessfully')


def store(path:str, rf_model:RandomForestClassifier):
    try:
        with open(path, 'wb') as file:
            pickle.dump(rf_model, file)
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model: {e}")
        raise
    
logger.debug("file stored sucessfully")

def main():
    try:
        n_estimators = yaml_params('params.yaml')
        
        train_path = 'data/interim/tfidf_train.csv'
        test_path = 'data/interim/tfidf_test.csv'
        tfidf_df_train, tfidf_df_test = read_data(train_path, test_path)
        
        rf_model = train(n_estimators, tfidf_df_train)
        
        model_path = 'model.pkl'
        store(model_path, rf_model)
    
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise
if __name__ == "__main__":
    main()

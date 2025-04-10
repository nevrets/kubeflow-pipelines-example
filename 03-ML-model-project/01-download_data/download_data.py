import json

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def download_data(config):
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    # Creates `data` structure to save 
    data = {'X_train' : X_train.tolist(),
            'y_train' : y_train.tolist(),
            'X_test' : X_test.tolist(),
            'y_test' : y_test.tolist()}
    
    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(config.data, 'w') as output_file:
        json.dump(data_json, output_file)
        
        
        
if __name__ == '__main__':
    
    # This component does not receive any input, it only outputs one artifact which is `data`.
    # Output argument: data
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str)
    
    config = p.parse_args()
    
    # Creating the directory where the OUTPUT file will be created, (the directory may or may not exist).
    # This will be used for other component's input (e.g. decision tree, logistic regression)
    Path(config.data).parent.mkdir(parents=True, exist_ok=True)
    
    download_data(config)
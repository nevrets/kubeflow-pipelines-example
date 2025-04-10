import json

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def svm(config):
    # Open and reads file "data"
    with open(config.data) as data_file:
        data = json.load(data_file)
    
    # Data type is 'dict', however since the file was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get the dict-type object.
    data = json.loads(data)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Initialize and train the model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save output into file
    with open(config.acc, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))
        
        

if __name__ == '__main__':
    
    # Defining and parsing the command-line arguments
    p = argparse.ArgumentParser(description='Program description')
    
    # Input argument: data
    # Output argument: accuracy
    p.add_argument('--data', type=str)
    p.add_argument('--acc', type=str)
    
    config = p.parse_args()
    
    # Creating the directory where the OUTPUT file will be created, (the directory may or may not exist).
    Path(config.acc).parent.mkdir(parents=True, exist_ok=True)
    
    svm(config)
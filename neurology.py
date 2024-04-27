import pandas as pd
import joblib

def load_model():
    # Load the model
    model = joblib.load('neurology_model.joblib')
    return model

def get_feature_names():
    # Load and preprocess the dataset to obtain feature names
    data = pd.read_csv('neurology.csv')
    data = data.replace({'yes': 1, 'no': 0})
    feature_names = data.drop('Disorder', axis=1).columns  # Assuming 'Disorder' is the target column
    return feature_names

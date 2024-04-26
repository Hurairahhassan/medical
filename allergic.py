import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Specify the path where the model is saved
model_path = 'allergies_model_nb.joblib'

# Load the model from the specified path
best_model= joblib.load(model_path)
print("Model loaded successfully.")

# Load dataset
allergies_data = pd.read_csv('Allergies_Dataset.csv')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'sym' column to obtain TF-IDF features
X = tfidf_vectorizer.fit_transform(allergies_data['sym'])

# Initialize the Label Encoder
label_encoder = LabelEncoder()

# Fit and transform the 'dis' column to obtain encoded labels
y = label_encoder.fit_transform(allergies_data['dis'])

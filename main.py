from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load
# Define a request model
class Symptoms(BaseModel):
    symptoms: str  # User will input symptoms as a single string separated by commas

app = FastAPI()

@app.get("/")
async def rootMsg():
    return "API IS RUNNING PERFECTLY"


# Load the logistic regression model
lr_loaded = load('logistic_regression_model.pkl')

# Load the dataset
df_norm = pd.read_csv("dis_sym_dataset_norm.csv")
X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]
dataset_symptoms = list(X.columns)  # List of symptoms from the dataset

@app.post("/predict/")
async def predict_disease(symptoms: Symptoms):
    user_input = symptoms.symptoms.lower().strip()
    user_symptoms = [sym.strip().replace('-', ' ').replace("'", '') for sym in user_input.split(',')]

    # Prepare sample input for the model
    sample_x = [0] * len(dataset_symptoms)
    for sym in user_symptoms:
        if sym in dataset_symptoms:
            sample_x[dataset_symptoms.index(sym)] = 1

    # Model prediction
    prediction = lr_loaded.predict_proba([sample_x])
    max_prob_index = prediction[0].argmax()  # Index of the highest probability
    highest_disease = Y['label_dis'].unique()[max_prob_index]
    highest_probability = prediction[0][max_prob_index] * 100  # Convert probability to percentage

    # Return the most probable disease and its probability
    return {
        "disease": highest_disease,
        "probability": f"{highest_probability:.2f}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

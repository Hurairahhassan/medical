from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from allergic import best_model, tfidf_vectorizer, label_encoder

# Define a request model with validation
class Symptoms(BaseModel):
    symptoms: str  # User will input symptoms as a single string separated by commas

    # Adding a validator to ensure the input is not empty
    @validator('symptoms')
    def check_symptoms(cls, value):
        if not value:
            raise ValueError('Symptoms input must not be empty')
        return value

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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

@app.post("/symptoms")
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


# -------------------------- Alergic -------------------------
# Function to predict allergy
def predict_allergy(symptom_input: str):
    symptom_tfidf = tfidf_vectorizer.transform([symptom_input])
    disease_pred = best_model.predict(symptom_tfidf)
    predicted_disease = label_encoder.inverse_transform(disease_pred)[0]
    return predicted_disease

# Route to predict allergy
@app.post("/allergic")
async def predict_allergy_route(symptoms: Symptoms):
    symptoms = [symptom.strip() for symptom in symptoms.symptoms.split(',')]
    all_predicted_diseases = [predict_allergy(symptom) for symptom in symptoms]
    # Find the most common predicted disease
    predicted_allergy = max(set(all_predicted_diseases), key=all_predicted_diseases.count)
    return {"predicted_allergy": predicted_allergy}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
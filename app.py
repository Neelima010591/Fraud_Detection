from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

app = FastAPI()

# Load model once (important)
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class Transaction(BaseModel):
  features: List[float]# all input features in order

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Transaction):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)

    return {
        "prediction": int(prediction[0]),
        "result": "Fraud" if prediction[0] == 1 else "Not Fraud"
    }
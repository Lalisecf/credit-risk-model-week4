import joblib
import pandas as pd

from fastapi import FastAPI

from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)

model = joblib.load(
    "models/best_model.pkl"
)

@app.get("/")
def home():
    return {
        "message": "Credit Risk Model API Running"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: CustomerData):

    df = pd.DataFrame(
        [data.dict()]
    )

    probability = model.predict_proba(df)[0][1]

    return PredictionResponse(
        risk_probability=float(probability)
    )
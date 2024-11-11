from fastapi import FastAPI
from api.utils import models

app = FastAPI()

# download all models
# uncomment this line to download models
# currently commented out to avoid downloading models every time the app starts
# models.download_models()


@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictions API"}


@app.get("/models")
def get_models():
    # Placeholder for getting model mapping
    return models.get_model_mapping()


@app.post("/models/{model_id}/predict")
def predict_price(model_id: str):
    # Placeholder for prediction logic
    return {"prediction": "This will be the house price prediction"}


@app.post("/models/predict_all")
def predict_all():
    # Placeholder for prediction logic
    return [{
            "model_id": "random_forest_model_v0_1",
            "prediction": "This will be the house price prediction"
            }]


@app.post("/models/{model_id}/predict_with_actuals")
def predict_with_actuals(model_id: str):
    # Placeholder for prediction logic
    return {"prediction": "This will be the house price prediction"}


@app.post("/predict")
def predict_price():
    # Placeholder for prediction logic
    return {"prediction": "This will be the house price prediction"}

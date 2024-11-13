from fastapi import FastAPI, HTTPException
from api.schemas.housing_data import HousingData
from api.utils import models
import pandas as pd
import os
import sys


app = FastAPI(title="House Price Predictions API", version="0.1",
              description="An API to make house price predictions")

# download all models
# uncomment this line to download models
# currently commented out to avoid downloading models every time the app starts
# models.download_models()


@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictions API"}


@app.get("/models")
def get_models():
    return models.get_model_mapping()


@app.post("/models/{model_id}/predict")
def predict_price(model_id: str, data: HousingData):
    model_path = models.get_model_path(model_id)
    if model_path:
        # load the model and make predictions
        print(f"Making predictions using model: {model_id}")
        # TODO loading the model everytime might be inefficient
        prediction = models.load_model_and_predict(
            model_path, data.model_dump())
        prediction["model_id"] = model_id
        return prediction
    else:
        # return 404 if model_id is not found
        raise HTTPException(status_code=404, detail=f"Model with id {
                            model_id} not found")


@app.post("/models/predict_all")
def predict_all(data: HousingData):
    # load model mapping
    model_mapping = models.load_model_mapping()
    response = []
    # load all models and make predictions
    for model in model_mapping:
        model_id = model["id"]
        model_path = model["path"]
        print(f"Making predictions using model: {model_id}")
        prediction = models.load_model_and_predict(
            model_path, data.model_dump())
        prediction["model_id"] = model_id
        response.append(prediction)
    return response


@app.post("/models/{model_id}/predict_with_actuals")
def predict_with_actuals(model_id: str):
    model_path = models.get_model_path(model_id)
    if model_path:
        # load dataset
        df = pd.read_csv("data/processed/housing/test_set.csv")
        # select random row from the dataset
        input = df.sample(1)
        # split the input into features and target
        features = input.drop("median_house_value", axis=1)
        labels = input["median_house_value"].copy()
        # load the model and make predictions
        print(f"Making predictions using model: {model_id}")
        # TODO loading the model everytime might be inefficient
        prediction = models.load_model_and_predict(model_path, features)
        prediction["model_id"] = model_id
        prediction["actual"] = labels.values[0]
        return prediction
    else:
        # return 404 if model_id is not found
        raise HTTPException(status_code=404, detail=f"Model with id {
                            model_id} not found")

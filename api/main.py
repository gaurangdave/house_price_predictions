from api.utils.transformers import (
    calculate_ratio,
    feature_ratio_transformer,
    MultimodalTransformer,
    ClusterSimilarityTransformer,
    heavy_tail_transformer
)
from fastapi import FastAPI, HTTPException
from api.schemas.housing_data import HousingData
from api.utils import models
import pandas as pd
import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import preprocessing utilities


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
    # Placeholder for getting model mapping
    return models.get_model_mapping()


@app.post("/models/{model_id}/predict")
def predict_price(model_id: str, data: HousingData):
    # Placeholder for prediction logic
    model_path = models.get_model_path(model_id)
    if model_path:
        # load the model and make predictions
        print(f"Making predictions using model: {model_id}")
        # TODO loading the model everytime might be inefficient
        model = models.load_model(model_path)
        df = pd.DataFrame(data.model_dump(), index=[0])
        prediction = model.predict(df)
        return {"prediction": prediction[0].round(2)}
    else:
        # return 404 if model_id is not found
        raise HTTPException(status_code=404, detail=f"Model with id {
                            model_id} not found")


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

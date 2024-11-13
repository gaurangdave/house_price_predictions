import gdown
import json
from pathlib import Path
import joblib
import pandas as pd


def download_models():
    # Placeholder for downloading models
    print("Downloading models...")
    try:
        url = 'https://drive.google.com/drive/folders/1_HihZZk7T5_InmIxBiKHxLoZVjr8YYXO?usp=drive_link'
        output = str(Path('api', 'models'))
    # download the models
        gdown.download_folder(url=url, output=output, quiet=False)
        print("Models downloaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Models download failed.")
        raise e


def load_model_mapping():
    # read data/models.json file
    with open(Path('api', 'data', 'models.json'), 'r') as f:
        model_mapping = f.read()

    # convert the json string to a array
    model_mapping = json.loads(model_mapping)
    return model_mapping


def load_model(model_path):
    # load joblib model
    model = joblib.load(model_path, mmap_mode='r')
    return model


def load_model_and_predict(model_path, data):
    # load the model and make predictions
    model = load_model(model_path)
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return {"prediction": prediction[0].round(2)}


def get_model_path(model_id):
    model_mapping = load_model_mapping()

    # filter out the model path for the given model_id
    model_path = [model["path"]
                  for model in model_mapping if model["id"] == model_id]
    return model_path[0] if model_path else None


def get_model_mapping():
    # Placeholder for getting model mapping
    model_mapping = load_model_mapping()

    # filter out path attribute from model_mapping array
    model_mapping = [{"id": model["id"], "name": model["name"]}
                     for model in model_mapping]
    return model_mapping

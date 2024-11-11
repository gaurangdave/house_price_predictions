import gdown
import json
from pathlib import Path


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


def load_models():
    # read data/models.json file
    with open(Path('api', 'data', 'models.json'), 'r') as f:
        model_mapping = f.read()

    # convert the json string to a array
    model_mapping = json.loads(model_mapping)
    return model_mapping


def get_model_mapping():
    # Placeholder for getting model mapping
    model_mapping = load_models()

    # filter out path attribute from model_mapping array
    model_mapping = [{"id": model["id"], "name": model["name"]}
                     for model in model_mapping]
    return model_mapping

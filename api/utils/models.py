import gdown
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


def get_model_mapping():
    # Placeholder for getting model mapping
    model_path = str(Path('api', 'models'))
    return [{
            "id": "random_forest_model_v0_1",
            "name": "Random Forest Regressor",
            "path": f"{model_path}/random_forest_model_v0_1.joblib"
            }]

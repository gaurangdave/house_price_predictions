import argparse
from sklearn.pipeline import Pipeline
from api.utils.transformers import (
    calculate_ratio,
    feature_ratio_transformer,
    MultimodalTransformer,
    ClusterSimilarityTransformer,
    heavy_tail_transformer,
    preprocessing_pipeline
)
import gdown
import json
from pathlib import Path
import joblib
import sys
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from joblib import dump


def download_processed_data():
    processed_data_path = Path("data", "processed", "housing")
    print(f"Downloading the dataset from {processed_data_path}")
    # read data
    data = pd.read_csv(Path(processed_data_path, "train_set.csv"))
    # before we create the pipeline lets split he training data into features and labels
    df_features = data.drop("median_house_value", axis=1)
    df_labels = data["median_house_value"].copy()
    return df_features, df_labels


def write_best_params(best_params, version, model):
    # TODO think about how to approach this
    # create csv of best parameters and rmse
    # best_params = rnd_search.best_params_
    # best_params["rmse"] = rnd_search.best_score_
    # best_params["version"] = version
    # best_params["model_name"] = "random_forest"
    # best_params_path = Path("..", "..", "data", "best_params.json")
    # with open(best_params_path, "w") as f:
    #     json.dump(best_params, f)
    pass


def save_model(model, model_name):
    # save the model
    print("Saving the model...")
    model_path = Path("models", model_name)
    dump(model, model_path)
    return model_path


def train_random_forest(version="v1"):
    # Placeholder for the training logic
    print("Training the model using Random Forest...")
    df_features, df_labels = download_processed_data()
    print("Training Random Forest Model for production API")
    random_forest_pipeline = Pipeline(
        [("preprocessing", preprocessing_pipeline), ("randomforestregressor", RandomForestRegressor())])
    param_distribs = {
        "randomforestregressor__n_estimators": randint(low=1, high=200),
        "randomforestregressor__max_features": randint(low=1, high=8),
        "preprocessing__cluster_similarity__calculate_cluster__n_clusters": randint(low=10, high=100),
    }

    rnd_search = RandomizedSearchCV(random_forest_pipeline, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42)
    rnd_search.fit(df_features, df_labels)

    # print the best parameters
    print("Best parameters found: ")
    print(rnd_search.best_params_)
    write_best_params(rnd_search.best_params_, version, "random_forest")
    model_name = f"random_forest_model_{version}.joblib"
    model_path = save_model(rnd_search.best_estimator_, model_name)
    print(f"Model saved at {model_path}")


def train_model():
    # Placeholder for the training logic
    print("Training the model...")

    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Train a machine learning model.")
        parser.add_argument('--model_name', type=str, default='random_forest',
                            help='Name of the model to be trained (default: random_forest)')
        parser.add_argument('--version', type=str, required=True,
                            help='Version of the model to be trained')
        return parser.parse_args()

    args = parse_arguments()
    print(f"Model name provided: {args.model_name}")
    print(f"Model version provided: {args.version}")

    if args.model_name == 'random_forest':
        train_random_forest()
    else:
        print("Model not found")


if __name__ == "__main__":
    train_model()

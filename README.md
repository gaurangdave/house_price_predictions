# House Price Prediction
This project is a learning endeavor aimed at building a machine learning model to predict housing prices in California using census data. Guided by examples and techniques from the book `Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow`, this project explores essential steps in the ML pipeline—from data preprocessing and feature engineering to model training and evaluation.

The primary objective is to replace the current manual and costly estimation process, which is prone to errors, with an automated and accurate model. By developing a regression model, this project demonstrates how ML can streamline and enhance real-world tasks, providing insights and automation that are both scalable and cost-effective.

## Project Goal

- Use California census data to build a model of housing prices in the state.
- This data includes metrics such as the population, median income, and median housing price for each block group in California.
- Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).
- Currently district house prices are estimated manually by experts who gather the up-to-date information about to either get the median housing price or estimate it using complex rules.
  - The current process is costly and time-consuming and estimates are off by more than 30%
- We need a ML model to **predict the median housing price in any district, given all the other metrics.**
- The model’s output will be fed to another ML system along with other signals to determine whether it is worth investing in a given area.


## Solution Details

- The solution needs to predict the housing price so this is going to be a regression task and since we have labeled dataset, we can use supervised learning algorithms.
  - Specifically its a `multiple regression problem` since we’ll use multiple features to make a prediction.
  - And since target is a single value its going to be `univariate regression` problem.
- We don’t expect the housing prices or the related matrices to change that often, so a `batch learning` approach should suffice.
  - The data is small enough to fit in memory so plain batch learning should be fine.

### Performance Measure

- We are going to use the `Root mean square error (RMSE)` to measure the performance.
- The mathematical formula for `RMSE` is

$$
RMSE(X,h) = \sqrt{{1 \over m} \sum_{i=1}^m(h(x^{(i)}) - y^{(i)})^2}
$$

- Here,
  - $m$ is the number of instances in the dataset we are measuring `RMSE` on.
  - $x^{(i)}$ is the vector of all the feature values of the $i^{th}$ instance in the dataset and $y^{(i)}$ is its label.
  - $X$ is the matrix containing all the feature values (excluding the labels) of all the instances in the dataset.
  - $h$ is the system’s prediction function, also called `hypothesis`.
    - When the prediction function receives feature vector $x^{(i)}$ as input it outputs the predicted value $\hat{y}^{(i)}$

$$
    \hat{y}^{(i)} = h(x^{(i)})
$$

### Data Transformation
We need the following data transformations (in same order)
* Fill in missing values
* Convert `ocean_proximity` to one hot encoding
* Feature Engineering `rooms_per_house`, `bedroom_ratio` and `people_per_house`
* Add cluster similarity features
* Drop Outliers
* Transform heavy tailed features using logarithm
* Scale all numeric features. 

### Dataset

* The data is downloaded from [Remote Data Repo](https://github.com/ageron/data/raw/main/housing.tgz)
* The data is stored in repo [Local Copy](https://github.com/gaurangdave/house_price_predictions/tree/main/data)


### Notebooks
* [00_get_data.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/00_get_data.ipynb) to download the dataset and create local copy.
* [01_explore_data.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/01_explore_data.ipynb) to create train/test set and data exploration.
* [02_transform_data.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/02_transform_data.ipynb) for data transformation.
* [03_training_evaluation.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/03_training_evaluation.ipynb) training and evaluation.
* [04_download_models.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/04_download_models.ipynb) to download all the existing trained models.

### Models
* All the trained models are stored here in [Google Drive](https://drive.google.com/drive/folders/1_HihZZk7T5_InmIxBiKHxLoZVjr8YYXO)
* The models can be downloaded either manually thru Google Drive web interface, or by running [04_download_models.ipynb](https://github.com/gaurangdave/house_price_predictions/blob/main/notebooks/04_download_models.ipynb) notebook. 


## Tech Stack

![Environment](https://img.shields.io/badge/Environment-Linux_64-FCC624?logo=linux&style=for-the-badge)
![Conda](https://img.shields.io/badge/Conda-24.9.1-342B029?logo=Anaconda&style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-2.14.0-F37626?logo=Jupyter&logoColor=F37626&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12.2-FFD43B?logo=Python&logoColor=blue&style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-2C2D72?logo=Pandas&logoColor=2C2D72&style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.24.1-239120?logo=Plotly&logoColor=239120&style=for-the-badge)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-1.5.1-F7931E?logo=scikit-learn&logoColor=F7931E&style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-109989?logo=Fastapi&logoColor=109989&style=for-the-badge)

## Installation

- Create conda environment with `Python 3.12`

```bash
  conda create -n ml python=3.12
```

- Activate the environment

```bash
  conda activate ml
```

- Install ML Libraries

```bash
conda install numpy pandas scikit-learn matplotlib seaborn plotly jupyter ipykernel -y
```

```bash
conda install -c conda-forge python-kaleido
```

- Install GDown
```bash
conda install conda-forge::gdown
```

- Install DotEnv
```bash
conda install conda-forge::python-dotenv
```

- Install FastAPI

```bash
conda install -c conda-forge fastapi uvicorn -y
```
## Running the API
* Run the following command to start the API server

```bash
uvicorn api.main:app --reload
```

* Go to the following URL to access API Docs
```URL
http://localhost:8000/docs
```

## API Reference

| Action                                           | HTTP Method | Endpoint                                 |
|--------------------------------------------------|-------------|------------------------------------------|
| List available models                            | **`GET`**       | `/models`                                |
| Get predictions using a certain model            | **`POST`**      | `/models/{model_id}/predict`             |
| Get predictions from all models                  | **`POST`**      | `/models/predict_all`                    |
| Get predictions with actual values for accuracy  | **`POST`**      | `/models/{model_id}/predict_with_actuals`|

## Visualizations
![Actual Values vs Predictions](https://github.com/gaurangdave/house_price_predictions/blob/8f1dbec1293f2403db4c9cc221b332662a082970/reports/figures/final_predictions_vs_actual_values.png?raw=true "A visualization showing performance of ML model on test data")


## Project Insights
* The final model achieved a **Relative RMSE of 19.85%**, which is a significant improvement over the current manual process, where estimates deviate by more than **30%**.
* This represents an approximate **33% improvement** in prediction accuracy compared to the manual approach.
* The scatter plot of predictions vs. actual values shows an overall **linear relationship**, indicating that the model is reasonably accurate in predicting housing prices.
For **low to mid-range prices (under 300K)**, predictions align closely with actual values.However, as prices increase, there is a tendency for predictions to fall below the ideal line, suggesting that the model struggles with higher price ranges.
* The concentration of points near the **500K mark** reflects the upper cap in the dataset, which likely limits the model’s ability to predict higher values accurately.

### Next Steps
* Explore **feature engineering** to add new features that may correlate with higher housing prices, potentially improving performance for higher price ranges.
* Experiment with **more complex models** (e.g., gradient boosting or neural networks) to capture nonlinear relationships that the current model might be missing.

## Lessons Learnt
* Gained experience in **identifying data distributions** and applying appropriate preprocessing techniques for machine learning training.
* Learned about **cluster similarity** and methods for measuring similarity between different feature types.
* Developed skills in creating **preprocessing pipelines** using scikit-learn’s Pipeline and custom transformers.
* Built knowledge on how to **deploy trained models as APIs** using FastAPI, allowing for seamless integration of predictions into applications.

## 🚀 About Me

A jack of all trades in software engineering, with 15 years of crafting full-stack solutions, scalable architectures, and pixel-perfect designs. Now expanding my horizons into AI/ML, blending experience with curiosity to build the future of tech—one model at a time.

## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://gaurangdave.me/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaurangvdave/)

## 🛠 Skills

`Python`, `Jupyter Notebook`, `scikit-learn`, `FastAPI`, `Plotly`, `Conda`

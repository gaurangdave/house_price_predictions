# House Price Prediction

## Problem Statement

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

## Data

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

- Install FastAPI

```bash
conda install -c conda-forge fastapi uvicorn -y
```

## API Reference

TBD

## Usage/Examples

TBD

## 🚀 About Me

I'm a full stack developer...

## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://gaurangdave.me/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gaurangvdave/)

## 🛠 Skills

Python, Jupyter Notebook,

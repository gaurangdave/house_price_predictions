from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np


def calculate_ratio_feature_names(function_transformer, feature_names_in):
    """Calculate the ratio feature names."""
    # feature_name = f"{feature_names_in[0]}_to_{feature_names_in[1]}_ratio"
    return ["ratio"]  # feature names out


def calculate_ratio(df):
    """Calculate the ratio of the first two columns."""
    result = df[:, [0]] / df[:, [1]]
    return result


# import this in models.py
feature_ratio_transformer = FunctionTransformer(
    calculate_ratio, feature_names_out=calculate_ratio_feature_names)


class MultimodalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, gamma=0.1):
        self.n_clusters = n_clusters
        self.gamma = gamma

    def fit(self, X, y=None):
        checked_X = check_array(X)
        self.X_ = checked_X
        self.feature_data_ = self.X_[:, 0]
        self.kde_ = gaussian_kde(self.feature_data_)
        self.x_grid_ = np.linspace(
            self.feature_data_.min(), self.feature_data_.max(), 1000)
        self.kde_values_ = self.kde_.evaluate(self.x_grid_)
        self.peaks_, _ = find_peaks(self.kde_values_)
        self.peaks_values_ = self.x_grid_[self.peaks_]
        return self

    def transform(self, X):
        check_is_fitted(self, ['X_', 'feature_data_',
                        'kde_', 'x_grid_', 'kde_values_', 'peaks_'])
        checked_X = check_array(X)
        feature_data = checked_X[:, 0].reshape(-1, 1)
        similarity_matrix = np.zeros((feature_data.shape[0], len(self.peaks_)))
        for i, peak in enumerate(self.peaks_):
            peak_value = self.x_grid_[peak]
            similarity_matrix[:, i] = rbf_kernel(
                feature_data, [[peak_value]], gamma=self.gamma).flatten()
        return similarity_matrix

    def get_feature_names_out(self, input_features=None):
        return [f"similarity_to_peak_{round(peak)}" for peak in self.peaks_values_]


class ClusterSimilarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self, ['kmeans_'])
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"similarity_to_cluster_{i}" for i in range(self.n_clusters)]


def heavy_tail_distribution(df):
    """Transform heavy tailed distribution using logarithm."""
    return np.log1p(df)


heavy_tail_transformer = FunctionTransformer(
    heavy_tail_distribution, inverse_func=np.expm1, feature_names_out="one-to-one")


def standard_scaler(df):
    """Scale features using StandardScaler."""
    return StandardScaler().fit_transform(df)


standard_scaler_transformer = FunctionTransformer(standard_scaler)

cat_pipeline = Pipeline([
    ("impute_categories", SimpleImputer(strategy="most_frequent")),
    ("encode_categories", OneHotEncoder(sparse_output=False))
])


ratio_pipeline = Pipeline([
    ("impute_ratios", SimpleImputer(strategy="median")),
    ("calculate_ratios", feature_ratio_transformer),
    ("standard_scaler", StandardScaler())
])

multimodal_similarity_pipeline = Pipeline([
    ("impute_multimodal", SimpleImputer(strategy="median")),
    ("calculate_multimodal", MultimodalTransformer(n_clusters=5, gamma=0.1)),
    ("standard_scaler", StandardScaler())
])

# most frequent imputer for latitude and longitude
cluster_similarity_pipeline = Pipeline([
    ("impute_cluster", SimpleImputer(strategy="most_frequent")),
    ("calculate_cluster", ClusterSimilarityTransformer(
        n_clusters=10, gamma=1, random_state=42)),
    ("standard_scaler", StandardScaler())
])

log_pipeline = Pipeline([
    ("impute_log", SimpleImputer(strategy="median")),
    ("log_transform", heavy_tail_transformer),
    ("standard_scaler", StandardScaler())
])

# copy/pasting the pipeline from above for easy access
cat_pipeline = Pipeline([
    ("impute_categories", SimpleImputer(strategy="most_frequent")),
    ("encode_categories", OneHotEncoder(sparse_output=False))
])

# lets create a full pipeline
preprocessing_pipeline = ColumnTransformer([
    ("bedrooms_per_room", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
    ("rooms_per_household", ratio_pipeline, ["total_rooms", "households"]),
    ("population_per_household", ratio_pipeline, ["population", "households"]),
    ("multimodal_similarity",
     multimodal_similarity_pipeline, ["housing_median_age"]),
    ("cluster_similarity", cluster_similarity_pipeline,
     ["latitude", "longitude"]),
    ("log_pipeline", log_pipeline, [
     "total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("categorical", cat_pipeline, ["ocean_proximity"])
])

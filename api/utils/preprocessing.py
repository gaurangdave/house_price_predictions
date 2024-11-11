def calculate_ratio_feature_names(function_transformer, feature_names_in):
    """Calculate the ratio feature names."""
    return ["ratio"]  # feature names out


def calculate_ratio(df):
    """Calculate the ratio of the first two columns."""
    result = df[:, [0]] / df[:, [1]]
    return result

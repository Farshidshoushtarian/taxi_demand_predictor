import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

import lightgbm as lgb
def average_rides_last_4_weeks(X: pd.DataFrame) -> np.ndarray:
    """
    Calculates the average number of rides for the past 4 weeks

    Args:
        X (pd.DataFrame): features

    Returns:
        np.ndarray: average number of rides for the past 4 weeks
    """
    # Check if 'average_rides_last_4_weeks' column exists
    if 'average_rides_last_4_weeks' not in X.columns:
        raise KeyError("Missing required column: 'average_rides_last_4_weeks'")

    # Extract the average number of rides for the past 4 weeks
    avg_rides = X['average_rides_last_4_weeks'].values.reshape(-1, 1)

    return avg_rides


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `pickup_hour` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_hour'].dt.hour
        X_["day_of_week"] = X_['pickup_hour'].dt.dayofweek
        
        return X_.drop(columns=['pickup_hour'])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )


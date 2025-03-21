import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

import hopsworks
import hsfs
from hsfs.feature_view import FeatureView
from hsfs.training_dataset import TrainingDataset

from config_types import FeatureViewConfig
import src.config as config  # Changed to absolute import
from feature_store_api import get_or_create_feature_view
from logger import get_logger

import os
import sys
from pathlib import Path
from joblib import load  # Import the load function

# Import the transform_raw_data_into_ts_data function
from src.data import transform_raw_data_into_ts_data

# Determine the project root directory
project_root = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()
src_path = project_root / "src"

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

logger = get_logger()

TRAINING_DATASET_VERSION = 1

# Correct the FEATURE_VIEW_PREDICTIONS_METADATA
FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',  # Correct name
    version=1,  # Correct version
    feature_group=None # not needed for now
)

def connect_to_feature_store() -> Tuple[hopsworks.project.Project, hsfs.feature_store.FeatureStore]:
    """
    Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        Tuple[hopsworks.project.Project, hsfs.feature_store.FeatureStore]: project, feature_store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    feature_store = project.get_feature_store()
    return project, feature_store

def get_training_data(
    feature_view: FeatureView,
    start_date: datetime,
    end_date: datetime,
    training_dataset_version: int = TRAINING_DATASET_VERSION
) -> pd.DataFrame:
    """
    Gets the training data from the feature store

    Args:
        feature_view (FeatureView): feature view object
        start_date (datetime): start date
        end_date (datetime): end date
        training_dataset_version (int): training dataset version

    Returns:
        pd.DataFrame: training data
    """
    logger.info(f"start_date:{start_date}")
    logger.info(f"end_date:{end_date}")

    training_data: TrainingDataset = feature_view.get_training_data(
        start_time=start_date,
        end_time=end_date,
        training_dataset_version=training_dataset_version
    )
    training_df = training_data[0]
    return training_df

def load_predictions_from_store(
    from_pickup_hour: datetime,
    to_pickup_hour: datetime
    ) -> pd.DataFrame:
    """
    Loads the predictions from the feature store

    Args:
        from_pickup_hour (datetime): from pickup hour
        to_pickup_hour (datetime): to pickup hour

    Returns:
        pd.DataFrame: predictions
    """
    predictions_fv = get_or_create_feature_view(FEATURE_VIEW_PREDICTIONS_METADATA)

    # read the feature view as a dataframe
    predictions_df = get_training_data(
        feature_view=predictions_fv,
        start_date=from_pickup_hour,
        end_date=to_pickup_hour
    )

    # Debug print to check the contents of predictions_df after get_training_data
    print("Predictions DataFrame after get_training_data:")
    print(predictions_df.head())
    print("Column Names:", predictions_df.columns)

    if predictions_df.empty:
        return pd.DataFrame()

    # Check if 'predicted_demand_feature' exists, if not, try 'prediction'
    if 'predicted_demand_feature' in predictions_df.columns:
        predictions_df = predictions_df.rename(
            columns={'predicted_demand_feature': 'predicted_demand'}
        )
    elif 'prediction' in predictions_df.columns:
        predictions_df = predictions_df.rename(
            columns={'prediction': 'predicted_demand'}
        )
    else:
        print("Error: Neither 'predicted_demand_feature' nor 'prediction' column found!")
        return pd.DataFrame()

    # Ensure 'pickup_hour' is in the columns before returning
    if 'pickup_hour' not in predictions_df.columns:
        print("Error: 'pickup_hour' column not found!")
        return pd.DataFrame()

    # only return `pickup_location_id` and `predicted_demand`
    return predictions_df[['pickup_location_id', 'predicted_demand', 'pickup_hour']]

def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Loads a batch of features from the feature store

    Args:
        current_date (datetime): current date

    Returns:
        pd.DataFrame: features
    """
    # one feature view contains all the features
    features_fv = get_or_create_feature_view(config.FEATURE_VIEW_FEATURES_METADATA) #LOAD FEATURES FEATURE VIEW

    # we want the last 25 hours of training data
    start_date = current_date - timedelta(hours=25)
    end_date = current_date

    # read the feature view as a dataframe
    features_df = get_training_data(
        feature_view=features_fv,
        start_date=start_date,
        end_date=end_date
    )

    if features_df.empty:
        return pd.DataFrame()

    features_df.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)
    
    return features_df

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the model predictions

    Args:
        model: model
        features (pd.DataFrame): features

    Returns:
        pd.DataFrame: predictions
    """

    # Transform raw data into time series data
    features = transform_raw_data_into_ts_data(features)

    predictions = model.predict(features)
    results = pd.DataFrame({'predicted_demand': predictions.flatten()})  # Flatten predictions
    return results

def load_model_from_registry():
    """
    Loads the model from the model registry

    Returns:
        model: model
    """
    # connect to the project
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

    mr = project.get_model_registry()
    # the model is the latest model in the Model Registry
    model = mr.get_model(
        name = config.MODEL_NAME,
        version = config.MODEL_VERSION
    )
    model_dir = model.download()
    # Modify the model path to point to the correct location
    model_path = Path(model_dir) / "model.pkl"  # Corrected path - REMOVE "models"
    print(f"Model path: {model_path}")
    model = load(model_path)

    print("Model loaded")
    return model
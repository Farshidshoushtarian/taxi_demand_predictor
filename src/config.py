import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
import sys

from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig
from src.paths import PARENT_DIR

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now imports should work
from src.paths import PARENT_DIR

# Explicitly load .env from the project root
project_root = Path(__file__).parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

# Get environment variables
HOPSWORKS_PROJECT_NAME = "Taxi_demand_predictor"  # Changed to match project name
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Raise an error if they are missing
if not HOPSWORKS_PROJECT_NAME or not HOPSWORKS_API_KEY:
    raise Exception(
        f"Missing environment variables! Ensure .env file exists at {dotenv_path} "
        "and contains HOPSWORKS_PROJECT_NAME and HOPSWORKS_API_KEY."
    )

DATA_DIR = PARENT_DIR / "data"  # Added DATA_DIR variable

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 4
FEATURE_VIEW_NAME = "time_series_hourly_feature_view"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

@dataclass
class FeatureGroupConfig:
    name: str
    version: int
    description: str
    primary_key: list
    event_time: str
    online_enabled: bool

@dataclass
class FeatureViewConfig:
    name: str
    version: int
    feature_group_name: str
    feature_group_version: int
    query: str

FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=1,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,
)

FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view',
    version=1,
    feature_group_name=FEATURE_GROUP_METADATA.name,
    feature_group_version=FEATURE_GROUP_METADATA.version,
    query="SELECT * FROM {}".format(FEATURE_GROUP_METADATA.name)
)

# added for monitoring purposes
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description='Predictions generate by our production model',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,  # Added online_enabled argument
)

FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group_name=FEATURE_GROUP_PREDICTIONS_METADATA.name,
    feature_group_version=FEATURE_GROUP_PREDICTIONS_METADATA.version,
    query="SELECT * FROM {}".format(FEATURE_GROUP_PREDICTIONS_METADATA.name)
)

MONITORING_FV_NAME = 'monitoring_feature_view'
MONITORING_FV_VERSION = 1

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# number of iterations we want Optuna to pefrom to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 1

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 30.0

INPUT_SEQ_LEN = 168  # Define input sequence length
STEP_SIZE = 1  # Define step size
import pandas as pd
from datetime import datetime, timedelta, UTC
import hopsworks
import hsfs
import os

# Load configurations
import sys
from pathlib import Path

# Determine the project root directory using GITHUB_WORKSPACE
project_root = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd())).resolve()

src_path = project_root / "src"

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Now you can do absolute imports
from inference import load_model_from_registry, get_model_predictions
from data import load_raw_data, transform_raw_data_into_ts_data
from config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
#from feature_store_api import create_feature_group_if_not_exists, get_feature_store #Not needed here

# Connect to Hopsworks
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)
fs = project.get_feature_store()

# Load the model
model = load_model_from_registry()

# Define the feature group
feature_group_name = FEATURE_GROUP_NAME
feature_group_version = FEATURE_GROUP_VERSION  # Increment the feature group version
feature_group_description = "Feature group for time series hourly data with predicted demand"
primary_key = ["pickup_location_id", "pickup_ts"]  # Adjust primary key as needed
event_time = "pickup_ts"  # Adjust event time column as needed
online_enabled = True

# Use timezone-aware UTC timestamp
current_date = pd.to_datetime(datetime.now(UTC)).floor('H')
print(f'{current_date=}')

# Fetch raw data for the last 28 days
fetch_data_to = current_date
fetch_data_from = current_date - timedelta(days=28)


def fetch_batch_raw_data(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 52 weeks ago (i.e. 1 year)
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)

    # Ensure from_date_ and to_date_ are timezone-aware
    from_date_ = from_date_.tz_convert("UTC")
    to_date_ = to_date_.tz_convert("UTC")

    print(f'{from_date=}, {to_date_=}')

    # Download 2 files from website
    rides = load_raw_data(year=from_date_.year, months=from_date_.month)
    rides["pickup_datetime"] = pd.to_datetime(rides["pickup_datetime"], utc=True)  # Ensure timezone awareness
    rides = rides[rides.pickup_datetime >= from_date_]

    rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
    rides_2["pickup_datetime"] = pd.to_datetime(rides_2["pickup_datetime"], utc=True)  # Ensure timezone awareness
    rides_2 = rides_2[rides_2.pickup_datetime < to_date_]

    rides = pd.concat([rides, rides_2])

    # Shift the data to pretend this is recent data
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides


rides = fetch_batch_raw_data(from_date=fetch_data_from, to_date=fetch_data_to)

ts_data = transform_raw_data_into_ts_data(rides)

# string to datetime
ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)

# add column with Unix epoch milliseconds
ts_data['pickup_ts'] = ts_data['pickup_hour'].astype(int) // 10**6

# Make sure that the order of the location ids is the same as the features
ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

# Generate predictions
features = ts_data[['pickup_hour', 'rides', 'pickup_location_id', 'pickup_ts']]
predictions_df = get_model_predictions(model, features)

# Add predictions to the time-series DataFrame
ts_data['predicted_demand'] = predictions_df['predicted_demand']

# connect to the project
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

# connect to the feature store
feature_store = project.get_feature_store()

# connect to the feature group
feature_group = feature_store.get_or_create_feature_group(
    name=feature_group_name,
    version=feature_group_version,
    description=feature_group_description,
    primary_key=["pickup_location_id", "pickup_ts"],
    event_time='pickup_ts',
    online_enabled=online_enabled
)

# insert the data into the feature group
feature_group.insert(ts_data, write_options={"wait_for_job": True})

print("Feature pipeline completed successfully!")
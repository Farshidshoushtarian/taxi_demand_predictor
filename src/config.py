import os
from dotenv import load_dotenv

from paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')
#print("Loaded HOPSWORKS_API_KEY:", os.getenv("HOPSWORKS_API_KEY"))

HOPSWORKS_PROJECT_NAME = 'Taxi_demand_predictor'
try:
    HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
except:
    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 3
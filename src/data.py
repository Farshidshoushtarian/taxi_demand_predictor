import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
import sys
from io import BytesIO
from typing import Optional, List, Tuple
from pdb import set_trace as stop

import numpy as np
from tqdm import tqdm

# Determine the project root directory
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"

# Add src to Python path
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.config import DATA_DIR
from src.logger import get_logger

logger = get_logger()

def download_file(year: int, month: int) -> Path:
    """
    Downloads the data for a given year and month from the NYC Taxi website

    Args:
        year (int): year
        month (int): month

    Returns:
        Path: the path to the downloaded file
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    local_file = DATA_DIR / f'yellow_tripdata_{year}-{month:02d}.parquet'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    logger.info(f'Downloading file {URL}')
    try:
        response = requests.get(URL)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        with open(local_file, 'wb') as f:
            f.write(response.content)

        logger.info(f'Downloaded file {local_file}')
        return local_file
    except requests.exceptions.RequestException as e:
        logger.error(f'Error downloading {URL}: {e}')
        return None

def load_raw_data(year: int, months: list[int]) -> pd.DataFrame:
    """
    Loads the raw data for a given year and month

    Args:
        year (int): year
        month (int): month

    Returns:
        pd.DataFrame: the raw data
    """
    # create an empty DataFrame to store the data
    rides = pd.DataFrame()

    for month in months:
        # download the file
        local_file = download_file(year, month)

        if local_file is not None:
            try:
                # read the data from the file
                rides_one_month = pd.read_parquet(local_file)

                # append the data to the DataFrame
                rides = pd.concat([rides, rides_one_month])
            except Exception as e:
                logger.error(f'Error reading {local_file}: {e}')
        else:
            logger.warning(f'File not found for year={year}, month={month}')

    return rides

def validate_raw_data(
    rides: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Removes rows with pickup_datetimes outside their valid range
    """
    # keep only rides for this month
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]
    
    return rides


def fetch_ride_events_from_data_warehouse(
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """
    This function is used to simulate production data by sampling historical data
    from 52 weeks ago (i.e. 1 year)
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # download 1 file of data only
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides = rides[rides.pickup_datetime < to_date_]

    else:
        # download 2 files from website
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
        rides_2 = rides_2[rides_2.pickup_datetime < to_date_]
        rides = pd.concat([rides, rides_2])

    # shift the pickup_datetime back 1 year ahead, to simulate production data
    # using its 7*52-days-ago value
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides


def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add necessary rows to the input 'ts_data' to make sure the output
    has a complete list of
    - pickup_hours
    - pickup_location_ids
    """
    location_ids = range(1, ts_data['pickup_location_id'].max() + 1)

    full_range = pd.date_range(ts_data['pickup_hour'].min(),
                               ts_data['pickup_hour'].max(),
                               freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        # keep only rides for this 'location_id'
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']]
        
        if ts_data_i.empty:
            # add a dummy entry with a 0
            ts_data_i = pd.DataFrame.from_dict([
                {'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}
            ])

        # quick way to add missing dates with 0 in a Series
        # taken from https://stackoverflow.com/a/19324591
        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)
        
        # add back `location_id` columns
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])
    
    # move the pickup_hour from the index to a dataframe column
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})
    
    return output


def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms the raw data into time series data

    Args:
        rides (pd.DataFrame): the raw data

    Returns:
        pd.DataFrame: the time series data
    """
    # group by pickup location ID and pickup hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    ts_data = rides.groupby(['pickup_location_id', 'pickup_hour'])[['ride_id']].count()
    ts_data = ts_data.reset_index()
    ts_data.rename(columns={'ride_id': 'rides'}, inplace=True)

    return ts_data


def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time-series format into a (features, target)
    format that we can use to train Supervised ML models
    """
    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        
        # keep only ts data for this `location_id`
        ts_data_one_location = ts_data.loc[
            ts_data.pickup_location_id == location_id, 
            ['pickup_hour', 'rides']
        ].sort_values(by=['pickup_hour'])

        # pre-compute cutoff indices to split dataframe rows
        indices = get_cutoff_indices_features_and_target(
            ts_data_one_location,
            input_seq_len,
            step_size
        )

        # slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        # numpy -> pandas
        features_one_location = pd.DataFrame(
            x,
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # numpy -> pandas
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # concatenate results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def get_cutoff_indices_features_and_target(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
    ) -> list:

        stop_position = len(data) - 1
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_mid_idx = input_seq_len
        subseq_last_idx = input_seq_len + 1
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_mid_idx += step_size
            subseq_last_idx += step_size

        return indices
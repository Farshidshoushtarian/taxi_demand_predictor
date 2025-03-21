from typing import Optional, List
from dataclasses import dataclass

import hsfs
import hopsworks

from config_types import FeatureGroupConfig, FeatureViewConfig
from logger import get_logger

logger = get_logger()

def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    import config as config
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_feature_store()

def create_feature_group_if_not_exists():
    feature_store = get_feature_store()
    try:
        feature_group = feature_store.get_feature_group(
            name="time_series_hourly_feature_group",
            version=1  # Use the correct version
        )
        print("Feature group already exists.")
    except hsfs.client.exceptions.RestAPIError as e:
        if e.error_code == 270009:
            print("Feature group does not exist. Creating feature group.")
            feature_group = feature_store.create_feature_group(
                name="time_series_hourly_feature_group",
                version=1,
                description="Feature group for time series hourly data",
                primary_key=["id"],  # Adjust primary key as needed
                event_time="event_time",  # Adjust event time column as needed
                online_enabled=True
            )
            print("Feature group created.")
        else:
            raise e

def get_or_create_feature_group(
    feature_group_metadata: FeatureGroupConfig
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_or_create_feature_group(
        name=feature_group_metadata.name,
        version=feature_group_metadata.version,
        description=feature_group_metadata.description,
        primary_key=feature_group_metadata.primary_key,
        event_time=feature_group_metadata.event_time,
        online_enabled=feature_group_metadata.online_enabled
    )

def get_or_create_feature_view(
    feature_view_metadata: FeatureViewConfig
) -> hsfs.feature_view.FeatureView:
    """"""

    # get pointer to the feature store
    feature_store = get_feature_store()

    # get pointer to the feature group
    feature_group = feature_store.get_feature_group(
        name="time_series_hourly_feature_group",  # Use the correct feature group name
        version=1  # Use the correct version
    )

    # create feature view if it doesn't exist
    try:
        feature_store.create_feature_view(
            name=feature_view_metadata.name,
            version=feature_view_metadata.version,
            query=feature_group.select_all()
        )
    except:
        logger.info("Feature view already exists, skipping creation.")
    
    # get feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=feature_view_metadata.name,
        version=feature_view_metadata.version,
    )

    return feature_view
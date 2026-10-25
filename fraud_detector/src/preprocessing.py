import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_train_data():
    logger.info('Loading training data...')

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some features
    train = add_features(train)
    logger.info('Train data processed. Shape: %s', train.shape)

    return train


def haversine(lat1, lon1, lat2, lon2):
    rad = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return rad * c

def manhattan_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def add_features(df):
    df_ = df.copy()
    df_['haversine'] = haversine(df_['lat'], df_['lon'],
                                 df_['merchant_lat'], df_['merchant_lon'])
    df_["manhattan_distance"] = manhattan_distance(df_['lat'], df_['lon'],
                                                   df_['merchant_lat'], df_['merchant_lon'])
    df_["manhattan_log"] = np.log1p(np.abs(df_["manhattan_distance"]))

    df_['haversine_log'] = np.log1p(np.abs(haversine(df_['lat'], df_['lon'],
                                     df_['merchant_lat'], df_['merchant_lon'])))
    df_["amount_log"] = np.log1p(np.abs(df_["amount"]))
    df_["amount_sqrt"] = np.sqrt(np.abs(df_["amount"]))
    df_["amount_normalized_by_dist"] = df_["amount"] / (df_["haversine_log"] + 1e-5)
    df_["transaction_time"] = pd.to_datetime(df_["transaction_time"])
    df_["population_city_log"] = np.log1p(np.abs(df_["population_city"]))
    df_["year"] = df_["transaction_time"].dt.year
    df_["month"] = df_["transaction_time"].dt.month
    df_["day"] = df_["transaction_time"].dt.day
    df_["hour"] = df_["transaction_time"].dt.hour
    df_["minute"] = df_["transaction_time"].dt.minute
    df_["weekend"] = (df_["transaction_time"].dt.dayofweek > 4).astype(int)
    return df_.drop(columns=["transaction_time", "name_1", "name_2", "street"])


# Main preprocessing function
def run_preproc(input_df):
    # Add features
    output_df = add_features(input_df)
    logger.info('Added new features. Output shape: %s', output_df.shape)

    # Return resulting dataset
    return output_df

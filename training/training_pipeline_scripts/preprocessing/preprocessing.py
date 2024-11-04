"""Script for preprocessing data for model training"""

import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ROOT = "/opt/ml/processing"
INPUT = os.path.join(ROOT, "input")
OUTPUT = os.path.join(ROOT, "output")


def get_data(file_list: list):
    """Read data from csv in file list and concat the input to a single df"""
    dfs = [pd.read_csv(file) for file in file_list]
    return pd.concat(dfs, ignore_index=True)


def create_features(data_df: pd.DataFrame):
    """Clean and preprocess data"""
    logger.info("Starting Preprocessing...")

    # Label encoding categorical columns
    label_encoders = {}
    for column in data_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data_df[column] = le.fit_transform(data_df[column])
        label_encoders[column] = le
    logger.info("Label encoding completed.")

    # Converting RestingBP and Cholesterol to numeric and handling outliers
    for col in ["RestingBP", "Cholesterol"]:
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")
    data_df["RestingBP"] = data_df["RestingBP"].apply(
        lambda x: np.nan if x > 140 or x < 40 else x
    )
    data_df["Cholesterol"] = data_df["Cholesterol"].replace(0, np.nan)

    # Replace NaN values with median
    data_df["RestingBP"].fillna(data_df["RestingBP"].median(), inplace=True)
    data_df["Cholesterol"].fillna(data_df["Cholesterol"].median(), inplace=True)

    logger.info("Preprocessing completed. Data shape: %s", data_df.shape)
    return data_df


def train_test_split(df: pd.DataFrame, val_size: float, test_size: float):
    """Split and save preprocessed features and target variable"""
    # Splitting features and target
    # x = data_df.drop("HeartDisease", axis=1)
    # y = data_df["HeartDisease"]

    train_size = 1 - (val_size + test_size)
    df_train, df_val, df_test = np.split(
        df, [int(train_size * len(df)), int((train_size + val_size) * len(df))]
    )
    logger.info(f"Training set size: {df_train.shape}")
    logger.info(f"Validation set size: {df_val.shape}")
    logger.info(f"Test set size: {df_test.shape}")
    return df_train, df_val, df_test


def save_training_files(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, baseline: pd.DataFrame
):
    os.makedirs(f"{OUTPUT}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT}/validation", exist_ok=True)
    os.makedirs(f"{OUTPUT}/test", exist_ok=True)
    os.makedirs(f"{OUTPUT}/baseline", exist_ok=True)
    """Save to csv (train-val-test) with same files for data and lables"""
    pd.DataFrame(train).to_csv(f"{OUTPUT}/train/train.csv", index=False)
    pd.DataFrame(val).to_csv(f"{OUTPUT}/validation/validation.csv", index=False)
    pd.DataFrame(test).to_csv(f"{OUTPUT}/test/test.csv", index=False)
    pd.DataFrame(baseline).to_csv(
        f"{OUTPUT}/baseline/baseline.csv", index=False, header=False
    )
    logger.info(f"Saved training files.")


def calculate_baseline(data: pd.DataFrame):
    numeric_data = data.select_dtypes(include=[np.number]).drop("HeartDisease", axis=1)
    baseline = numeric_data.mean().values.reshape(1, -1)
    return baseline


def preprocess_training(data_df: pd.DataFrame, val_size: float, test_size: float):
    logger.info(f"Preprocessing data for TRAINING")
    df = create_features(data_df)
    df = df.dropna().drop(["date"], axis=1, errors="ignore")
    target_col = df.pop("HeartDisease")
    df.insert(0, target_col.name, target_col)
    logger.info("Data shape after dropping rows with NaNs: %s", df.shape)

    # Split the data into train, val, and test and save it
    df_train, df_val, df_test = train_test_split(df, val_size, test_size)
    baseline = calculate_baseline(df_train)
    save_training_files(df_train, df_val, df_test, baseline)
    return


def parse_args() -> None:
    """Argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_size", default=0.2, type=float)
    parser.add_argument("--test_size", default=0.1, type=float)
    args, _ = parser.parse_known_args()
    logger.info("Arguments: %s", args)
    return args


if __name__ == "__main__":
    logger.info("Preprocessing Job - STARTING")
    args = parse_args()

    # List files in the input data directory
    input_file_list = glob.glob(f"{INPUT}/*.csv")
    logger.info("Input file list: %s", input_file_list)

    if len(input_file_list) == 0:
        raise Exception(f"No input files found in {INPUT}")

    # Load input data
    data_df = get_data(input_file_list)
    logger.info("Input file shape: %s", data_df.shape)

    preprocess_training(data_df, args.val_size, args.test_size)
    logger.info("Preprocessing Job - ENDED")

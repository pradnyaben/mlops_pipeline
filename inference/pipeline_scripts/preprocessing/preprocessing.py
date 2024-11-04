"""
Heart failure prediction - preprocessing
"""

# Importing packages
import os
import glob
import logging
import argparse
import pandas as pd
import numpy as np
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


def preprocess_data(data_df: pd.DataFrame):
    """Clean and preprocess data"""
    logger.info("Starting Preprocessing...")

    # Label encoding categorical columns
    label_encoders = {}
    for column in data_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data_df[column] = le.fit_transform(data_df[column])
        label_encoders[column] = le

    # Converting RestingBP and Cholesterol to numeric and handling outliers
    data_df["RestingBP"] = pd.to_numeric(data_df["RestingBP"], errors="coerce")
    data_df["Cholesterol"] = pd.to_numeric(data_df["Cholesterol"], errors="coerce")
    data_df["RestingBP"] = data_df["RestingBP"].apply(
        lambda x: np.nan if x > 140 else x
    )
    data_df["RestingBP"] = data_df["RestingBP"].apply(lambda x: np.nan if x < 40 else x)
    data_df["Cholesterol"] = data_df["Cholesterol"].replace(0, np.nan)

    # Replace NaN values with median
    data_df["RestingBP"] = data_df["RestingBP"].replace(
        np.nan, data_df["RestingBP"].median()
    )
    data_df["Cholesterol"] = data_df["Cholesterol"].replace(
        np.nan, data_df["Cholesterol"].median()
    )

    logger.info("Preprocessing completed. Data shape: %s", data_df.shape)
    return data_df


def save_inference_files(data_df: pd.DataFrame):
    """Save preprocessed features and target variable"""
    # Splitting features and target
    x = data_df.drop("HeartDisease", axis=1)
    y = data_df["HeartDisease"]

    # Save to csv
    x.to_csv(f"{OUTPUT}/features.csv", index=False)
    y.to_csv(f"{OUTPUT}/predict_variable.csv", index=False)
    logger.info("Preprocessed data saved")


def parse_args() -> None:
    """Argument Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT)
    parser.add_argument("--output-dir", default=OUTPUT)
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


if __name__ == "__main__":
    logger.info("Preprocessing Job - Starting")
    args = parse_args()

    # List files in the input data directory
    input_file_list = glob.glob(f"{args.input_dir}/*.csv")
    logger.info("Input file list: %s", input_file_list)

    if len(input_file_list) == 0:
        raise Exception(f"No input files found in {args.input_dir}")

    # Load input data
    data_df = get_data(input_file_list)
    logger.info("Input file shape: %s", data_df.shape)

    # Preprocess data
    preprocessed_data = preprocess_data(data_df)

    # Save the preprocssed data
    save_inference_files(preprocessed_data)

    logger.info("Preprocessing Job - Completed")

# -*- coding: utf-8 -*-
"""
Heart failure prediction - training the data
"""
# Importing packages
import os
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def classification_report_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Generate classification report metrics
    """
    report = classification_report(y_true, y_pred)
    return report


def get_data(train_dir: str, val_dir: str):
    """Read training and validation data from csv"""
    df_train = pd.read_csv(os.path.join(train_dir, "train.csv"))
    X_train = df_train.iloc[:, 1:]
    y_train = df_train.iloc[:, :1]
    logger.info(f"X_train shape {X_train.shape}")
    logger.info(f"y_train shape {y_train.shape}")

    df_val = pd.read_csv(os.path.join(val_dir, "validation.csv"))
    X_val = df_val.iloc[:, 1:]
    y_val = df_val.iloc[:, :1]
    logger.info(f"X_val shape {X_val.shape}")
    logger.info(f"y_val shape {y_val.shape}")

    return X_train, X_val, y_train, y_val


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    maxIter: int,
    solver: str,
):
    """Train Logistic Regression Model"""
    clf = LogisticRegression(max_iter=maxIter, solver=solver)
    clf.fit(X_train, y_train.values.ravel())
    return clf


def save_model(model, model_dir: str):
    """Save the trained model"""
    filename = os.path.join(model_dir, "model.joblib")
    logger.info(f"Saving model to {filename}")
    joblib.dump(model, filename)


def parse_args():
    """Argument Parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--maxIter", type=int, default=320)
    parser.add_argument("--solver", type=str, default="liblinear")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


if __name__ == "__main__":
    logger.info("Starting training job")
    args = parse_args()

    X_train, X_val, y_train, y_val = get_data(args.train, args.validation)
    model = train_model(X_train, y_train, args.maxIter, args.solver)
    prediction = model.predict(X_val)

    report = classification_report_metrics(np.array(y_val).ravel(), prediction)
    logger.info(f"Classification report: {report}")

    save_model(model, args.model_dir)

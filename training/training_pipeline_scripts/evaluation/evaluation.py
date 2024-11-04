# -*- coding: utf-8 -*-
"""
Heart failure prediction - testing the model
"""
# Importing packages
import os
import json
import logging
import tarfile
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ROOT = "/opt/ml/processing"
INPUT = os.path.join(ROOT, "input")
OUTPUT = os.path.join(ROOT, "output")
MODEL = os.path.join(ROOT, "model")


def safe_extract(tar, path="."):
    """Safely extract members from a tar file"""
    safe_members = []
    for member in tar.getmembers():
        # Validate member name to avoid path traversal
        member_path = os.path.join(path, member.name)

        # Check if the member's name is safe (i.e., does not contain '..')
        if os.path.commonpath([path, os.path.abspath(member_path)]) == os.path.abspath(
            path
        ) and not member.name.startswith("../"):
            safe_members.append(member)  # Add to safe members list
        else:
            logger.warning("Unsafe member detected - %s. Skipping.", member.name)

    # Check if there are any safe members before extraction
    if safe_members:
        logger.info(
            "Extracting safe members: %s", [member.name for member in safe_members]
        )

        # Extract each safe member individually to avoid bandit's concern
        for member in safe_members:
            tar.extract(member, path)  # Extract safe member
    else:
        logger.warning("No safe members to extract.")


def load_model():
    """Load model from path"""
    logger.info("Unzip the model")
    model_path = os.path.join(MODEL, "model.tar.gz")

    with tarfile.open(model_path) as tar:
        safe_extract(tar, path=".")

    logger.info("Load the model")
    return joblib.load("model.joblib")


def get_data():
    """Load data from path"""
    logger.info("Reading input data")
    test_path = os.path.join(INPUT, "test/test.csv")
    df_test = pd.read_csv(test_path)

    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    logger.info("Making predictions and evaluating the model")
    y_pred = model.predict(X_test)

    # Classification report
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification Report: \n%s", classification_report(y_test, y_pred))

    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy Score: {accuracy:.4f}")

    return clf_report, accuracy


def save_outputs(report: str, accuracy: float):
    """Save evaluation outputs"""
    logger.info("Saving evaluation outputs")
    output_path = os.path.join(OUTPUT, "evaluation_report.json")
    report["accuracy"] = accuracy
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        # f.write(json.dumps(report))
        # f.write(f"Accuracy Score: {accuracy:.4f}\n")


if __name__ == "__main__":
    logger.info("Starting model evaluation")

    # Extract and load the model
    model = load_model()

    # Load test data
    X_test, y_test = get_data()

    logger.info("Predicting on test data")
    predictions = model.predict(X_test)

    # Evaluate the model
    report, accuracy = evaluate_model(model, X_test, y_test)

    # save evaluation results
    save_outputs(report, accuracy)

    logger.info("Model evaluation completed.")

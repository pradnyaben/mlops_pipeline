import os
import joblib
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def input_fn(input_data, content_type):
    if content_type == "text/csv":
        logger.info("Deserializing input CSV")
        df = pd.read_csv(StringIO(input_data), index_col=False, header=None)
        logger.info("Input data share: %s", df.shape)
        logger.info("First row: %s", df.head(1))
        return df
    else:
        raise ValueError(f"{content_type} not supported by script!")


def predict_fn(input_data, model):
    print("Predicting")
    output_data = model.predict(input_data)
    return output_data


def model_fn(model_dir):
    print("Loading model")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

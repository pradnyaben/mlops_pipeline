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
        logger.info(f"Input Data: {input_data}")
        df = pd.read_csv(StringIO(input_data), index_col=False, header=None)
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"First row: {df.head(1)}")
        return df
    else:
        raise ValueError("{} not supported by Script!".format(content_type))


def predict_fn(input_data, model):
    print("Predicting")
    output_data = model.predict(input_data)
    return output_data


def model_fn(model_dir):
    print("Loading model")
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

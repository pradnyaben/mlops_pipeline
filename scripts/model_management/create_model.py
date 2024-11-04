# This script is for model management in the sagemaker
import json
import os
import sys
import boto3
import logging
from datetime import datetime, timezone

DEFAULT_DEPLOYMENT_REGION = "us-west-2"

# Connect to sagemaker
iam = boto3.client("iam")
s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
sm_client = boto3.client("sagemaker", region_name=DEFAULT_DEPLOYMENT_REGION)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def describe_model_package(model_package_arn):
    """Fetches the latest Model from Model Registry, which will be used for Inference"""

    response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        image = response["InferenceSpecification"]["Containers"][0]["Image"]
        model_data_url = response["InferenceSpecification"]["Containers"][0][
            "ModelDataUrl"
        ]
        logger.info(f"Inference image: {image}")
        logger.info(f"Model data url: {model_data_url}")
        return {"image": image, "model_data_url": model_data_url}

    return ""


def get_data_quality_baseline_uri(model_package_arn):
    """Fetches the data quality baseline results from training pipeline of the model,
    which will be used for data quality monitoring during inference
    """

    repsonse = sm_client.describe_model_package(ModelPackageName=model_package_arn)

    training_pipeline = sm_client.describe_pipeline_definition_for_execution(
        PipelineExecutionArn=repsonse["MetadataProperties"]["GeneratedBy"]
    )

    pipeline_steps = json.loads(training_pipeline["PipelineDefinition"])["Steps"]
    data_quality_step = [
        step for step in pipeline_steps if step["Type"] == "QualityCheck"
    ]

    if data_quality_step:
        baseline_results_uri = data_quality_step[0]["Arguments"][
            "ProcessingOutputConfig"
        ]["Outputs"][0]["S3Output"]["S3Uri"]
        logger.info(f"Data quality baseline results url: {baseline_results_uri}")
        return baseline_results_uri

    return ""


def create_model(config, create=False):
    "This function creates a model in Sagemaker"
    if create:
        # Generate model name from the time stamp
        now = datetime.now().replace(tzinfo=timezone.utc)
        timestamp = now.strftime("%Y%m%d%H%M%S")
        config["model_name"] = f"{config['model_group_name']}-{timestamp}"

    # Create model from the data we have
    logger.info(config)
    model_data_url = f"s3://{config['s3_local_bucket']}/{config['model_file']}"
    s3_code_path = f"s3://{config['s3_local_bucket']}/{config['code_file']}"
    create_model = sm_client.create_model(
        ModelName=config["model_name"],
        PrimaryContainer={
            "Image": config["model_name"],
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": s3_code_path,
            },
        },
        ExecutionRoleArn=config["execution_role_arn"],
        EnableNetworkIsolation=False,
    )
    if create_model["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return create_model["ModelArn"]
    return ""


def get_model(model, model_package_group_name):
    "This function finds the model in the sagemaker inference"
    # Create model from the data we have
    logger.info(model)
    models_list = sm_client.list_models(
        MaxResults=100,
        NameContains=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    for test_model in models_list["Models"]:
        # Collect additional info about the model
        test_model_info = sm_client.describe_model(ModelName=test_model["ModelName"])
        test_model_template = {
            "image": test_model_info["PrimaryContainer"]["Image"],
            "model_data_url": test_model_info["PrimaryContainer"]["ModelDataUrl"],
        }
        if model == test_model_template:
            logger.info(f"We found target model {test_model['ModelName']}")
            return test_model["ModelName"]
        # If no model found, return empty string
        return ""


def s3_get_config(model_package_group_name, s3_remote_bucket):
    "This function return config from the remote bucket"
    response = s3_client.get_object(
        Bucket=s3_remote_bucket, Key=f"{model_package_group_name}/config.json"
    )
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        logger.info(f"Get config file from the {s3_remote_bucket}")
        return json.loads(response["Body"].read())
    logger.error(f"Can't get config file from the {s3_remote_bucket}")
    return ""


def s3_copy_data(config):
    "This function copy the model from dev account"
    bucket = s3_resource.Bucket(config["s3_local_bucket"])
    keys = [config["code_file"], config["model_file"]]
    if "data_quality_baseline_uri" in config:
        keys.append("{}/statistics.json".format(config["data_quality_baseline_uri"]))
        keys.append("{}/constraint.json".format(config["data_quality_baseline_uri"]))

    for key in keys:
        copy_source = {"Bucket": config["s3_remote_bucket"], "Key": key}
        bucket.copy(copy_source, key)
        try:
            s3_resource.meta.client.copy(copy_source, config["s3_local_bucket"], key)
        except:
            logger.error("Something went wrong during file copy!")
            return False
        return True


def s3_put_config(config):
    "This function store results in the S3 after DEV execution"
    # Save result into S3 bucket
    response = s3_client.put_object(
        Body=json.dumps(config),
        Bucket=config["s3_local_bucket"],
        Key=f"{config['model_group_name']}/config.json",
    )
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        logger.info(f"Store config file to the {config['s3_local_bucket']}")
        return True
    return ""


def check_model_exists(config):
    """Function checks whether a model already exists in the account"""
    response = sm_client.list_models(NameContains=config["model_name"])
    if response["Models"]:
        logger.info(f"Model {config['model_name']} exists in the account")
    return response["Models"]


def main():
    "Execute core logic and create a model"
    # The TARGET_ENV is comming from CI/CD, default is dev, could be preprod and prod
    TARGET_ENV = os.getenv("TARGET_ENV", "dev")
    logger.info(f"Target environment is {TARGET_ENV}")

    # Dynamic global env. Can change from different runtimes
    REMOTE_BUCKET = os.getenv("REMOTE_BUCKET", "")
    ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET", "")
    MODEL_PACKAGE_GROUP_NAME = "model-group"
    SAGEMAKER_MODEL_ROLE_ARN = os.getenv("SAGEMAKER_MODEL_ROLE_ARN", "")

    # Return empty string by default
    result = ""

    # Main logic for DEV. If we are in the dev.
    # 1. Get the model package arn from the env vars
    # 2. Describe the package to get the image url & model data url
    # 3. Create a new model using these params
    if TARGET_ENV == "dev":
        S3_CODE_PATH = os.getenv("S3_CODE_PATH", "")
        MODEL_PACKAGE_ARN = os.getenv("MODEL_PACKAGE_ARN", "")

        # Get parameters required for creating the model
        model_package_metadata = describe_model_package(MODEL_PACKAGE_ARN)
        if not model_package_metadata:
            logger.error(f"Can't get additional package info. Stop execution.")
            return result

        # Model parameters
        config = {}
        try:
            # Store configuration info
            config["image"] = model_package_metadata["image"]
            config["model_file"] = "/".join(
                model_package_metadata["model_data_url"].split("/")[3:]
            )
            config["code_file"] = "/".join(S3_CODE_PATH.split("/")[3:])
            config["execution_role_arn"] = SAGEMAKER_MODEL_ROLE_ARN
            config["model_group_name"] = MODEL_PACKAGE_GROUP_NAME
            config["model_package_arn"] = MODEL_PACKAGE_ARN
            config["s3_local_bucket"] = ARTIFACT_BUCKET

            # If training pipeline had data quality check, get S3 uri of the baseline files
            data_quality_baseline_uri = get_data_quality_baseline_uri(MODEL_PACKAGE_ARN)
            if data_quality_baseline_uri:
                config["data_quality_baseline_uri"] = "/".join(
                    data_quality_baseline_uri.split("/")[3:]
                )
        except:
            logger.error("Error in configuration creation")
            return result

        # Create a model in DEV
        model_arn = create_model(config, True)
        config["model_arn"] = model_arn
        result = json.dumps(config)

        # Store config in S3
        if model_arn:
            s3_put_config(config)

    # 1. Read data from the config
    # 2. Check if model already exists in the account
    # 3. Copy model from the remote S3 to local S3
    # 4. Create the model locally
    if TARGET_ENV in ["preprod", "prod"]:
        config = s3_get_config(MODEL_PACKAGE_GROUP_NAME, REMOTE_BUCKET)
        if not config:
            logger.error(
                f"Can't find config file at the {REMOTE_BUCKET}. Stop execution."
            )
            return result
        # Check if model exists in the account
        model_exists = check_model_exists(config)

        if model_exists:
            logger.info(model_exists)
            result = json.dumps(
                s3_get_config(MODEL_PACKAGE_GROUP_NAME, ARTIFACT_BUCKET)
            )
        else:
            config["execution_role_arn"] = SAGEMAKER_MODEL_ROLE_ARN
            config["s3_local_bucket"] = ARTIFACT_BUCKET
            config["s3_remote_bucket"] = REMOTE_BUCKET
            if not s3_copy_data(config):
                logger.error(
                    f"Can't copy files from the {REMOTE_BUCKET}. Stop execution."
                )
                return result

            # Create a clone model from the config file
            model_arn = create_model(config, False)
            config["model_arn"] = model_arn
            result = json.dumps(config)

            # Store config in S3
            if model_arn:
                s3_put_config(config)

    # Return resutls
    return result


if __name__ == "__main__":
    model_arn = main()
    # Print result to get standart output or return error code
    if model_arn:
        print(model_arn)
    else:
        sys.stderr.write(
            "There is problem in scripts/model_management/create_model.py\n\n"
        )
        sys.exit(1)

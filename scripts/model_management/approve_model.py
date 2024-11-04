import os
import sys
import boto3
import logging

DEFAULT_DEPLOYMENT_REGION = "us-west-2"


s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")
sm_client = boto3.client("sagemaker", region_name=DEFAULT_DEPLOYMENT_REGION)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def approve_package(model_package_arn):
    "Approve package by ARN"
    response = sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
    )
    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        return response["ModelPackageArn"]
    return ""


def get_package(model_package_group_name, approved):
    "Return the last package from the group name"
    if approved:
        model_approval_status = "Approved"
    else:
        model_approval_status = "PendignManualApproval"

    # Request Sagemaker for the target model
    response = sm_client.list_model_packages(
        MaxResults=1,
        ModelApprovalStatus=model_approval_status,
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    # Return ARN or empty string
    if response["ModelPackageSummaryList"]:
        model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        logger.info(f"Latest model package arn: {model_package_arn}")
        return model_package_arn
    return ""


def main():
    MODEL_PACKAGE_GROUP_NAME = "model-group"
    approved_model_package_arn = ""

    # Retrieve the latest pending model package
    model_package_arn = get_package(MODEL_PACKAGE_GROUP_NAME, approved=False)

    if model_package_arn:
        approved_model_package_arn = approve_package(model_package_arn)
        if not approved_model_package_arn:
            logger.error(f"Can't approve package by ARN. Stop execution.")
    else:
        logger.info(
            f"Can't get package ARN with pending, try to get one with Approved state."
        )
        approved_model_package_arn = get_package(
            MODEL_PACKAGE_GROUP_NAME, approved=True
        )
    return approved_model_package_arn


if __name__ == "__main__":
    """
    Take the latest pending model package and approve it. If there are no pending packages,
    take the latest approved one. Export the model package arn to env vars
    """
    approved_model_package_arn = main()

    # Print result to get standard output or return error code
    if approved_model_package_arn:
        print(approved_model_package_arn)
    else:
        sys.stderr.write(
            "There is problem in scripts/model_management/model_management.py\n\n"
        )
        sys.exit(1)

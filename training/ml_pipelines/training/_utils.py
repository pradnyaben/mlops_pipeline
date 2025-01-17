"""Utility functions for training module."""

import logging

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    # Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version["ImageVersionStatus"] == "CREATED":
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info("Identified the latest image version: %s", image_arn)
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name, Version=version
            )
            return response["ContainerImage"]
    return None


def resolve_ecr_uri(sagemaker_session, image_arn):
    # sourcery skip: raise-from-previous-error, raise-specific-error
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn
    # (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_t = ""
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy="VERSION",
                SortOrder="DESCENDING",
                NextToken=next_t,
            )

            ecr_uri = resolve_ecr_uri_from_image_versions(
                sagemaker_session, response["ImageVersions"], image_name
            )

            if ecr_uri is not None:
                return ecr_uri

            if "NextToken" in response:
                next_t = response["NextToken"]
            else:
                break

        # Return error if no versions of the image found
        error_message = f"No image version found for image name: {image_name}"
        logger.error(error_message)
        raise Exception(error_message)

    except (
        ClientError,
        sagemaker_session.sagemaker_client.exceptions.ResourceNotFound,
    ) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

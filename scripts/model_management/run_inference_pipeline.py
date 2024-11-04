import os
import sys
import time
import boto3
import logging

DEFAULT_DEPLOYMENT_REGION = "us-west-2"
SAGEMAKER_INFERENCE_PIPELINE_NAME = os.getenv("SAGEMAKER_INFERENCE_PIPELINE_NAME", "")

# Connect to Sagemaker
sm_client = boto3.client("sagemaker", region_name=DEFAULT_DEPLOYMENT_REGION)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def execute_inference_pipeline(pipeline_name):
    """Function starts the SageMaker pipeline execution"""
    response = sm_client.start_pipeline_execution(PipelineName=pipeline_name)
    return response["PipelineExecutionArn"]


def wait_pipeline_execution(pipeline_execution_arn):
    while True:
        # Get status each 60 seconds
        time.sleep(60)
        pipeline_execution_status = sm_client.describe_pipeline_execution(
            PipelineExecutionArn=pipeline_execution_arn
        )["PipelineExecutionStatus"]
        # value for status: 'Executing'|'Stopping'|'Stopped'|'Failed'|'Succeeded'
        if pipeline_execution_status in ["Stopping", "Stopped", "Failed"]:
            sys.stderr.write(
                f"Pipeline execution failed. Pipeline status: {pipeline_execution_status}\n\n"
            )
            sys.exit(1)
        elif pipeline_execution_status == "Succeeded":
            print(
                f"Pipeline execution succeeded. Pipeline status: {pipeline_execution_status}"
            )
            break
    return pipeline_execution_status


if __name__ == "__main__":
    pipeline_execution_arn = execute_inference_pipeline(
        SAGEMAKER_INFERENCE_PIPELINE_NAME
    )
    wait_pipeline_execution(pipeline_execution_arn)

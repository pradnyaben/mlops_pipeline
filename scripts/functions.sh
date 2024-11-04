#!/bin/bash

# Env Variables for Sagemaker Pipelines
export AWS_REGION="us-west-2"
PROJECT_NAME="prediction"
PROJECT_ID="heart-failure-poc"
export  SAGEMAKER_TRAINING_PIPELINE_NAME="$PROJECT_NAME-training-$PROJECT_ID"
export SAGEMAKER_INFERENCE_PIPELINE_NAME="$PROJECT_NAME-inference-$PROJECT_ID"
export PYTHONUNBUFFERED=TRUE

echo "Project name: ${PROJECT_NAME}"
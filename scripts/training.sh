#!/usr/bin/env bash
set -e
# This is experiment pipeline for the SageMaker

# shellcheck disable=SC1091
source ./scripts/functions.sh --source-only

if [ -z "$CI" ]; then
    # User mode, provide additional info, fixing issues
    echo "Manual execution"
else
    # Only commands for pipeline, check but not edit
    echo "Pipeline execution"
    # Check if we have ENV var AWS_ROLE_ARN
    if [ -z "${AWS_ROLE_ARN}" ]; then
        echo "No role to assume. Exit with error."
        exit 1
    fi
fi

AWS_REGION="us-west-2"
PROJECT_ID="heart-failure-poc"
PROJECT_NAME="prediction"
ARTIFACT_BUCKET="heartfialurepoc-artifacts-390402562756-us-west-2"
SAGEMAKER_PIPELINE_ROLE_ARN="arn:aws:iam::390402562756:role/TrainingPipelineExecutionRoleARN-mlops-poc"

export ARTIFACT_BUCKET
export SAGEMAKER_PIPELINE_ROLE_ARN

DATA_BIAS_SKIP_CHECK=true
DATA_BIAS_REGISTER_NEW_BASELINE=true
export DATA_BIAS_SKIP_CHECK
export DATA_BIAS_REGISTER_NEW_BASELINE


# Enter to the training folder
pushd ./training/ || exit 1
echo "Start SageMaker Pipeline execution"
pipenv run run-pipeline --module-name ml_pipelines.training.pipeline \
    --role-arn "${SAGEMAKER_PIPELINE_ROLE_ARN}" \
    --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${PROJECT_ID}\"}]" \
    --kwargs "{\"region\":\"${AWS_REGION}\", \"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\", \"default_bucket\":\"${ARTIFACT_BUCKET}\", \"pipeline_name\":\"${SAGEMAKER_TRAINING_PIPELINE_NAME}\", \"model_package_group_name\":\"model-group\", \"project_name\":\"${PROJECT_NAME}\", \"data_bias_register_new_baseline\":\"${DATA_BIAS_REGISTER_NEW_BASELINE}\", \"data_bias_skip_check\":\"${DATA_BIAS_SKIP_CHECK}\"}"
echo "Create/Update of the SageMaker Pipeline and execution completed."
# Exit from the training folder
popd || exit 1
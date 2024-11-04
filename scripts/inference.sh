#!/usr/bin/env bash
set -e
# This is deploy pipeline for the SageMaker

# shellcheck disable=SC1091
source ./scripts/functions.sh --source-only
if [ -z "$CI" ]; then
    # User mode, provide additional info, fixing issues
    echo "Manual execution"
else
    # Only commands for pipeline, check but not edit
    echo "Pipeline execution"
    # Check if we have ENV var TARGET_ENV
    if [ -z "${TARGET_ENV}" ]; then
        echo "The TARGET_ENV is not set. Exit!"
        exit 1
    else
        # These are dev, preprod or prod accounts. Execute accordingly.
        if [[ "${TARGET_ENV}" = "dev" ]]; then
            echo "This is a dev account"
            # This is dev account. Collects values from proper variables
            export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" \
            $(aws sts assume-role-with-web-identity \
            --duration-seconds 3600 \
            --role-session-name "cicd" \
            --role-arn "${AWS_ROLE_ARN}" \
            --web-identity-token "${GITLAB_OIDC_TOKEN}" \
            --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
            --output text ))
        fi
        if [[ "${TARGET_ENV}" = "preprod" ]]; then
            echo "This is a pre-prod account"
            # This is preprod account. Collect values from proper variables
            export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" \
            $(aws sts assume-role-with-web-identity \
            --duration-seconds 3600 \
            --role-session-name "cicd" \
            --role-arn "${AWS_ROLE_ARN}" \
            --web-identity-token "${GITLAB_OIDC_TOKEN}" \
            --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
            --output text ))
        fi
        if [[ "${TARGET_ENV}" = "prod" ]]; then
            echo "This is a prod account"
            # This is prod account. Collect values from proper variables
            export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" \
            $(aws sts assume-role-with-web-identity \
            --duration-seconds 3600 \
            --role-session-name "cicd" \
            --role-arn "${AWS_ROLE_ARN}" \
            --web-identity-token "${GITLAB_OIDC_TOKEN}" \
            --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
            --output text ))
        fi
    fi
    # Execute command examples
    aws sts get-caller-identity
fi

ARTIFACT_BUCKET="s3://heartfialurepoc-artifacts-390402562756-us-west-2"
PROJECT_ID="heart-failure-poc"
PROJECT_NAME="prediction"
SAGEMAKER_PIPELINE_ROLE_ARN="arn:aws:iam::390402562756:role/InferencePipelineExecutionRoleARN-mlops-poc"


# Temporarry for model execute same SM role as for the inference pipeline 
SAGEMAKER_MODEL_ROLE_ARN=$(aws cloudformation list-exports --query "Exports[?Name=='InferencePipelineExecutionRoleARN-${PROJECT_ID}'].Value" --output text)
export ARTIFACT_BUCKET
export SAGEMAKER_PIPELINE_ROLE_ARN
export SAGEMAKER_MODEL_ROLE_ARN



if [[ "${TARGET_ENV}" = "dev" ]] || [[ -z "$CI" ]]; then
    # Compress the inference script and upload it to S3 for use by the model
    echo "Compressing the inference files and uploading them to S3"
    target_file="./inference_scripts.tar.gz"
    tar czf "${target_file}" --directory="./inference/pipeline_scripts/inference" .
    time=$(date +'%Y-%m-%d_%H-%M-%S')
    S3_CODE_PATH="s3://${ARTIFACT_BUCKET}/model-group/InferenceScripts/model-group_${time}.tar.gz"
    export S3_CODE_PATH
    echo "${S3_CODE_PATH}"
    aws s3 cp "${target_file}" "${S3_CODE_PATH}"
    rm "${target_file}"

    echo "Approve the latest pending package"
    MODEL_PACKAGE_ARN=$(pipenv run python ./scripts/model_management/approve_model.py)
    export MODEL_PACKAGE_ARN
    echo "Latest model package arn: ${MODEL_PACKAGE_ARN}"

    echo "Create model from the package"
    MODEL_CONFIG=$(pipenv run python ./scripts/model_management/create_model.py)
    export MODEL_CONFIG
    echo "Model config: ${MODEL_CONFIG}"
fi


if [[ "${TARGET_ENV}" = "preprod" ]] || [[ "${TARGET_ENV}" = "prod" ]]; then
    echo "Creating model"
    MODEL_CONFIG=$(pipenv run python ./scripts/model_management/create_model.py)
    export MODEL_CONFIG
    echo "Model config: ${MODEL_CONFIG}"
fi

# Deploy the resources & inf pipeline
echo "Current CDK version"
npx cdk --version
pipenv run npx cdk deploy --all --require-approval never
PROD_BUCKET="${PROJECT_NAME}-artifacts-${PROJECT_ID}-${PROD_ACC_NUMBER}-eu-central-1"
export PROD_BUCKET
output_destination="s3://${PROD_BUCKET}/model-group/${SAGEMAKER_INFERENCE_PIPELINE_NAME}"
export output_destination
# # If we are not in prod then run the inference pipeline
if ! [[ "${TARGET_ENV}" = "prod" ]]; then
    echo "Run inference pipeline"
    pipenv run python ./scripts/model_management/run_inference_pipeline.py
fi
if [[ "${TARGET_ENV}" = "prod" ]]; then
    echo "Run inference pipeline"
    pipenv run python ./scripts/model_management/run_inference_pipeline.py
fi

"""
Stack creates resources required for running inference:
    - Inference pipeline for batch
    - EventBridge event for triggering the pipeline on schedule
    - Role for the rule
"""

import os
import json

from aws_cdk import (
    Stack,
    CfnOutput,
    aws_iam as iam,
    aws_events as events,
    aws_sagemaker as sagemaker,
)

from constructs import Construct
from inference.pipelines import get_pipeline_definition

REGION = "us-west-2"
MODEL_CONFIG = json.loads(os.getenv("MODEL_CONFIG", ""))
MODEL_NAME = MODEL_CONFIG["model_name"]
TARGET_ENV = os.getenv("TARGET_ENV")
PROJECT_ID = "heart-failure-poc"
PROJECT_NAME = "prediction"
S3_CODE_PATH = os.getenv("S3_CODE_PATH")
ARTIFACT_BUCKET = "s3://heartfialurepoc-artifacts-390402562756-us-west-2"
MODEL_PACKAGE_GROUP_NAME = "model-group"
SAGEMAKER_PIPELINE_ROLE_ARN = os.getenv("SAGEMAKER_PIPELINE_ROLE_ARN")
SAGEMAKER_INFERENCE_PIPELINE_NAME = os.getenv("SAGEMAKER_INFERENCE_PIPELINE_NAME")


class BatchInferenceStack(Stack):
    "Deploy Endpoint stack which provisions sagemaker model endpoint resources"

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        print(f"Model Name: {MODEL_NAME}")

        # Get SM pipeline json definition
        pipeline_definition = get_pipeline_definition.main(
            module_name="inference.pipelines.batch_inference.pipeline",
            role_arn=SAGEMAKER_PIPELINE_ROLE_ARN,
            file_name="inference_pipeline_definition.json",
            tags=json.dumps(
                [
                    {"key": "sagemaker:project_name", "value": PROJECT_NAME},
                    {"key": "sagemaker:project_id", "value": PROJECT_ID},
                    {
                        "key": "sagemaker:deployment-stage",
                        "value": TARGET_ENV,
                    },
                ]
            ),
            kwargs=json.dumps(
                {
                    "region": REGION,
                    "model_name": MODEL_NAME,
                    "project_name": PROJECT_NAME,
                    "default_bucket": ARTIFACT_BUCKET,
                    "role": SAGEMAKER_PIPELINE_ROLE_ARN,
                    "pipeline_name": SAGEMAKER_INFERENCE_PIPELINE_NAME,
                }
            ),
        )

        print(f"Pipeline definition for {TARGET_ENV}: {pipeline_definition}")

        batch_pipeline = sagemaker.CfnPipeline(
            self,
            "BatchInferencePipeline",
            pipeline_name=SAGEMAKER_INFERENCE_PIPELINE_NAME,
            pipeline_definition={"PipelineDefinitionBody": pipeline_definition},
            role_arn=SAGEMAKER_PIPELINE_ROLE_ARN,
        )

        eventbridge_role = iam.Role(
            self,
            "EventBridgeInvokeInferencePipelineRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
            inline_policies={
                "AllowInferencePipelineInvokation": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["sagemaker:StartPipelineExecution"],
                            resources=[
                                f"arn:aws:sagemaker:{self:region}:{self.account}:pipeline/{SAGEMAKER_INFERENCE_PIPELINE_NAME}"
                            ],
                        )
                    ]
                ),
            },
        )

        pipeline_event_rule = events.CfnRule(
            self,
            "InferencePipelineRule",
            schedule_expression="cron(30 19 * * ? *)",
            targets=[
                events.CfnRule.TargetProperty(
                    arn=f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/{batch_pipeline.pipeline_name}",
                    id="InferencePipelineTarget",
                    role_arn=eventbridge_role.role_arn,
                    sage_maker_pipeline_parameters=events.CfnRule.SageMakerPipelineParametersProperty(
                        pipeline_parameter_list=[
                            events.CfnRule.SageMakerPipelineParameterProperty(
                                name="BatchInstanceType", value="ml.t2.medium"
                            )
                        ]
                    ),
                )
            ],
        )

        pipeline_event_rule.add_depends_on(batch_pipeline)

        CfnOutput(
            self,
            "EventBridgeRuleArn",
            value=pipeline_event_rule.attr_arn,
            export_name=f"EventBridgeRuleArn-{PROJECT_ID}",
        )

        CfnOutput(
            self, "ModelName", value=MODEL_NAME, export_name=f"ModelName-{PROJECT_ID}"
        )
        CfnOutput(
            self,
            "ModelPackageArn",
            value=MODEL_CONFIG["model_package_arn"],
            export_name=f"ModelPackageArn-{PROJECT_ID}",
        )

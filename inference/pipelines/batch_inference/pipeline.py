import os
import uuid
from time import strftime
import logging
import json
from datetime import datetime

import boto3
import boto3.session
import sagemaker
import sagemaker.session
from sagemaker.network import NetworkConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn import SKLearn as SKLearnClass
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.processing import (
    FrameworkProcessor,
    ProcessingOutput,
    ProcessingInput,
)
from sagemaker.workflow.steps import (
    ProcessingStep,
    Transformer,
    TransformStep,
    TransformInput,
    CacheConfig,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.inputs import BatchDataCaptureConfig

logger = logging.getLogger(__name__)

MODEL_CONFIG = json.loads(os.getenv("MODEL_CONFIG", ""))
MODEL_PACKAGE_GROUP_NAME = "model-group"
PROJECT_ID = "heart-failure-poc"


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        repsonse = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = repsonse["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)

    except Exception as e:
        print(f"Error getting project tag: {e}")
    return new_tags


def get_pipeline(
    region,
    role=None,
    model_name=None,
    default_bucket=None,
    pipeline_name="inference-pipeline",
    project_name="prediction",
):
    logger.info("role: %s", role)
    sagemaker_session = get_session(region, default_bucket)

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

        print(f"Model Name: {model_name}")

        date = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        pipeline_run_id = str(uuid.uuid4())[:8]
        output_destination = f"s3://{default_bucket}/{MODEL_PACKAGE_GROUP_NAME}/{pipeline_name}/{project_name}-inference--{date}--{pipeline_run_id}"
        print(f"Sagemaker Version: {sagemaker.__version__}")

        framework_version = "1.2-1"

        cache_config = CacheConfig(enable_caching=True, expire_after="30d")

        processing_instance_type = "ml.t3.medium"
        processing_instance_count = ParameterInteger(
            name="ProcessingInstanceCount", default_value=1
        )

        batch_inference_instance_count = ParameterInteger(
            "BatchInstanceCount", default_value=1
        )

        batch_inference_instance_type = ParameterString(
            "BatchInstanceType", default_value="ml.t2.medium"
        )

        input_path = ParameterString(
            "InputPath",
            default_value=f"s3://{default_bucket}/Heart_Failure_Inference_data/heart.csv",
        )

        #### ----- 1. Preprocessing ------ !#####
        preprocess_job_name = f"preprocessing--{date}"

        sklearn_preprocessing = FrameworkProcessor(
            role=role,
            estimator_cls=SKLearnClass,
            framework_version=framework_version,
            instance_count=processing_instance_count,
            instance_type=processing_instance_type,
            sagemaker_session=sagemaker_session,
            code_location=output_destination,
            # network_config=network_config,
        )

        step_args = sklearn_preprocessing.run(
            job_name=preprocess_job_name,
            code="preprocessing.py",
            source_dir="inference/pipeline_scripts/preprocessing",
            inputs=[
                ProcessingInput(
                    source=input_path, destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="preprocess_inference",
                    source="/opt/ml/processing/output",
                    destination=Join(
                        on="/",
                        values=[output_destination, preprocess_job_name, "output"],
                    ),
                )
            ],
        )

        step_preprocess = ProcessingStep(
            name=f"{project_name}-preprocessing",
            step_args=step_args,
            cache_config=cache_config,
        )

        ##### ------ 2. Transform ------ ####
        transform_job_name = f"batch-transform--{date}"

        transformer = Transformer(
            model_name=model_name,
            instance_type=batch_inference_instance_type,
            instance_count=batch_inference_instance_count,
            max_payload=50,
            accept="text/csv",
            strategy="MultiRecord",
            assemble_with="Line",
            output_path=f"{output_destination,}/{transform_job_name}/output",
        )

        s3_capture_upload_path = f"s3://{default_bucket}/monitoring_model/capture_data"

        batch_data_cpature_config = BatchDataCaptureConfig(
            destination_s3_uri=s3_capture_upload_path,
        )

        step_transform = TransformStep(
            name=f"{project_name}-transform",
            transformer=transformer,
            inputs=TransformInput(
                data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "preprocess_inference"
                ].S3Output.S3Uri,
                input_filter="$[1:]",
                join_source="Input",
                output_filter="$[0,-1]",
                content_type="text/csv",
                split_type="Line",
                batch_data_capture_config=batch_data_cpature_config,
            ),
            cache_config=cache_config,
        )

        ##### ----- 3. Data Quality Monitoring ------ #####

        data_monitoring_job_name = f"data-quality-monitoring--{date}"
        baseline_results_uri = (
            f"s3://{default_bucket}/{MODEL_CONFIG['data_quality_baseline_uri']}"
        )

        job_config = CheckJobConfig(
            role=role,
            instance_count=1,
            instance_type=processing_instance_type,
            volume_size_in_gb=1,
            max_runtime_in_seconds=3600,
            base_job_name=data_monitoring_job_name,
            sagemaker_session=sagemaker_session,
            # network_config=network_config,
        )

        data_quality_config = DataQualityCheckConfig(
            baseline_dataset=step_preprocess.arguments["ProcessingOutputConfig"][
                "Outputs"
            ][0]["S3Output"]["S3Uri"],
            dataset_format=DatasetFormat.csv(header=False),
            output_s3_uri=f"{output_destination}/{data_monitoring_job_name}/reports",
        )

        step_data_monitor = QualityCheckStep(
            name=f"{project_name}-monitor-data-quality",
            skip_check=False,
            register_new_baseline=False,
            quality_check_config=data_quality_config,
            check_job_config=job_config,
            fail_on_violation=False,
            supplied_baseline_statistics=f"{baseline_results_uri}/statistics.json",
            supplied_baseline_constraints=f"{baseline_results_uri}/constraints.json",
            depends_on=[step_preprocess],
        )

        #### ------ Pipeline Instance ----- ####
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[
                model_name,
                processing_instance_type,
                processing_instance_count,
                batch_inference_instance_count,
                batch_inference_instance_type,
                input_path,
            ],
            steps=[
                step_preprocess,
                step_transform,
                step_data_monitor,
            ],
            sagemaker_session=sagemaker_session,
        )

        return pipeline

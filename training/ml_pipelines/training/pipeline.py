import boto3
import logging
import uuid
from datetime import datetime
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
import sagemaker.session
from sagemaker import clarify, image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.network import NetworkConfig
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearn as SKLearnClass
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import (
    ContinuousParameter,
    HyperparameterTuner,
    CategoricalParameter,
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    ClarifyCheckStep,
    ModelExplainabilityCheckConfig,
    ModelPredictedLabelConfig,
    ModelBiasCheckConfig,
    DataBiasCheckConfig,
)
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TuningStep

logger = logging.getLogger(__name__)


def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="model-group",
    pipeline_name="training-pipeline",
    project_name="prediction",
    data_bias_register_new_baseline=True,
    data_bias_skip_check=True,
):
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    date = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    pipeline_run_id = str(uuid.uuid4())[:8]
    output_destination = f"s3://{default_bucket}/{model_package_group_name}/{pipeline_name}/{project_name}-train--{date}--{pipeline_run_id}"

    print(f"Output Destination: {output_destination}")
    print(f"Sagemaker Version: {sagemaker.__version__}")

    framework_version = "1.2-1"

    processing_instance_type = "ml.m2.large"
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    training_instance_type = "ml.m2.large"

    input_data = ParameterString(
        name="InputDataURL",
        default_value="s3://heartfialurepoc-artifacts-390402562756-us-west-2/heart.csv",
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    if type(data_bias_register_new_baseline) == str:
        if data_bias_register_new_baseline in ["True", "true"]:
            data_bias_register_new_baseline = True
        elif data_bias_register_new_baseline in ["False", "false"]:
            data_bias_register_new_baseline = False
    else:
        pass

    if type(data_bias_skip_check) == str:
        if data_bias_skip_check in ["True", "true"]:
            data_bias_skip_check = True
        elif data_bias_skip_check in ["False", "false"]:
            data_bias_skip_check = False
    else:
        pass

    #### ----- 1. Preprocessing ----- ####
    preprocessing_job_name = f"preprocessing-{date}"

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
        job_name=preprocessing_job_name,
        code="preprocessing.py",
        source_dir="training_pipeline_scripts/preprocessing",
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=Join(
                    on="/",
                    values=[
                        output_destination,
                        preprocessing_job_name,
                        "output",
                        "train",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination=Join(
                    on="/",
                    values=[
                        output_destination,
                        preprocessing_job_name,
                        "output",
                        "validation",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=Join(
                    on="/",
                    values=[
                        output_destination,
                        preprocessing_job_name,
                        "output",
                        "test",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="baseline",
                source="/opt/ml/processing/output/baseline",
                destination=Join(
                    on="/",
                    values=[
                        output_destination,
                        preprocessing_job_name,
                        "output",
                        "baseline",
                    ],
                ),
            ),
        ],
        arguments=["--val-size", "0.15", "--test-size", "0.15"],
    )

    step_preprocess = ProcessingStep(
        name=f"{project_name}-preprocessing",
        step_args=step_args,
        cache_config=cache_config,
    )

    #### ------ 2. Data Quality Monitoring Baseline ------ ####
    data_monitoring_job_name = f"data-quality-baseline--{date}"

    job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m2.large",
        volume_size_in_gb=1,
        max_runtime_in_seconds=3600,
        base_job_name=data_monitoring_job_name,
        sagemaker_session=sagemaker_session,
        # network_config=network_config,
    )

    data_quality_config = DataQualityCheckConfig(
        baseline_dataset=step_preprocess.arguments["ProcessingOutputConfig"]["Outputs"][
            0
        ]["S3Output"]["S3Uri"],
        dataset_format=DatasetFormat.csv(),
        output_s3_uri=f"{output_destination}/{data_monitoring_job_name}/baselining/results",
    )

    step_data_quality_baseline = QualityCheckStep(
        name=f"{project_name}-data-quality-baseline",
        skip_check=True,
        register_new_baseline=True,
        quality_check_config=data_quality_config,
        check_job_config=job_config,
    )

    step_data_quality_baseline.add_depends_on([step_preprocess])

    #### ----- 3. Data Bias ----- ####

    s3_data_preprocessed_input_path = (
        step_preprocess.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri
    )

    bias_report_output_path = f"s3://{default_bucket}/clarify-bias/{model_package_group_name}/data-bias/{pipeline_name}/{project_name}-train--{date}--{pipeline_run_id}"

    clarify_processor = clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type="ml.m2.large",
        sagemaker_session=sagemaker_session,
    )

    bias_data_config = clarify.DataConfig(
        s3_data_input_path=s3_data_preprocessed_input_path,
        s3_output_path=bias_report_output_path,
        label="HeartDisease",
        dataset_type="text/csv",
    )

    bias_config = clarify.BiasConfig(
        label_values_or_threshold=[1],
        facet_name="Sex",
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=bias_data_config,
        data_bias_config=bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name=f"{project_name}-data-bias",
        clarify_check_config=data_bias_check_config,
        check_job_config=job_config,
        skip_check=data_bias_skip_check,
        register_new_baseline=data_bias_register_new_baseline,
        model_package_group_name=model_package_group_name,
    )

    #### ----- 4. HPO ------ #####
    sklearn_estimator = SKLearn(
        role=role,
        entry_point="train_lreg.py",
        source_dir="training_pipeline_scripts/training",
        instance_type=processing_instance_type,
        framework_version=framework_version,
        output_path=output_destination,
        code_location=output_destination,
    )

    objective_metric_name = "accuracy"

    hyperparameter_ranges = {
        "C": ContinuousParameter(0.01, 10),
        "penalty": CategoricalParameter(["l1", "l2"]),
        "solver": CategoricalParameter(["liblinear", "saga", "lbfgs"]),
    }

    metric_definitions = [{"Name": "accuracy", "Regex": "Accuracy: ([0-9\\.]+)"}]

    hpo_tuner = HyperparameterTuner(
        estimator=sklearn_estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=2,
        max_parallel_jobs=2,
        strategy="Random",
    )

    step_tune = TuningStep(
        name=f"{project_name}-tune-model",
        tuner=hpo_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    #### ----- 5. Evaluation ----- #####
    evaluation_job_name = f"evaluation-{date}"

    sklearn_evaluation = FrameworkProcessor(
        role=role,
        estimator_cls=SKLearnClass,
        framework_version=framework_version,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        code_location=output_destination,
        sagemaker_session=sagemaker_session,
    )

    step_args = sklearn_evaluation.run(
        job_name=evaluation_job_name,
        code="evaluation.py",
        source_dir="training_pipeline_scripts/evaluation",
        inputs=[
            ProcessingInput(
                source=step_tune.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=output_destination.split("/", 3)[2],
                    prefix=output_destination.split("/", 3)[3],
                ),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=Join(
                    on="/", values=[output_destination, evaluation_job_name, "output"]
                ),
            ),
        ],
    )

    step_evaluation = ProcessingStep(
        name=f"{project_name}-evaluation",
        step_args=step_args,
        cache_config=cache_config,
    )

    #### ----- 6. Model Create ----- ####
    processing_image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=framework_version,
        py_version="py3",
        instance_type=training_instance_type,
    )

    model = Model(
        role=role,
        name=f"{project_name}-model",
        entry_point="inference.py",
        source_dir="training_pipeline_scripts/inference",
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=output_destination.split("/", 3)[2],
            prefix=output_destination.split("/", 3)[3],
        ),
        image_uri=processing_image_uri,
        code_location=output_destination,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=False,
    )

    step_create_model = ModelStep(
        name=f"{project_name}-create-model",
        step_args=model.create(instance_type=training_instance_type),
    )

    #### ----- 7. Clarify Model Bias ----- ####
    model_bias_output_path = f"s3://{default_bucket}/clarify-bias/{model_package_group_name}/model-bias-output/{pipeline_name}/{project_name}-train--{date}--{pipeline_run_id}"
    model_bias_cfg_output_path = f"s3://{default_bucket}/clarify-bias/{model_package_group_name}/model-bias-cfg/{pipeline_name}/{project_name}-train--{date}--{pipeline_run_id}"

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        volume_size_in_gb=1,
        sagemaker_session=sagemaker_session,
        # network_config=network_config,
    )

    model_bias_data_config = clarify.DataConfig(
        s3_data_input_path=step_preprocess.properties.ProcessingOutputConfig.Outputs[
            "train"
        ].S3Output.S3Uri,
        s3_output_path=model_bias_output_path,
        s3_analysis_config_output_path=model_bias_cfg_output_path,
        label="HeartDisease",
        dataset_type="text/csv",
    )

    model_config = clarify.ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type="ml.m2.large",
    )

    model_bias_config = clarify.BiasConfig(
        label_values_or_threshold=[1], facet_name=["Sex"]
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig(),
    )

    model_bias_check_step = ClarifyCheckStep(
        name=f"{project_name}-model-bias",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=True,
        register_new_baseline=True,
        model_package_group_name=model_package_group_name,
    )

    #### ------ 8. Clarify Model Explainability ----- ####
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        volume_size_in_gb=1,
        sagemaker_session=sagemaker_session,
        # network_config=network_config,
    )

    model_config = clarify.ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_type=training_instance_type,
        instance_count=1,
    )

    baseline_s3_str = (
        f"{output_destination}/{preprocessing_job_name}/output/baseline/baseline.csv"
    )

    shap_config = clarify.SHAPConfig(
        baseline=baseline_s3_str, num_samples=20, agg_method="mean_abs"
    )

    explainability_job_name = f"explainability--{date}"

    explainability_output_path = f"{output_destination}/{explainability_job_name}"
    model_explainability_analysis_cfg_output_path = (
        f"{output_destination}/{explainability_job_name}"
    )

    explainability_data_config = clarify.DataConfig(
        s3_data_input_path=step_preprocess.properties.ProcessingOutputConfig.Outputs[
            "test"
        ].S3Output.S3Uri,
        label="HeartDisease",
        dataset_type="text/csv",
        s3_output_path=explainability_output_path,
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
    )

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )

    model_explainability_check_step = ClarifyCheckStep(
        skip_check=True,
        register_new_baseline=True,
        check_job_config=check_job_config,
        model_package_group_name=model_package_group_name,
        name=f"{project_name}-model-explainability-step",
        clarify_check_config=model_explainability_check_config,
    )

    #### ----- 9. Export Metrics ------ ####
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluation.arguments["ProcessingOutputConfig"]["Outputs"][0][
                        "S3Output"
                    ]["S3Uri"],
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    model_explainability_check_step.arguments["ProcessingOutputConfig"][
                        "Outputs"
                    ][0]["S3Output"]["S3Uri"],
                    "analysis.json",
                ],
            ),
            content_type="application/json",
        ),
        bias_pre_training=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    data_bias_check_step.arguments["ProcessingOutputConfig"]["Outputs"][
                        0
                    ]["S3Output"]["S3Uri"],
                    "analysis.json",
                ],
            ),
            content_type="application/json",
        ),
        bias_post_training=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    model_bias_check_step.arguments["ProcessingOutputConfig"][
                        "Outputs"
                    ][0]["S3Output"]["S3Uri"],
                    "analysis.json",
                ],
            ),
            content_type="application/json",
        ),
    )

    #### ----- 10. Register Model ------ ####
    step_register = RegisterModel(
        name=f"{project_name}-register-model",
        estimator=sklearn_estimator,
        image_uri=processing_image_uri,
        model_data=step_tune.get_top_model_s3_uri(
            top_k=0,
            s3_bucket=output_destination.split("/", 3)[2],
            prefix=output_destination.split("/", 3)[3],
        ),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m2.large"],
        model_package_group_name=model_package_group_name,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    #### ----- Pipeline Instance ----- ####
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data,
        ],
        steps=[
            step_preprocess,
            step_data_quality_baseline,
            data_bias_check_step,
            step_tune,
            step_evaluation,
            step_create_model,
            model_bias_check_step,
            model_explainability_check_step,
            step_register,
        ],
        sagemaker_session=sagemaker_session,
    )
    return pipeline

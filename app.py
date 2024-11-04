"CDK stack for deploying resources required for inference"

import os
import aws_cdk as cdk

from cdk_stacks.batch_inference_stack import BatchInferenceStack

app = cdk.App()

target_environment = os.getenv("TARGET_ENV", "dev")
PROJECT_ID = "heart-failure-poc"
# Check if we execute in the pipeline or not.
REAL_ACCOUNT = bool(os.getenv("CI", ""))
if REAL_ACCOUNT:
    print("!!! Generate stacks for real AWS account !!!")
else:
    print("Generate regular test stacks")

BatchInferenceStack(app, f"BatchInference-{PROJECT_ID}")

app.synth()

from __future__ import absolute_import

import argparse
import json
from os import pipe
import sys

from _utils import (
    get_pipeline_driver,
    convert_struct,
    get_pipeline_custom_tags,
)


def main(module_name, role_arn, tags, kwargs, description=""):
    if module_name is None or role_arn is None:
        sys.exit(2)
    tags = convert_struct(tags)

    try:
        pipeline = get_pipeline_driver(module_name, kwargs)
        print(
            "#### Creating/Updating a Sagemaker pipeline with the following definition:"
        )

        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        upsert_response = pipeline.upsert(
            role_arn=role_arn, description=description, tags=tags
        )

        print("\n ### Created/Updated Sagemaker pipeline: Response received:")
        print(upsert_response)

        execution = pipeline.start()
        execution.wait()

        return pipeline.definition()
    except Exception as e:
        print(f"Execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Creates or Updates and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module-name",
        type=str,
        help="The module name of the pipeline to import.",
    )

    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )

    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role-arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )

    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline",
    )

    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )

    args = parser.parse_args()

    pipeline_arn = main(
        args.module_name, args.role_arn, args.tags, args.kwargs, args.description
    )

    print(f"Pipeline ARN: {pipeline_arn}")

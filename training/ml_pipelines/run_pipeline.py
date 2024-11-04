""" A CLI to create or update and run pipelines. """

from __future__ import absolute_import

import argparse
import json
import sys

from ml_pipelines._utils import (
    get_pipeline_driver,
    convert_struct,
    get_pipeline_custom_tags,
)


def main():

    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline script"
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="The module name of the pipeline to import",
    )

    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword argument for the pipeline generation (if supported)",
    )

    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
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
        help="""Lise of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )

    args = parser.parse_args()

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print(
            "### Creating/Updating a Sagemaker pipeline with the following definition:"
        )
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)

        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description, tags=all_tags
        )
        print("\n#### Created/Updated Sagemaker PIpeline: Response received")
        print(upsert_response)

        execution = pipeline.start()
        print(f"\n#### Execution started with pipelineexecutionarn: {execution.arn}")

        print("Waiting for execution to finish")
        execution.wait(delay=120, max_attempts=200)
        print("\n Execution completed. Execution step details")

        print(execution.list_steps())
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

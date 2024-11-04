""" A CLI to get pipeline definition from pipeline modules """

from __future__ import absolute_import

import sys
import json
import argparse

from ._utils import (
    get_pipeline_driver,
    convert_struct,
)


def main(module_name, role_arn, tags, kwargs, file_name=None):
    if module_name is None or role_arn is None:
        sys.exit(2)
    tags = convert_struct(tags)

    try:
        pipeline = get_pipeline_driver(module_name, kwargs)
        content = pipeline.definition()
        print(
            "#### Creating/Updating a sagemaker pipeline with the following definition:"
        )
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        if file_name is not None:
            print(f"Saving pipeline definition to {file_name}")
            with open(file_name, "w", encoding="utf8") as f:
                f.write(content)

        return pipeline.definition()

    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
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
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )

    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role",
    )

    parser.add_argument(
        "-f",
        "--file-name",
        dest="file_name",
        type=str,
        default=None,
        help="The file to output the pipeline definition json to.",
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
        args.module_name,
        args.role_arn,
        args.tags,
        args.kwargs,
        args.description,
        args.file_name,
    )
    print(f"Pipeline ARN: {pipeline_arn}")

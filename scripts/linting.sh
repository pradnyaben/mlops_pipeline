#!/bin/bash
#!/usr/bin/env bash
set -e

if [ -z "$CI" ]; then
    # User mode, provide additional info, fixing issues
    echo "Manual execution"
    pipenv run black --version
    pipenv run black --verbose . --extend-exclude="tests"
else
    # Only commands for pipeline, check but not edit
    echo "Pipeline execution"
    pipenv run black --version
    pipenv run black --check . --extend-exclude="tests"
fi
pipenv run pylint --version
pipenv run pylint ./* --fail-under=2 --ignore-patterns='^build$,^cdk.out$,^ml_pipelines.egg-info$,^node_modules$,^tests$' --ignore='cdk.json,README.md,package.json,package-lock.json,Pipfile'

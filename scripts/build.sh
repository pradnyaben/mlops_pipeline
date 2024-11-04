#!/usr/bin/env bash
set -e

source ./scripts/functions.sh --source-only
#
if [ -z "$CI" ]; then
    echo "Manual execution"
else
    echo "Pipeline execution"
fi

# Synth the stack
echo "Current CDK version"
npx cdk --version
echo "Run npx in the pipevn for synth"
pipenv run npx cdk synth

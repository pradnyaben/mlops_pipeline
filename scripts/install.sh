#!/usr/bin/env bash
set -e

echo "Installing pipenv version 2024.2.0"
pip install pipenv==2024.2.0

if [ -z "$CI" ]; then
    # User mode, provide additional info
    echo "Manual execution"
    # Change PIP URL local install
    export PIP_URL="https://pypi.org/simple"
    npm i
else
    # Only commands for pipeline
    echo "Pipeline execution"
    if ! command -v npm; then
        # This is python container.
        echo "The npm is not installed, skip."
    else
        # Npm exists, install packages
        npm --version
        npm i
    fi
fi

#Override PIP_URL
export PIP_URL="https://pypi.org/simple"

# Install common tools for tests
echo "Install dependendencies for testing"
pipenv install --dev --skip-lock
pipenv graph

# Enter to the training folder
echo "Install dependendencies for training"
pushd ./training/ || exit 1
pipenv --version
pipenv install --dev --skip-lock -e . 
pipenv graph

# Exit from the training folder
popd || exit 1

# Enter to the inference folder
echo "Install dependendencies for inference"
pushd ./inference/ || exit 1
pipenv --version
pipenv install --skip-lock
pipenv graph

# Exit from the inference folder
popd || exit 1

# Install specific versions for the training folder dependencies
echo "Installing specific versions of packages for the training folder"
pushd ./training/ || exit 1
# Assuming you have a requirements.txt for specific versions, use it like this:
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found for specific versions in the training folder."
fi
# Exit from the training folder
popd || exit 1

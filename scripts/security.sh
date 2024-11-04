#!/bin/bash
#!/usr/bin/env bash
set -e

pipenv run bandit --version
pipenv run bandit -r . -x "**build/*,**tests/*"
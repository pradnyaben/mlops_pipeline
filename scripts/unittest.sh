#!/usr/bin/env bash
set -e

if [ -z "$CI" ]; then
    # User mode, provide additional info
    echo "Manual Execution"
    pipenv run coverage run --source ./ --omit="./app.py,./tests/*" -m pytest -sv
    pipenv run coverage report --fail-under=80
else
    # Only commands for pipeline, check but not edit
    echo "Pipeline execution"
    some_tests_failed=0 ; average_coverage_failed=0 ; individual_coverage_failed=0
    pipenv run coverage run --source ./ --omit="./app.py,./tests/*" -m pytest -sv --junitxml=report.xml || some_tests_failed=1
    pipenv run coverage xml
    pipenv run coverage report --fail-under=${MINIMAL_UNIT_TEST_COVERAGE_PERCENT:=50} || average_coverage_failed=1
    pipenv run coverage report | tail -n +3 | head -n -2 | awk -v treshold="$MINIMAL_UNIT_TEST_COVERAGE_PERCENT" ' int($4) &lt; treshold {exit 3} ' || individual_coverage_failed=1
   ((some_tests_failed)) &amp;&amp; echo "At least one test failed." &amp;&amp; exit 10
   ((average_coverage_failed)) &amp;&amp; echo "Average tests coverage is below minimal treshold of $MINIMAL_UNIT_TEST_COVERAGE_PERCENT%." &amp;&amp; exit 11
   ((individual_coverage_failed)) &amp;&amp; echo "At least one individual tests coverage is below minimal treshold of $MINIMAL_UNIT_TEST_COVERAGE_PERCENT%." &amp;&amp; exit 12
   echo "All tests passed and are above minimal coverage." ; exit 0
fi

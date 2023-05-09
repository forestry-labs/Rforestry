#!/usr/bin/env bash
set -e

echo "=== Running Python script started  ==="
python3 python_test.py > /tmp/python_output
echo "=== Running Python script finished ==="
echo
echo
echo "=== Running R script started       ==="
Rscript R_test.R > /tmp/R_output
echo "=== Running R script finished      ==="
echo

echo "*** Differences                    ***"

diff -s /tmp/python_output /tmp/R_output


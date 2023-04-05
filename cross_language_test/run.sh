#!/usr/bin/env bash

echo "=== Running Python script started  ==="
python3 python_test.py > /tmp/python_output
echo "=== Running Python script finished ==="
echo
echo "=== Build and install R package    ==="
R CMD build ../R
PACKAGE=$(ls -1 *.tar.gz) R -e "install.packages(Sys.getenv('PACKAGE'), repos=NULL, type='source')"
echo "=== R package installed            ==="
echo
echo "=== Running R script started       ==="
Rscript R_test.R > /tmp/R_output
echo "=== Running R script finished      ==="
echo

echo "*** Differences                    ***"
diff -s /tmp/python_output /tmp/R_output
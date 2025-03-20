#!/bin/bash
# Set up any environment variables or paths if needed
# export PYTHONPATH=/path/to/add

# Activate virtual environment if you're using one
# source /path/to/venv/bin/activate

# Use unbuffered output for Python (-u flag)
# and redirect both stdout and stderr to the log file
python3 -u scripts/seq2seq_train.py > training_output.log 2>&1
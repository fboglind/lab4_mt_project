#!/bin/bash
# Set up any environment variables or paths if needed
# export PYTHONPATH=/path/to/add

# Use unbuffered output for Python (-u flag)
# and redirect both stdout and stderr to the log file
nohup python3 -u scripts/seq2seq_train.py \
  --batch-size 8 \
  --accum-steps 4 \
  > training_output.log 2>&1 &
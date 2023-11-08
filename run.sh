#!/bin/bash

# Check if exactly two arguments are given ($# is the number of arguments)
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

# Assign the first argument to INPUT_PATH
INPUT_PATH="$1"

# Assign the second argument to OUTPUT_PATH
OUTPUT_PATH="$2"

# Run the Python program with the provided arguments
python ./inference.py --model_path ./model --validation_dataset "$INPUT_PATH" --output_file_name "$OUTPUT_PATH"


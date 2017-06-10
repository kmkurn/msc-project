#!/bin/bash

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DATASET_FNAME="dataset.txt"
WHICHES=( "train" "valid" "test" )
NUM_SPLIT=5

for i in `seq 0 $((NUM_SPLIT-1))`; do
    for which in "${WHICHES[@]}"; do
        mv "$INPUT_DIR/$DATASET_FNAME.$i.$which" "$OUTPUT_DIR/$which.$i.txt"
    done
done

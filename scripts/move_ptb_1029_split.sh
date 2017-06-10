#!/bin/bash

usage() {
    echo -e "Move and rename PTB 1029 dataset to a directory.\n"
    cat <<EOF
Usage: $0 [-h] INPUT_DIR OUTPUT_DIR

-h, --help		Display help

Note that under the input directory, the dataset filenames are assumed
to be "dataset.txt.0.train", "dataset.txt.0.valid", etc. The number
of splits is assumed to be 5.
EOF
}

if [ "$@" = "-h" -o "$@" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -le 1 ]; then
    usage
    exit 1
fi

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

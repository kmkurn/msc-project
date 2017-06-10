#!/bin/bash

usage() {
    echo -e "Move and rename PTB 1029 dataset to a directory.\n"
    cat <<EOF
Usage: $0 [-h] INPUT_DIR OUTPUT_DIR

-h, --help		Display help

Note that under the input directory, the dataset filenames are assumed
to be "dataset.txt.0.train", "dataset.txt.0.valid", etc. The number
of splits is assumed to be 5. This script also removes empty lines
which causing problem when generating the oracle files.
EOF
}

while true; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [ "$#" -le 1 ]; then
                echo "Error: expected 2 positional arguments but got $#" >&2
                usage
                exit 1
            fi
            input_dir="$1"
            output_dir="$2"
            break
            ;;
    esac
done

DATASET_FNAME="dataset.txt"
WHICHES=( "train" "valid" "test" )
NUM_SPLIT=5

for i in `seq 0 $((NUM_SPLIT-1))`; do
    for which in "${WHICHES[@]}"; do
        sed '/^$/d' "$input_dir/$DATASET_FNAME.$i.$which" > "$output_dir/$which.$i.txt"
    done
done

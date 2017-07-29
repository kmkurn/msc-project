#!/bin/bash

WORK_DIR=`pwd`
SCRIPT_DIR=`dirname $(readlink -f "$0")`
TMP_DIR="/tmp"
PARSER_FILENAME="stanford-parser-full-2016-10-31.zip"
PARSER_URL="https://nlp.stanford.edu/software/$PARSER_FILENAME"

# Download parser
cd "$TMP_DIR"
curl -O "$PARSER_URL"
unzip "$PARSER_FILENAME"

# Install parser under misc
mv "$(basename "$PARSER_FILENAME" .zip)" "$SCRIPT_DIR/stanford-parser"

# Cleanup
rm "$TMP_DIR/$PARSER_FILENAME"
cd "$WORK_DIR"

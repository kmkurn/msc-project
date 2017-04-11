#!/bin/bash

WORK_DIR=`pwd`
SCRIPT_DIR=`dirname $(readlink -f "$0")`
TMP_DIR="/tmp"
EVALB_URL="http://nlp.cs.nyu.edu/evalb/EVALB.tgz"

# Download evalb
cd "$TMP_DIR"
curl -O "$EVALB_URL"
tar xvfz EVALB.tgz

# Compile evalb
cd EVALB
sed -e 's/malloc\.h/malloc\/malloc.h/' evalb.c > new_evalb.c
mv new_evalb.c evalb.c
make

# Install evalb under rnng
cd "$WORK_DIR"
mv "$TMP_DIR/EVALB" "$SCRIPT_DIR/rnng/"

# Cleanup
rm "$TMP_DIR/EVALB.tgz"

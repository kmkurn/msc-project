#!/usr/bin/env python

from argparse import ArgumentParser
from math import sqrt
import os
import subprocess
import sys


if __name__ == '__main__':
    parser = ArgumentParser(
        description="""Wrapper script to cluster words using Brown clustering. The number
                    of cluster is square root of the number of unique words in the given
                    text file. Note that brown-cluster must be installed on your system.
                    (see https://github.com/percyliang/brown-cluster)""")
    parser.add_argument('file', help='path to text file whose words are to be clustered')
    parser.add_argument('--wcluster', default='wcluster', help='path to wcluster binary')
    parser.add_argument('--outdir', default=os.getcwd(), help='path to output directory')
    args = parser.parse_args()

    vocab = set()
    with open(args.file) as f:
        for line in f:
            for word in line.strip().split():
                vocab.add(word)
    clust_size = int(sqrt(len(vocab)))
    print(f'vocab size: {len(vocab)}', file=sys.stderr)
    print(f'cluster size: {clust_size}', file=sys.stderr)

    cmd = f'{args.wcluster} --text {args.file} --c {clust_size} --output_dir {args.outdir}'
    print(f'command: {cmd}', file=sys.stderr)
    subprocess.run(cmd, check=True, shell=True)

#!/usr/bin/env python

from __future__ import print_function

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
    print('vocab size: {}'.format(len(vocab)), file=sys.stderr)
    print('cluster size: {}'.format(clust_size), file=sys.stderr)

    cmd = '{} --text {} --c {} --output_dir {}'.format(
        args.wcluster, args.file, clust_size, args.outdir)
    print('command: {}'.format(cmd), file=sys.stderr)
    subprocess.call(cmd, shell=True)

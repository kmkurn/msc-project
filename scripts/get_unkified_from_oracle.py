#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser

from pypkg.oracle import oracle_iter, gen_oracle_iter


if __name__ == '__main__':
    parser = ArgumentParser(description='Get unkified sentence from a given oracle file.')
    parser.add_argument('file', help='path to oracle file')
    parser.add_argument('-g', '--gen', action='store_true',
                        help='whether the oracle is for the generative model')
    args = parser.parse_args()

    the_iter = gen_oracle_iter if args.gen else oracle_iter
    with open(args.file) as f:
        for oracle in the_iter(f):
            print(oracle.unkified)

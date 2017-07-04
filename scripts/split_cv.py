#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser
import os
import random


def get_split_indices(size, k=5):
    if k < 2:
        raise ValueError('`k` must be at least 2')
    if size < k:
        raise ValueError('`size` must be greater than number of splits `k`')

    indices = list(range(size))
    random.shuffle(indices)
    fold_sizes = [size // k for _ in range(k)]
    for i in range(size % k):
        fold_sizes[i] += 1
    for i in range(k):
        skip = sum(fold_sizes[:i])
        fsz = fold_sizes[i]
        yield indices[skip:skip+fsz]


def split(iterable, indices, hold_out=0.):
    indices = set(indices)
    selected, discarded = [], []
    for i, x in enumerate(iterable):
        if i in indices:
            selected.append(x)
        else:
            discarded.append(x)

    random.shuffle(discarded)
    n = int(hold_out*len(discarded))
    return selected, discarded[:n], discarded[n:]


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate CV split of the given file.')
    parser.add_argument('file', help='path to file to split')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument('-k', type=int, default=5, help='number of splits')
    parser.add_argument('--output-dir',
                        help='output directory (default is the same as file\'s directory)')
    parser.add_argument('--hold-out', type=float, default=0.25,
                        help='proportion of training set that will '
                        'be held out as validation set')
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = os.path.dirname(args.file)
    else:
        output_dir = args.output_dir
    basename = os.path.basename(args.file)

    random.seed(args.seed)
    with open(args.file) as f:
        lines = f.readlines()

    for i, indices in enumerate(get_split_indices(len(lines), k=args.k)):
        test, valid, train = split(lines, indices, hold_out=args.hold_out)
        fname_test = os.path.join(output_dir, '{}.{}.test'.format(basename, i))
        fname_valid = os.path.join(output_dir, '{}.{}.valid'.format(basename, i))
        fname_train = os.path.join(output_dir, '{}.{}.train'.format(basename, i))
        with open(fname_test, 'w') as f:
            print(''.join(test), file=f)
        with open(fname_train, 'w') as f:
            print(''.join(train), file=f)
        if valid:
            with open(fname_valid, 'w') as f:
                print(''.join(valid), file=f)

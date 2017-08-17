#!/usr/bin/env python

from __future__ import print_function

import sys

from nltk.tree import Tree


if __name__ == '__main__':
    for line in sys.stdin:
        print(' '.join(Tree.fromstring(line.strip().decode('utf-8')).leaves()).encode('utf-8'))

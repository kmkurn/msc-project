#!/usr/bin/env python

from argparse import ArgumentParser

from nltk.tree import Tree


if __name__ == '__main__':
    parser = ArgumentParser(
        description=('Find the percentage of words in training data that '
                     'also exist in pretrained embedding'))
    parser.add_argument('train', help='path to training data, one parsed sentence per line')
    parser.add_argument('pretrained', help='path to pretrained embedding file')
    args = parser.parse_args()

    train_words = set()
    with open(args.train) as f:
        for line in f:
            t = Tree.fromstring(line.strip())
            for word in t.leaves():
                train_words.add(word)

    pretrained_words = set()
    with open(args.pretrained) as f:
        f_iter = iter(f)
        next(f_iter)  # skip first line
        for line in f_iter:
            word = line.split()[0]
            pretrained_words.add(word)

    pre_words_in_training = train_words.intersection(pretrained_words)
    pre_words_rate = len(pre_words_in_training) / len(train_words)
    print(f'Number of pretrained words: {len(pretrained_words)}')
    print(f'Number of pretrained words in training: {len(pre_words_in_training)}')
    print(f'Number of word in training: {len(train_words)}')
    print(f'Percentage of pretrained words in training: {pre_words_rate:.2%}')

#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser

from nltk.tree import Tree

from pypkg.oracle import oracle_iter, gen_oracle_iter


def is_leaf(tree):
    return not isinstance(tree, Tree)


def get_nt_labels(tree):
    if is_leaf(tree) or (len(tree) == 1 and is_leaf(tree[0])):
        return set()

    res = {tree.label()}
    for child in tree:
        res.update(get_nt_labels(child))
    return res


def get_unk_tokens(unkified):
    return {token for token in unkified.split() if token.startswith('UNK')}


def has_no_unseen(nt_labels, unk_tokens, gen_unk_tokens, parsed_line, oracle, gen_oracle):
    parsed_line = parsed_line.decode('utf-8')
    no_unseen_nt = all(label in nt_labels
                       for label in get_nt_labels(Tree.fromstring(parsed_line)))
    no_unseen_unk = all(unk in unk_tokens
                        for unk in get_unk_tokens(oracle.unkified))
    no_unseen_gen_unk = all(unk in gen_unk_tokens
                            for unk in get_unk_tokens(gen_oracle.unkified))
    return no_unseen_nt and no_unseen_unk and no_unseen_gen_unk


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Remove unseen nonterminals and UNK tokens from valid/test data')
    parser.add_argument('train_file', help='path to train txt file')
    parser.add_argument('train_oracle', help='path to discriminative train oracle file')
    parser.add_argument('train_gen_oracle', help='path to generative train oracle file')
    parser.add_argument('test_file', help='path to test txt file')
    parser.add_argument('test_oracle', help='path to discriminative test oracle file')
    parser.add_argument('test_gen_oracle', help='path to generative test oracle file')
    parser.add_argument('--save-file-to', default='test-removed.txt',
                        help='where to save the txt file output')
    parser.add_argument('--save-oracle-to', default='test-removed.oracle',
                        help='where to save the discriminative oracle file output')
    parser.add_argument('--save-gen-oracle-to', default='test-removed-gen.oracle',
                        help='where to save the generative oracle file output')
    parser.add_argument('--save-nt-to', help='where to save the list of nonterminal labels')
    parser.add_argument('--save-unk-to', help='where to save the list of UNK tokens')
    parser.add_argument('--save-gen-unk-to',
                        help='where to save the list of UNK tokens (generative)')
    args = parser.parse_args()

    with open(args.train_file) as f:
        nt_labels = set()
        for line in f:
            nt_labels.update(get_nt_labels(Tree.fromstring(line.strip().decode('utf-8'))))

    if args.save_nt_to is not None:
        with open(args.save_nt_to, 'w') as f:
            print('\n'.join(sorted(nt_labels)), file=f)

    with open(args.train_oracle) as f:
        unk_tokens = set()
        for oracle in oracle_iter(f):
            unk_tokens.update(get_unk_tokens(oracle.unkified))

    if args.save_unk_to is not None:
        with open(args.save_unk_to, 'w') as f:
            print('\n'.join(sorted(unk_tokens)), file=f)

    with open(args.train_gen_oracle) as f:
        gen_unk_tokens = set()
        for oracle in gen_oracle_iter(f):
            gen_unk_tokens.update(get_unk_tokens(oracle.unkified))

    if args.save_gen_unk_to is not None:
        with open(args.save_gen_unk_to, 'w') as f:
            print('\n'.join(sorted(gen_unk_tokens)), file=f)

    with open(args.test_file) as testf, \
            open(args.test_oracle) as testof, \
            open(args.test_gen_oracle) as testgof, \
            open(args.save_file_to, 'w') as outf, \
            open(args.save_oracle_to, 'w') as outof, \
            open(args.save_gen_oracle_to, 'w') as outgof:
        for parsed_line, oracle, gen_oracle in zip(testf, oracle_iter(testof),
                                                   gen_oracle_iter(testgof)):
            parsed_line = parsed_line.strip()
            if has_no_unseen(nt_labels, unk_tokens, gen_unk_tokens, parsed_line, oracle,
                             gen_oracle):
                print(parsed_line, file=outf)
                print(str(oracle), file=outof, end='\n\n')
                print(str(gen_oracle), file=outgof, end='\n\n')

from argparse import ArgumentParser

from nltk.tree import Tree


def is_leaf(tree):
    return not isinstance(tree, Tree)


def get_nt_labels(tree):
    if is_leaf(tree):
        return []
    yield tree.label()
    for child in tree:
        yield from get_nt_labels(child)


if __name__ == '__main__':
    parser = ArgumentParser(description='Remove unknown NT labels from valid and test data')
    parser.add_argument('train', help='Path to train data')
    parser.add_argument('valid', help='Path to valid/test data')
    parser.add_argument('--nt-dump', help='Where to save the list of nonterminal labels')
    args = parser.parse_args()

    nt_labels = set()
    for line in open(args.train):
        tree = Tree.fromstring(line)
        nt_labels.update(set(get_nt_labels(tree)))

    if args.nt_dump is not None:
        with open(args.nt_dump, 'w') as f:
            print('\n'.join(sorted(nt_labels)), file=f)

    for line in open(args.valid):
        line = line.strip()
        tree = Tree.fromstring(line)
        if all(label in nt_labels for label in get_nt_labels(tree)):
            print(line)

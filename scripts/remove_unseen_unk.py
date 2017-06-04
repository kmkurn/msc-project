from argparse import ArgumentParser
import sys


def get_unk_tokens(lines):
    k = 0
    for line in lines:
        line = line.strip()
        if k == 2:
            yield from (token for token in line.split() if token.startswith('UNK'))
        if not line:
            k = 0
        else:
            k += 1


if __name__ == '__main__':
    parser = ArgumentParser(description='Remove unseen UNK tokens from valid and test data')
    parser.add_argument('train', help='Path to train data')
    parser.add_argument('valid', help='Path to valid/test data')
    parser.add_argument('--unk-dump',
                        help='Where to save the list of UNK tokens found in train data')
    args = parser.parse_args()

    with open(args.train) as f:
        unk_tokens = set(get_unk_tokens(f))

    if args.unk_dump is not None:
        with open(args.unk_dump, 'w') as f:
            print('\n'.join(sorted(unk_tokens)), file=f)

    linum = 0
    with open(args.valid) as f:
        while True:
            linum += 1
            try:
                parsed_line = next(f).strip()
            except StopIteration:
                break
            linum += 1
            try:
                tokens_line = next(f).strip()
            except StopIteration:
                print(f'{linum}: expected a line of word tokens but found EOF instead.',
                      file=sys.err)
                sys.exit(1)
            linum += 1
            try:
                unkified_line = next(f).strip()
            except StopIteration:
                print(f'{linum}: expected a line of unkified tokens but found EOF instead.',
                      file=sys.err)
                sys.exit(1)

            no_unseen_unks = all(token in unk_tokens for token in unkified_line.split()
                                 if token.startswith('UNK'))
            if no_unseen_unks:
                print('\n'.join([parsed_line, tokens_line, unkified_line]))

            linum += 1
            try:
                line = next(f).strip()
            except StopIteration:
                print(f'{linum}: expected min. one parser action but found EOF instead.',
                      file=sys.err)
                sys.exit(1)
            while line:
                if no_unseen_unks:
                    print(line)
                linum += 1
                try:
                    line = next(f).strip()
                except StopIteration:
                    break
            print()

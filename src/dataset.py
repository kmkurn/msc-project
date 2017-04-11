from argparse import ArgumentParser
import glob
from itertools import islice
import os
import re

from torch.utils.data import Dataset

from src.utils import load_args, dump_args


class PennTreebank:
    def __init__(self, corpus_dir, which='train', version='3.0', corrected=True, merged=True,
                 max_num_sentences=None):
        """An iterable class for Penn Treebank corpus.

        Instance of this class iterates through the corpus one line at a time. A single line
        corresponds to a single parsed sentence, even though in the original corpus, one parsed
        sentence may span multiple lines.

        Args:
            corpus_dir (str): Path to Penn Treebank corpus directory.
            which (str): Which dataset to iterate. Must be one of "train", "valid", or "test".
                (default: train)
            version (str): Version of Penn Treebank. Must be one of "2.0" or "3.0".
                (default: 3.0)
            corrected (bool): Whether to use the corrected version. If False then original
                version will be used. (default: True)
            merged (bool): Whether to use the merged version (with POS tags). If False then no
                POS tag will occur in the parsed sentence. (default: True)
            max_num_sentences (int): Maximum number of sentences to iterate. If None then no
                limit will be imposed. (default: None)
        """
        if which not in ['train', 'valid', 'test']:
            raise ValueError(
                f'`which` should be one of "train", "valid", or "test". Got "{which}".')
        if version not in ['2.0', '3.0']:
            raise ValueError(f'`version` should be "2.0" or "3.0". Got "{version}".')

        if which == 'train':
            self._sections = range(2, 22)
        elif which == 'valid':
            self._sections = [24]
        else:
            self._sections = [23]

        if version == '2.0':
            corrected_dir = ''
            parsed_dir = 'combined' if merged else 'parsed'
        else:
            corrected_dir = 'corrected' if corrected else 'original'
            if merged:
                parsed_dir = os.path.join('parsed', 'mrg')
            else:
                parsed_dir = os.path.join('parsed', 'prd')

        self._ext = 'mrg' if merged else 'prd'
        self._path = os.path.join(
            corpus_dir, version, corrected_dir, parsed_dir, 'wsj')
        self._max_num_sentences = max_num_sentences
        # Matches one or more spaces between two opening or closing parentheses
        pattern = r'(?<=\() +(?=\()|(?<=\)) +(?=\))'
        self._prog = re.compile(pattern)

    def _concat_parsed_sentences(self, sentences):
        buff = []
        bracket_cnt = 0
        for sent in sentences:
            s = sent.strip()
            buff.append(s)
            for c in s:
                if c == '(':
                    bracket_cnt += 1
                elif c == ')':
                    bracket_cnt -= 1
            if bracket_cnt == 0:
                yield ' '.join(buff)
                buff = []
                bracket_cnt = 0

    def _squeeze_sentence(self, sentence):
        # Remove excess spaces and extra enclosing parenthesis
        return self._prog.sub('', sentence)[1:-1]

    def _get_iterator(self):
        for sec in self._sections:
            glob_pattern = os.path.join(self._path, f'{sec:02}', f'*.{self._ext}')
            for filename in sorted(glob.glob(glob_pattern)):
                with open(filename) as f:
                    lines = (line.rstrip() for line in f if line.rstrip())
                    yield from (self._squeeze_sentence(sent)
                                for sent in self._concat_parsed_sentences(lines))

    def __iter__(self):
        return islice(self._get_iterator(), self._max_num_sentences)


class PennTreebankDataset(Dataset):
    def __init__(self, ptb_iter):
        """Penn Treebank dataset.

        This is meant to be used with `torch.utils.data.DataLoader` class.

        Args:
            ptb_iter: Iterator for or iterable of Penn Treebank corpus. Can be obtained
                by creating instance of `PennTreebank` class.
        """
        self._dataset = list(ptb_iter)

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


if __name__ == '__main__':
    parser = ArgumentParser(description='Fetch sentences from PTB dataset.')
    parser.add_argument('--corpus-dir', required=True, help='path to corpus directory')
    parser.add_argument('--which', choices=['train', 'valid', 'test'], default='train',
                        help='which dataset (train/valid/test) to fetch')
    parser.add_argument('--version', choices=['2.0', '3.0'], default='3.0', help='PTB version')
    parser.add_argument('--no-corrected', action='store_false', dest='corrected',
                        help='use non-corrected (original) version')
    parser.add_argument('--no-merged', action='store_false', dest='merged',
                        help='use non-merged (without POS tags) version')
    parser.add_argument('-n', '--max-sentences', type=int, default=None,
                        help='max number of sentences')
    parser.add_argument('--dump-args', help='where to dump script arguments')
    parser.add_argument('--load-args', help='load script arguments from this file')
    args = parser.parse_args()

    if args.dump_args is not None:
        attrs = ['which', 'version', 'corrected', 'merged', 'max_sentences']
        dump_args(args, attrs, args.dump_args)

    if args.load_args is not None:
        load_args(args, args.load_args,
                  typecast=dict(max_sentences=lambda x: None if x == 'None' else int(x)))

    ptb = PennTreebank(args.corpus_dir, which=args.which, version=args.version,
                       corrected=args.corrected, merged=args.merged,
                       max_num_sentences=args.max_sentences)
    for line in ptb:
        print(line)

from argparse import ArgumentParser
import glob
from itertools import islice
import os
import re

from nltk.tree import Tree
from torch.utils.data import Dataset

from src.utils import load_args, dump_args, augment_parser


class PennTreebank:
    NULL_TAG = '-NONE-'
    # Matches one or more spaces between two opening or closing parentheses
    ONE_OR_MORE_SPACES_PROG = re.compile(r'(?<=\() +(?=\()|(?<=\)) +(?=\))')
    # Matches nonterminal label with grammatical function label, e.g. NP-SBJ, PP-DIR
    # The first group only matches the nonterminal label, e.g. NP, PP
    NT_FUNC_LABEL_PROG = re.compile(r'(?<=\()(\w+)-[^ ]+(?= )')

    def __init__(self, corpus_dir, which='train', version='3.0', corrected=True, merged=True,
                 max_num_sentences=None):
        """An iterable class for Penn Treebank corpus.

        Instance of this class iterates through the corpus one line at a time. A single line
        corresponds to a single parsed sentence, even though in the original corpus, one parsed
        sentence may span multiple lines.

        Args:
            corpus_dir (str): Path to Penn Treebank corpus directory.
            which (str): Which dataset to iterate. Must be one of "train", "valid", or "test".
            version (str): Version of Penn Treebank. Must be one of "2.0" or "3.0".
            corrected (bool): Whether to use the corrected version. If `False` then original
                version will be used.
            merged (bool): Whether to use the merged version (with POS tags). If `False` then
                no POS tag will occur in the parsed sentence.
            max_num_sentences (int): Maximum number of sentences to load. If `None` then all
                sentences will be loaded.
        """
        if which not in ['train', 'valid', 'test']:
            raise ValueError(
                f'`which` should be one of "train", "valid", or "test". Got "{which}".')
        if version not in ['2.0', '3.0']:
            raise ValueError(f'`version` should be "2.0" or "3.0". Got "{version}".')

        self.corpus_dir = corpus_dir
        self.which = which
        self.version = version
        self.corrected = corrected
        self.merged = merged
        self.max_num_sentences = max_num_sentences

    @property
    def sections(self):
        if self.which == 'train':
            return range(2, 22)
        elif self.which == 'valid':
            return [24]
        else:
            return [23]

    @property
    def corrected_dir(self):
        if self.version == '2.0':
            return ''
        return 'corrected' if self.corrected else 'original'

    @property
    def parsed_dir(self):
        if self.version == '2.0':
            return 'combined' if self.merged else 'parsed'
        if self.merged:
            return os.path.join('parsed', 'mrg')
        return os.path.join('parsed', 'prd')

    @property
    def ext(self):
        return 'mrg' if self.merged else 'prd'

    def _get_iterator(self):
        path = os.path.join(self.corpus_dir, self.version, self.corrected_dir,
                            self.parsed_dir, 'wsj')
        for sec in self.sections:
            glob_pattern = os.path.join(path, f'{sec:02}', f'*.{self.ext}')
            for filename in sorted(glob.glob(glob_pattern)):
                with open(filename) as f:
                    lines = (line.rstrip() for line in f if line.rstrip())
                    yield from (self._preprocess_sentence(sent)
                                for sent in self._concat_parsed_sentences(lines))

    def _preprocess_sentence(self, sentence):
        return self._remove_empty_categories(
            self._strip_grammatical_function_label(self._squeeze_sentence(sentence)))

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
        return self.ONE_OR_MORE_SPACES_PROG.sub('', sentence)[1:-1]

    def _strip_grammatical_function_label(self, sentence):
        return self.NT_FUNC_LABEL_PROG.sub(r'\1', sentence)

    def _remove_empty_categories(self, sentence):
        t = self._remove_null_elements(Tree.fromstring(sentence))
        # Convert tree to string and squeeze it into one line
        return re.sub(r'  +', ' ', re.sub(r'\n', '', str(t)))

    @classmethod
    def _remove_null_elements(cls, tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return None if tree.label() == cls.NULL_TAG else tree
        new_children = [cls._remove_null_elements(child) for child in tree]
        new_children = [x for x in new_children if x is not None]
        if not new_children:
            return None
        return Tree(tree.label(), new_children)

    def __iter__(self):
        return islice(self._get_iterator(), self.max_num_sentences)


class IDNTreebank:
    FILENAME = 'Indonesian_Treebank.bracket'
    TWO_OR_MORE_SPACES_PROG = re.compile(r'  +')
    BRACKETED_TOKEN_PROG = re.compile(r'\(([^ ]+)\)')

    def __init__(self, corpus_dir, which='train', split_num=0, max_num_sentences=None):
        """An iterable class for IDN Treebank corpus.

        Args:
            corpus_dir (str): Path to IDN treebank corpus directory.
            which (str): Which dataset to load. Must be one of 'train', 'valid', or 'test'.
            split_num (int): Split number of the treebank to load. This assumes
                that the IDN treebank has been split for cross-validation.
            max_num_sentences (int): Maximum number of sentences to load. If `None` then all
                sentences will be loaded.
        """
        if which not in ['train', 'valid', 'test']:
            raise ValueError(
                f'`which` should be one of "train", "valid", or "test". Got "{which}".')

        self.corpus_dir = corpus_dir
        self.which = which
        self.split_num = split_num
        self.max_num_sentences = max_num_sentences

    def _get_iterator(self):
        filename = os.path.join(self.corpus_dir,
                                f'{self.FILENAME}.{self.split_num}.{self.which}')
        with open(filename) as f:
            for line in f:
                yield self._preprocess(line.strip())

    def _preprocess(self, line):
        return self.BRACKETED_TOKEN_PROG.sub(
            r'\1', self.TWO_OR_MORE_SPACES_PROG.sub(' ', line.expandtabs()))

    def __iter__(self):
        return islice(self._get_iterator(), self.max_num_sentences)


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
    parser = ArgumentParser(
        description='Load sentences from treebank dataset and print to stdout.')
    parser.add_argument('treebank', choices=['penn', 'idn'], help='treebank corpus to load')
    parser.add_argument('--corpus-dir', required=True, help='path to corpus directory')
    parser.add_argument('--which', choices=['train', 'valid', 'test'], default='train',
                        help='which dataset to load')
    parser.add_argument('-n', '--max-sentences', type=int, default=None,
                        help='max number of sentences')
    # Penn Treebank specific arguments
    parser.add_argument('--version', choices=['2.0', '3.0'], default='3.0',
                        help='corpus version (penn treebank)')
    parser.add_argument('--no-corrected', action='store_false', dest='corrected',
                        help='use non-corrected (original) version (penn treebank)')
    parser.add_argument('--no-merged', action='store_false', dest='merged',
                        help='use non-merged (without POS tags) version (penn treebank)')
    # IDN Treebank specific arguments
    parser.add_argument('--split-num', type=int, default=0, help='split number (idn treebank)')
    augment_parser(parser)
    args = parser.parse_args()

    dump_args(args, excludes=['corpus_dir'])
    load_args(args, typecast=dict(version=lambda x: x))

    if args.treebank == 'penn':
        treebank = PennTreebank(args.corpus_dir, which=args.which, version=args.version,
                                corrected=args.corrected, merged=args.merged,
                                max_num_sentences=args.max_sentences)
    else:
        treebank = IDNTreebank(args.corpus_dir, which=args.which, split_num=args.split_num,
                               max_num_sentences=args.max_sentences)

    for line in treebank:
        print(line)

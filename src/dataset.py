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

    def __init__(self, corpus_dir, which='train', version='3.0', corrected=True,
                 max_num_sentences=None):
        """An iterable class for Penn Treebank corpus.

        Instance of this class iterates through the corpus one line at a time. A single line
        corresponds to a single parsed sentence, even though in the original corpus, one parsed
        sentence may span multiple lines. Preprocessing is also performed for each parsed
        sentence:
        1) Stripping off the grammatical function labels of the nonterminals
           (e.g. NP-SBJ becomes NP)
        2) Removing null elements (i.e. all elements with -NONE- as POS tag)

        Note that this class only iterates through the combined/merged version of Penn
        Treebank, meaning that POS tags are always included.

        Args:
            corpus_dir (str): Path to Penn Treebank corpus directory.
            which (str): Which dataset to iterate. Must be one of "train", "valid", or "test".
            version (str): Version of Penn Treebank. Must be one of "2.0" or "3.0".
            corrected (bool): Whether to use the corrected version. If `False` then original
                version will be used.
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
        return 'combined' if self.version == '2.0' else os.path.join('parsed', 'mrg')

    def _get_iterator(self):
        path = os.path.join(self.corpus_dir, self.version, self.corrected_dir,
                            self.parsed_dir, 'wsj')
        for sec in self.sections:
            glob_pattern = os.path.join(path, f'{sec:02}', '*.mrg')
            for filename in sorted(glob.glob(glob_pattern)):
                with open(filename) as f:
                    lines = (line.rstrip() for line in f if line.rstrip())
                    yield from (self._preprocess_sentence(sent)
                                for sent in self._concat_parsed_sentences(lines))

    @classmethod
    def _preprocess_sentence(cls, sentence):
        t = Tree.fromstring(sentence, remove_empty_top_bracketing=True)
        t = cls._remove_null_elements(t)
        t = cls._strip_function_labels(t)
        return cls._squeeze_line(str(t))

    @staticmethod
    def _concat_parsed_sentences(sentences):
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

    @staticmethod
    def _squeeze_line(line):
        return re.sub(r'  +', ' ', line.strip().replace('\n', ''))

    @classmethod
    def _strip_function_labels(cls, tree):
        if cls._is_leaf(tree):
            return tree

        label = tree.label()
        ixs = [label.find(c) for c in '-=|' if label.find(c) >= 0]
        ix = min(ixs) if ixs else 0
        new_label = label[:ix] if ix > 0 else label
        return Tree(new_label, [cls._strip_function_labels(child) for child in tree])

    @classmethod
    def _remove_null_elements(cls, tree):
        if len(tree) == 1 and cls._is_leaf(tree[0]):
            return None if tree.label() == cls.NULL_TAG else tree

        new_children = []
        for child in tree:
            new_child = cls._remove_null_elements(child)
            if new_child is not None:
                new_children.append(new_child)
        return Tree(tree.label(), new_children) if new_children else None

    @staticmethod
    def _is_leaf(tree):
        return not isinstance(tree, Tree)

    def __iter__(self):
        return islice(self._get_iterator(), self.max_num_sentences)


class IDNTreebank:
    FILENAME = 'Indonesian_Treebank.bracket'
    NULL_ELEMS = ['*T*', '0', '*U*', '*?*', '*NOT*', '*RNR*', '*ICH*', '*EXP*', '*PPA*']
    WORD_UNIT_PROG = re.compile(r'\(([^\(\)]+)\)')

    def __init__(self, corpus_dir, which='train', split_num=0, max_num_sentences=None):
        """An iterable class for IDN Treebank corpus.

        An instance of this class will iterate through all the parsed sentences one by one,
        even though in the original dataset, one line might contain more than one parsed
        sentences. Preprocessing is performed for each parsed sentence:
        1) Stripping off the grammatical function labels of the nonterminals
           (e.g. NP-SBJ becomes NP)
        2) Removing null elements (e.g. *T*, *U*, etc; full list in section 4 Penn Treebank
           annotation guideline)
        3) Joining multiple words under the same nonterminal label, delimited by underscore
           (e.g. "(NN (kunjungan kerja))" becomes "(NN kunjungan_kerja)")

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
                yield from (self._preprocess_sentence(sent)
                            for sent in self._get_parsed_sentences(line))

    @staticmethod
    def _get_parsed_sentences(line):
        i = 0
        while i < len(line) and line[i] != '(':
            i += 1

        while i < len(line):
            j = i
            bracket_cnt = 0
            while j < len(line):
                if line[j] == '(':
                    bracket_cnt += 1
                elif line[j] == ')':
                    bracket_cnt -= 1
                j += 1
                if bracket_cnt == 0:
                    yield line[i:j]
                    break
            i = j
            while i < len(line) and line[i] != '(':
                i += 1

    @classmethod
    def _preprocess_sentence(cls, sentence):
        s = cls._squeeze_line(sentence.replace('\t', ''))
        t = cls._strip_function_labels(cls._to_tree(s))
        t = cls._remove_null_elements(t)
        return cls._squeeze_line(str(t))

    @classmethod
    def _to_tree(cls, line):
        line = cls.WORD_UNIT_PROG.sub(r'[\1]', line)
        tree = Tree.fromstring(line)
        return cls._combine_multiword(tree)

    @classmethod
    def _combine_multiword(cls, tree):
        if cls._is_leaf(tree):
            return tree[1:-1] if tree[0] == '[' and tree[-1] == ']' else tree

        if len(tree) > 1 and all(cls._is_leaf(child) for child in tree):
            assert tree[0][0] == '[' and tree[-1][-1] == ']'
            new_child = '_'.join(tree)[1:-1]
            return Tree(tree.label(), [new_child])

        return Tree(tree.label(), [cls._combine_multiword(child) for child in tree])

    @staticmethod
    def _squeeze_line(line):
        return re.sub(r'  +', ' ', line.strip().replace('\n', ''))

    @classmethod
    def _strip_function_labels(cls, tree):
        if cls._is_leaf(tree):
            return tree

        label = tree.label()
        ix = label.find('-')
        new_label = label[:ix] if ix > 0 else label
        return Tree(new_label, [cls._strip_function_labels(child) for child in tree])

    @classmethod
    def _remove_null_elements(cls, tree):
        if cls._is_leaf(tree):
            return None if cls._get_head_label(tree) in cls.NULL_ELEMS else tree

        if (cls._get_head_label(tree.label()) == 'NP' and len(tree) == 1 and
                cls._is_leaf(tree[0]) and cls._get_head_label(tree[0]) == '*'):
            return None

        new_children = []
        for child in tree:
            new_child = cls._remove_null_elements(child)
            if new_child is not None:
                new_children.append(new_child)
        return Tree(tree.label(), new_children) if new_children else None

    @staticmethod
    def _is_leaf(tree):
        return not isinstance(tree, Tree)

    @staticmethod
    def _get_head_label(label):
        ix = label.find('-')
        return label[:ix] if ix > 0 else label

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
    # IDN Treebank specific arguments
    parser.add_argument('--split-num', type=int, default=0, help='split number (idn treebank)')
    augment_parser(parser)
    args = parser.parse_args()

    dump_args(args, excludes=['corpus_dir'])
    load_args(args, typecast=dict(version=lambda x: x))

    if args.treebank == 'penn':
        treebank = PennTreebank(args.corpus_dir, which=args.which, version=args.version,
                                corrected=args.corrected, max_num_sentences=args.max_sentences)
    else:
        treebank = IDNTreebank(args.corpus_dir, which=args.which, split_num=args.split_num,
                               max_num_sentences=args.max_sentences)

    for line in treebank:
        print(line)

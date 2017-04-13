from configparser import ConfigParser
import subprocess
import sys


ARGUMENTS_SECTION = 'ARGUMENTS'
META_SECTION = 'META'


def dump_args(args, path=None, excludes=None, override_excludes=False):
    """Dump a given script argument namespace.

    The namespace will be dumped to a config INI file. By default, all attributes
    of `args` that does not start with an underscore will be dumped, except for
    attributes named `'dump_args'` and `'load_args'`.

    Args:
        args: Script argument namespace to dump. Usually this just what
            `parse_args()` returns.
        path (str): Where the dumped arguments should be saved. If None then
            the path is assumed to be stored in `args.dump_args`.
            (default: None)
        excludes: Iterable of additional attribute names to exclude from dumping.
            (default: None)
        override_excludes (bool): Whether to override the excludes list instead of appending
            to the default one. (default: False)
    """
    default_excludes = ['dump_args', 'load_args']
    excludes = default_excludes + ([] if excludes is None else excludes)
    if override_excludes:
        excludes = excludes[2:]
    if path is None:
        path = args.dump_args

    if path is not None:
        config = ConfigParser()
        # Arguments section
        config[ARGUMENTS_SECTION] = {}
        attrs = [attr for attr in dir(args) if attr[0] != '_' and attr not in excludes]
        for attr in attrs:
            config[ARGUMENTS_SECTION][attr] = str(getattr(args, attr))
        # Meta section
        config[META_SECTION] = {
            'script_path': sys.argv[0],
            'commit_hash': _get_last_commit_hash()
        }

        with open(path, 'w') as f:
            config.write(f)


def _get_last_commit_hash():
    return subprocess.run(
        'git log -n 1 --pretty=format:%H'.split(), stdout=subprocess.PIPE,
        encoding='UTF-8').stdout


def load_args(obj, path=None, typecast=None):
    """Load script arguments from a given file.

    The loaded file should be a valid config INI file.

    Args:
        obj: Target argument namespace object.
        path (str): Path to config file to load. If None then the path will be
            read from `obj.load_args`. (default: None)
        typecast (dict): A dictionary whose keys are attribute names and
            values are unary functions. The function will receive the loaded
            value and should return the new value for the attribute.
            (default: None)
    """
    if path is None:
        path = obj.load_args
    if typecast is None:
        typecast = {}

    if path is not None:
        config = ConfigParser()
        config.read(path)
        for key in config[ARGUMENTS_SECTION]:
            val = config[ARGUMENTS_SECTION][key]
            val = typecast[key](val) if key in typecast else _default_typecast(val)
            setattr(obj, key, val)


def _default_typecast(val):
    for new_val in [None, True, False]:
        if str(new_val) == val:
            return new_val

    try:
        new_val = int(val)
    except ValueError:
        try:
            new_val = float(val)
        except ValueError:
            new_val = val

    return new_val


def augment_parser(parser):
    """Add default arguments regarding script args to parser.

    Args:
        parser: An instance of `argparse.ArgumentParser` to which the default
            arguments will be added.
    """
    parser.add_argument('--dump-args', help='where to dump script arguments')
    parser.add_argument('--load-args', help='load script arguments from this file')

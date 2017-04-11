from configparser import ConfigParser


def dump_args(args, attrs, path, section='DEFAULT'):
    """Dump a given script argument namespace.

    The namespace will be dumped to a config INI file.

    Args:
        args: Script argument namespace to dump. Usually this just what
            `parse_args()` returns.
        attrs (iterable): Iterable of attribute names to dump.
        path (str): Where the dumped arguments should be saved.
        section (str): Section name in the dump file. (default: DEFAULT)
    """
    config = ConfigParser()
    config[section] = {}
    for attr in attrs:
        config[section][attr] = str(getattr(args, attr))
    with open(path, 'w') as f:
        config.write(f)


def load_args(obj, path, section='DEFAULT', typecast=None):
    """Load script arguments from a given file.

    The loaded file should be a valid config INI file.

    Args:
        obj: Target argument namespace object.
        path (str): Path to config file to load.
        section (str): Section name to load. (default: DEFAULT)
        typecast (dict): A dictionary whose keys are attribute names and
            values are unary functions. The function will receive the loaded
            value and should return the new value for the attribute.
            (default: None)
    """
    if typecast is None:
        typecast = {}

    config = ConfigParser()
    config.read(path)
    for key in config[section]:
        val = config[section][key]
        val = typecast[key](val) if key in typecast else val
        setattr(obj, key, val)

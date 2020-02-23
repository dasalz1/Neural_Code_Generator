import glob, os, re

CODE_TYPE = ".py"

def preprocess_source(source):
    """
    Remove # in strings for comment_ptr.

    TODO. Later on, fix regex instead.
    """
    for string in re.findall(r'".*?#.*?"|\'.*?#.*?\'', source):
        if '#' in string:
            modified = string.replace('#', '')
            source = source.replace(string, modified)
    return source


def read_data(path):
    if not os.path.exists(path):
        print("[!] Data does not exist")
    elif os.path.isfile(path):
        return read_file(path)
    else:
        return read_dir(path)


def read_file(path):
    num_file = 1
    sources = []
    try:
        with open(path, 'r') as f:
            source.append(preprocess_source(f.read()))
    except Exception as e:
        print(e)
    return sources, num_file


def read_dir(path):
    num_file = 0
    sources = []
    for filename in glob.iglob(os.path.join(path, "**/*.py"), recursive=True):
        if not filename.endswith(CODE_TYPE): continue
        num_file += 1
        with open(filename, "r") as f:
            sources.append(preprocess_source(f.read()))
    return sources, num_file
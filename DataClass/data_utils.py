import glob, os, re

CODE_TYPE = ".py"

# Regex
newline_ptr = re.compile(r'(?:"[^"]*"|.)+')  # \n outside of quotes (https://stackoverflow.com/questions/24018577/parsing-a-string-in-python-how-to-split-newlines-while-ignoring-newline-inside)
comment_ptr = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|#[^"\']*?(?=\n)|#.*?(".*?"|\'.*?\').*?(?=\n)', re.DOTALL|re.MULTILINE)
literal_ptr = re.compile(r'".*?"|\'.*?\'|[-+]?\d*\.\d+|\d+')
camelcase_ptr = re.compile(r"(?<=[a-z])([A-Z]+[a-z]*)")
number_ptr = re.compile(r'(?<=[^a-zA-Z])([-+]?\d*\.\d+|\d+)')
number_with_alpha_ptr = re.compile(r'(?<=[a-zA-Z])([-+]?\d*\.\d+|\d+)')  # split numbers from alpha
string_ptr = re.compile(r'".*?"|\'.*?\'')
code_ptr = re.compile(r"([^a-zA-Z0-9])")
whitespace_ptr = re.compile(r"(\s+)")


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
    # filenames = []
    for filename in glob.iglob(os.path.join(path, "**/*.py"), recursive=True):
        if not filename.endswith(CODE_TYPE): continue
        num_file += 1
        with open(filename, "r") as f:
            sources.append(preprocess_source(f.read()))

        # filenames.append(filename)

    return sources


###############################################################################
# Tokenize
###############################################################################

def tokenize(s,
             split_camelcase,
             split_number_from_alpha,
             keep_literal,
             keep_whitespace,
             verbose=False):

    numbers, strings, delimiters = [], [], []

    if keep_literal:
        numbers = remove_null(set(re.findall(number_ptr, s)))
        strings = remove_null(set(re.findall(string_ptr, s)))

        literals = [f"(?<=[^a-zA-Z0-9]){re.escape(l)}|^{re.escape(l)}" for l in numbers]  # Add negative look ahead to exclude cases like fc1
        literals.extend([re.escape(l) for l in strings])

        delimiters = sorted(literals, key=len, reverse=True)

    # Basic tokenization based on non alphanumeric tokens
    delimiters.append("[^a-zA-Z0-9]")  # Be careful of the order
    delimiters = remove_null(delimiters)

    tmp_code_ptr = "({})".format("|".join(delimiters))
    tokens = remove_null(re.split(tmp_code_ptr, s))
    if verbose:
        print('[tokenize] Basic:', tokens)

    if split_camelcase:
        before = tokens
        tokens = []
        for token in before:
            if not token:
                continue
            elif token in numbers or token in strings:
                tokens.append(token)
            else:
                tokens.extend(re.split(camelcase_ptr, token))
        tokens = remove_null(tokens)
        if verbose:
            print('[tokenize] Split camel cases:', tokens)

    if split_number_from_alpha:
        before = tokens
        tokens = []
        for token in before:
            if not token:
                continue
            elif token in numbers or token in strings:
                tokens.append(token)
            else:
                tokens.extend(re.split(number_with_alpha_ptr, token))
        tokens = remove_null(tokens)
        if verbose:
            print('[tokenize] Split numbers from alpha:', tokens)

    if not keep_whitespace:
        tokens = [token for token in tokens if len(token.strip()) > 0]
        if verbose:
            print('[tokenize] Remove whitespace:', tokens)

    return tokens


def tokenize_fine_grained(s, keep_whitespace=False):
    """
    Tokenize as much as possible. Used when calculating edit distance.

    E.g., camelCase45 = "hi there" -> camel, Case, 45, ", hi, there, "
    """
    return tokenize(s,
                    split_camelcase=True,
                    split_number_from_alpha=True,
                    keep_literal=False,
                    keep_whitespace=keep_whitespace)


def tokenize_keywords(s):
    """
    Tokenize as much as human-preferable. Used when generating keywords.

    Do not tokenize based on literals.
    By default, whitespace is entirely removed.
    """
    return tokenize(s,
                    split_camelcase=True,
                    split_number_from_alpha=True,
                    keep_literal=True,
                    keep_whitespace=False)


def rand_select(list, k):
    print(f"Randomly selected {k} examples from {len(list)} examples:")
    return random.choices(list, k=k)


def strip_empty_lines(s):
    """
    Remove empty lines at first and last.
    """
    lines = s.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def split_newlines(s):
    """
    Split based on new lines (\n) outside of quotes.

    Note that this coalesces several newlines into one,
    as blank lines are ignored. To avoid that, give a null case:

    (?:"[^"]*"|.)+|(?!\Z)
    """
    return re.findall(newline_ptr, s)


def remove_null(l):
    return list(filter(None, l))


def remove_comments(source):
    return re.sub(r"\n\n+", "\n\n", re.sub(comment_ptr, "", source))


def remove_redundant_indentation(code):
    lines = split_newlines(code)
    redundant_indentation = min([len(line) - len(line.lstrip())
                                 for line in lines
                                 if len(line.strip()) > 0])
    lines = [line[redundant_indentation:] for line in lines]
    return lines


def to_string_opt(opt):
    """
    Generate a string for filename that contains current options

    E.g. u_line__d_metric_n__d_thre_0.5__n_1__max_can_5
    """
    s = []
    s.append(f'u_{opt.unit}')
    s.append(f'd_metric_x_{opt.distance_metric_x}')
    s.append(f'd_thre_x_{opt.distance_threshold_x}')
    s.append(f'd_metric_y_{opt.distance_metric_y}')
    s.append(f'd_thre_y_{opt.distance_threshold_y}')
    s.append(f'n_{opt.n_leftmost_tokens}')
    s.append(f'max_can_{opt.max_num_candidates}')
    return '__'.join(s)


def get_lines_from_source(source,
                          remove_comments_from_source,
                          remove_empty_lines_from_source):
    """
    Remove comments and empty lines from source.
    Return a list of lines
    """
    if remove_comments_from_source:
        source = remove_comments(source)

    lines = split_newlines(source)

    if remove_empty_lines_from_source:
        lines = [line for line in lines if len(line.strip()) > 0]
    return lines
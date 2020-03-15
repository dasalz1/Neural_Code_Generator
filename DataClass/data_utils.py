import glob, os, re, csv, logging, pickle
from string import Template
from DataClass.Constants import PAD_WORD, START_WORD, END_WORD, PAD_IDX, START_IDX, END_IDX, NO_CONTEXT_IDX, NO_CONTEXT_WORD, UNKNOWN_IDX, UNKNOWN_WORD, SEP_CONTEXT_WORD, SEP_CONTEXT_IDX, SEP_PAIR_WORD, SEP_PAIR_IDX, SEP_RET_WORD, SEP_RET_IDX

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


def preprocess_tokens(tokens, max_dim):
    tokens = [START_WORD] + tokens
    n = len(tokens) + 1
    # minus one since end word needs to go on too
    tokens = tokens[:min(n, max_dim-1)] + [END_WORD] + [PAD_WORD]*max(0, max_dim-n)
    return tokens

def preprocess_context(context, n, max_dim):
    context_tokens = preprocess_tokens(tokenize_fine_grained(context[0, 0]), max_dim)
    context_tokens += [SEP_CONTEXT_WORD]
    
    for idx in range(n-1):
        context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, idx*2+2]), max_dim) + [SEP_PAIR_WORD]
        context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, idx*2+3]), max_dim) + [SEP_RET_WORD]
    
    context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, -2]), max_dim) + [SEP_PAIR_WORD]
    context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, -1]), max_dim)
    return context_tokens

def check_quote_str(line):
    if "\'" in line:
        return line.replace('"', "'")
    else:
        return line.replace('""', '"').replace("'", '"')

def fix_quote_strings(v):
    idx = v.find('","')
    x, y = v[1:idx], v[idx+3:-1]
    x = check_quote_str(x)
    y = check_quote_str(y)
    
    return [[x, y]]

def fix_quote_strings_context(v, n):
    idx = v.find('","')
    x = check_quote_str(v[1:idx])

    v = v[idx+3:]; idx = v.find('","')
    y = check_quote_str(v[:idx])
    values = [x, y]
    v = v[idx+3:]
    for _ in range(n*2 - 1):
        idx = v.find('","')
        values.append(check_quote_str(v[:idx]))
        v = v[idx+3:]

    values.append(check_quote_str(v[:-1]))
    return [values]

def create_vocab_dictionary(path, file, save=False, filter_count=4500):
    tokens_dict = pickle.load(open(path+'/'+file, 'rb'))
    idx2word = {}
    word2idx = {}

    if PAD_WORD in tokens_dict: del tokens_dict[PAD_WORD]
    if NO_CONTEXT_WORD in tokens_dict: del tokens_dict[NO_CONTEXT_WORD]
    if START_WORD in tokens_dict: del tokens_dict[START_WORD]
    if END_WORD in tokens_dict: del tokens_dict[END_WORD]
    if UNKNOWN_WORD in tokens_dict: del tokens_dict[UNKNOWN_WORD]
    
    word2idx[PAD_WORD] = PAD_IDX
    word2idx[END_WORD] = END_IDX
    word2idx[START_WORD] = START_IDX
    word2idx[NO_CONTEXT_WORD] = NO_CONTEXT_IDX
    word2idx[UNKNOWN_WORD] = UNKNOWN_IDX
    word2idx[SEP_CONTEXT_WORD] = SEP_CONTEXT_IDX
    word2idx[SEP_PAIR_WORD] = SEP_PAIR_IDX
    word2idx[SEP_RET_WORD] = SEP_RET_IDX
    idx2word[PAD_IDX] = PAD_WORD
    idx2word[END_IDX] = END_WORD
    idx2word[START_IDX] = START_WORD
    idx2word[NO_CONTEXT_IDX] = NO_CONTEXT_WORD
    idx2word[UNKNOWN_IDX] = UNKNOWN_WORD
    idx2word[SEP_CONTEXT_IDX] = SEP_CONTEXT_WORD
    idx2word[SEP_PAIR_IDX] = SEP_PAIR_WORD
    idx2word[SEP_RET_IDX] = SEP_RET_WORD

    embedding_idx = SEP_RET_IDX+1
    for word, count in tokens_dict.items():
        if count < filter_count: continue
        word2idx[word] = embedding_idx
        idx2word[embedding_idx] = word
        embedding_idx+=1


    if save:
        pickle.dump(word2idx, open(path+'/word2idx_tokens.pickle', 'wb'))
        pickle.dump(idx2word, open(path+'/idx2word_tokens.pickle', 'wb'))

    return word2idx, idx2word

    # {word:idx for idx, word in enumerate(tokens_dict)}


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

def get_urls_from_github(num_repos):
    urls = []
    page = 1
    while len(urls) < num_repos:
        url = "https://api.github.com/search/repositories?q=tensorflow+" \
             + "language:python&sort=stars&order=desc&per_page=100&page={}".format(page)
        response = requests.get(url)
        repos = response.json()
        urls.extend([repo["clone_url"] for repo in repos["items"]])
        page += 1

    urls = urls[:num_repos]
    print(f"[.] Collected {len(urls)} URLs.")
    return urls


def get_urls_from_csv(path_csv):
    url_template = Template('https://github.com/$url.git')
    urls = []
    with open(path_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                pass
            else:
                urls.append(url_template.substitute(url=row[0]))
    print(f'Read {len(urls)} urls from csv file')
    return urls


# Logger
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter((logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(log_path)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

# Metric

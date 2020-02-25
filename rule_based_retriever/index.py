import os
import re
import glob
import pickle
import random
import distance
import collections
import configargparse
import pathlib
from tqdm import tqdm

from rule_based_retriever import opts

from flask import Flask, render_template, jsonify, request  # noqa
from flask_cors import CORS


# Basic Flask setup
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'fe093b0354f9ba0b1237a5e36f58caf55cc9f5682c2627b7463c30f0bbd97672'  # noqa
opt = opts.get_params(path = str(pathlib.Path.cwd()  / 'github_data' / 'tensorflow-vgg'))


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




###############################################################################
# Read data
###############################################################################

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
    print(path)
    if not os.path.exists(path):
        print("[!] Data does not exist")
        print(path)
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
        num_file += 1
        with open(filename, "r") as f:
            sources.append(preprocess_source(f.read()))
    return sources, num_file


def read_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_as_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


###############################################################################
# Utils
###############################################################################

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


def get_prefix(y,
               n=opt.n_leftmost_tokens):
    return tuple(tokenize_keywords(y)[:n])


def get_suffix(x,
               n=opt.n_leftmost_tokens):
    return tuple(tokenize_keywords(x)[-n:])


###############################################################################
# Print stuff in color
###############################################################################

def print_example(example):
    x, y = example
    print(colored(x, 'red') + colored(y, 'blue'))


def print_examples(examples):
    print("-------------------------------")
    for example in examples:
        print_example(example)
        print("-------------------------------")


def print_candidates(example, candidates):
    print('[x] -------------------------------')
    print(colored(example[0], 'red') + colored(example[1], 'blue'))
    for i, (edit_distance, x, y) in enumerate(candidates, 1):
        print(f"[{i}] {edit_distance:.2f} --------------------------")
        print_example((x, y))


def print_example_edit(example_edit):
    x, y_abs, x_prime, y_prime = example_edit
    print_example((x_prime, y_prime))
    print(colored('-------------------------------', 'white'))
    print_example((x, y_abs))


def print_dataset_edit(dataset_edit):
    for i, example_edit in enumerate(dataset_edit, 1):
        print(f"[{i}] -------------------------------")
        print_example_edit(example_edit)


def plot_histogram(array, title="", xlabel="", ylabel=""):
    n, bins, patches = plt.hist(array, bins=30, facecolor='g', alpha=0.75)
    plt.grid(True)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


###############################################################################
# Extract (x, y) pairs
###############################################################################

def process_example(x, y, n=opt.n_leftmost_tokens):
    """
    Cut and paste the prefix of y at the end of x
    """
    prefix_tokens = get_prefix(y, n=n)
    tmp_prefix_ptr = '\s*' + '\s*'.join([re.escape(token) for token in prefix_tokens])
    prefix_index = re.search(tmp_prefix_ptr, y).end()
    x += '\n' + y[:prefix_index]  # Valid only if each unit is separated by lines
    y = y[prefix_index:]
    return x, y


def reverse_process_example(x, y, n=opt.n_leftmost_tokens):
    """
    Cut and paste the suffix of x at the beginning of y
    """
    suffix_tokens = get_suffix(x, n=n)
    tmp_suffix_ptr = '\s*' + '\s*'.join([re.escape(token) for token in suffix_tokens]) + '$'
    prefix_index = re.search(tmp_suffix_ptr, x).start()
    y = strip_empty_lines(x[prefix_index:] + y)
    x = x[:prefix_index]
    return x, y


def extract_examples_line_by_line(source,
                                  n=opt.n_leftmost_tokens,
                                  min_len=opt.min_len):  # If line, no need to filter based on length
    examples = []
    def add_example(x, y):
        x, y = process_example(x, y, n=n)
        examples.append((x, y))

    lines = get_lines_from_source(source,
                                  remove_comments_from_source=True,
                                  remove_empty_lines_from_source=True)
    if not lines:
        return []

    add_example(x="", y=lines[0])  # First example doesn't have context
    for i in range(1, len(lines) - 1):
        add_example(x=lines[i], y=lines[i + 1])

    # Filter out examples that are too short
    examples = [(x, y) for (x, y) in examples if len(x.strip()) > min_len and len(y.strip()) > min_len]

    # Filter out examples that are import stmts
    examples = [(x, y) for (x, y) in examples if 'import' not in x and 'import' not in y]
    return examples


###############################################################################
# Calculate distance
###############################################################################

def abstract(s):
    return re.sub(literal_ptr, ' ', s)


def has_alpha(tokens):
    return any([token.isalpha() for token in tokens])


def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t

        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost) # substitution
    return dist


def backtrack_levenshtein(tokens1, tokens2, dist, verbose=False):
    """
    Edit tokens2 to tokens1
    """
    i = len(tokens1)
    j = len(tokens2)

    replaced_pairs = []  # list of (token2, token1) pairs
    replaced_indices = [] # list of (index2, index1) pairs

#     if verbose:
#         for row in dist:
#             print(row)
#         print(tokens1)
#         print(tokens2)

    while i > 0 or j > 0:
        if j <= 0:
            if verbose:
                print("Insert", tokens1[i - 1])
            i -= 1
            continue
        if i <= 0:
            if verbose:
                print("Delete", tokens2[j - 1])
            j -= 1
            continue

        if tokens1[i - 1] == tokens2[j - 1]:
            if verbose:
                print(f"Same {tokens1[i - 1]} (i={i - 1}, j={j - 1})")
            i -= 1
            j -= 1
        elif j > 0 and dist[i][j] == dist[i][j - 1] + 1:
            if verbose:
                print("Delete", tokens2[j - 1])
            j -= 1
        elif i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + 1:
            if verbose:
                print(f"Replace {tokens2[j - 1]} with {tokens1[i - 1]}")
            replaced_pairs.append((tokens2[j - 1], tokens1[i - 1]))
            replaced_indices.append((j - 1, i - 1))
            i -= 1
            j -= 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            if verbose:
                print("Insert", tokens1[i - 1])
            i -= 1
        else:
            if verbose:
                print(f"Error: i={i}, j={j}")
            break


    return replaced_pairs, replaced_indices


def collapse_edit_distance(tokens1, tokens2, verbose=False):
    dist = iterative_levenshtein(tokens1, tokens2)
    replaced_pairs, replaced_indices = backtrack_levenshtein(tokens1, tokens2, dist, verbose=verbose)
    if verbose:
        print(replaced_pairs)

    edit_distance = dist[-1][-1]
    collapse = len(replaced_pairs) - len(set(replaced_pairs))  # Do not count same replacement multiple times
    return edit_distance - collapse


def calculate_edit_distance(code_block1,
                            code_block2,
                            ignore_literals,
                            distance_metric,
                            verbose=False):
    if ignore_literals:  # Todo. Just ignore difference in strings if they are substentially different
        block1 = abstract(code_block1)
        block2 = abstract(code_block2)
        if verbose:
            print("[.] Abstracted code blocks:")
            print(block1.strip())
            print(block2.strip())

    else:
        block1 = code_block1
        block2 = code_block2

    # Tokenize
    tokens1 = tokenize_fine_grained(block1, keep_whitespace=False)
    tokens2 = tokenize_fine_grained(block2, keep_whitespace=False)

    if not tokens1 or not tokens2:
        return float('inf')

    if not has_alpha(tokens1) or not has_alpha(tokens2):
        return float('inf')

    if verbose:
        print(tokens1)
        print(tokens2)

    # https://github.com/doukremt/distance
    if distance_metric == "j":
        return distance.jaccard(tokens1, tokens2)
    elif distance_metric == "l":
        return distance.levenshtein(tokens1, tokens2)
    elif distance_metric == "h":
        return distance.hamming(tokens1, tokens2)
    elif distance_metric == "s":
        return distance.sorensen(tokens1, tokens2)
    elif distance_metric == "n":  # Normalized Levenshtein
        return distance.nlevenshtein(tokens1, tokens2)
    elif distance_metric == "c":  # Collapsed Levenshtein edit distance
        return collapse_edit_distance(tokens1, tokens2, verbose=verbose)
    elif distance_metric == "nc":  # Normalized collapsed Levenshtein edit distance
        collapsed = collapse_edit_distance(tokens1, tokens2, verbose=verbose)
        return collapsed / max(len(tokens1), len(tokens2))


def replace_diff_with_placeholders(string1, string2):
    """
    Replace string2-specific tokens with placeholders. Keep original values.

    E.g. string1 = 'self.fc7 = (self.relu7, 4096, 4096, "fc7")'
         string2 = 'self.fc8 = (self.relu8, 4096, 1000, "fc8")'

         return 'self.fc[[8]] = (self.relu[[8]], 4096, [[1000]], "fc[[8]]")'

    Parameters:
        string1 (str): y
        string2 (str): y'; to be abstracted to be consistent with y
    """
    tokens1 = tokenize_fine_grained(string1, keep_whitespace=True)  # Need to keep whitespace
    tokens2 = tokenize_fine_grained(string2, keep_whitespace=True)  # Need to keep whitespace

    dist = iterative_levenshtein(tokens1, tokens2)
    replaced_pairs, replaced_indices = backtrack_levenshtein(tokens1, tokens2, dist)

    replaced = tokens2
    for index2, index1 in replaced_indices:
        replaced[index2] = f'[[{replaced[index2]}]]'
    return ''.join(replaced)


def rank_based_on_distance(opt,
                           example,
                           examples,
                           include_unsatisfying_examples,
                           exclude_same_context,
                           exclude_same_example,
                           n=opt.n_leftmost_tokens,
                           check_exact_match_suffix=opt.check_exact_match_suffix,
                           verbose=False):
    """
    Parameters:
        check_exact_match_suffix (bool): make sure that the suffix of x
            (n_leftmost_tokens tokens) exactly matches
        exclude_unsatisfying_examples (bool): filter out examples whose
            edit distance is larger than distance_threshold
        exclude_same_context (bool): exclude examples whose context exactly matches x
    """
    ranked_examples = []
    x, y = example
    if verbose:
        print("[..] x:", x)

    candidate_examples = examples

    # Select examples whose suffix of x exactly matches with that of example
    if check_exact_match_suffix:
        candidate_examples = []
        x_suffix = get_suffix(x, n)
        if verbose:
            print(f"[.] Enforce to have same {n} tokens as suffix: {x_suffix}")
        for x_prime, y_prime in examples:
            x_prime_suffix = get_suffix(x_prime, n)
            if verbose:
                print("[..] x':", x_prime)
                print("[..] Rightmost tokens:", x_prime_suffix, "\n")
            if x_suffix == x_prime_suffix:
                candidate_examples.append((x_prime, y_prime))

    if verbose:
        print(f"[.] Ranking {len(candidate_examples)} examples")
    for x_prime, y_prime in candidate_examples:
        edit_distance_x = calculate_edit_distance(x,
                                                  x_prime,
                                                  distance_metric=opt.distance_metric_x,
                                                  ignore_literals=False)
        edit_distance_y = calculate_edit_distance(y,
                                                  y_prime,
                                                  distance_metric=opt.distance_metric_y,
                                                  ignore_literals=False)

        if include_unsatisfying_examples:  # Include all examples
            ranked_examples.append((edit_distance, x_prime, y_prime))
        elif edit_distance_x <= opt.distance_threshold_x and edit_distance_y <= opt.distance_threshold_y:
            if exclude_same_context and edit_distance_x == 0:
                continue
            if exclude_same_example and edit_distance_x == 0 and edit_distance_y == 0:
                continue
            ranked_examples.append((edit_distance_x, edit_distance_y, x_prime, y_prime))

    return sorted(set(ranked_examples), key=lambda x:(x[1], x[0]))  # Remove duplicates


###############################################################################
# Construct datasets
###############################################################################

def construct_examples_with_suffix(dataset):
    """
    To speed up, construct clusters of examples based on their suffix
    """
    examples_with_suffix = collections.defaultdict(list)
    for example in dataset:
        x = example[0]
        x_suffix = get_suffix(x)
        examples_with_suffix[x_suffix].append(example)
    return examples_with_suffix


def construct_dataset(sources):
    """
    From a project, extract all (x, y) pairs
    """
    dataset = []
    num_lines = 0
    for source in sources:
        num_lines += len(get_lines_from_source(source,
                                               remove_comments_from_source=True,
                                               remove_empty_lines_from_source=True))
        examples = extract_examples_line_by_line(source)
        dataset.extend(examples)

    print(f"\n[construct_dataset] Number of lines in the project: {num_lines}")
    print(f"[construct_dataset] Number of examples: {len(dataset)} ({len(set(dataset))} unique examples)")
    return dataset


def construct_dataset_edit(opt,
                           dataset,
                           verbose=False):
    """
    Given (x, y) pairs, generate (x, y_abs, x', y')
    """
    dataset_edit = []
    num_examples_with_candidates = 0
    examples_with_suffix = construct_examples_with_suffix(dataset)

    print("\n[construct_dataset_edit] Start generating dataset for edit")
    for example in tqdm(dataset):
        x, y = example
        x_suffix = get_suffix(x)
        candidates = rank_based_on_distance(opt,
                                            example,
                                            examples_with_suffix[x_suffix],
                                            check_exact_match_suffix=False,  # No need to check if passing examples_with_suffix
                                            include_unsatisfying_examples=False,  # Difference: filter out
                                            exclude_same_context=False,
                                            exclude_same_example=True,
                                            verbose=verbose)
        if not candidates:
            pass
        else:
            num_examples_with_candidates += 1
            for candidate in candidates[:opt.max_num_candidates]:
                edit_distance_x, edit_distance_y, x_prime, y_prime = candidate
                y_abs = y  # TODO
                dataset_edit.append((x, y_abs, x_prime, y_prime))
    print(f"[construct_dataset_edit] Number of examples covered for edit: {num_examples_with_candidates}/{len(dataset)} ({num_examples_with_candidates/len(dataset)*100:.2f}%)")
    print(f"[construct_dataset_edit] Number of examples for edit: {len(dataset_edit)} ({len(set(dataset_edit))} unique examples for edit)")
    return dataset_edit


def generate_datasets(opt):
    """
    Check if processed datasets exist.

    If so, read. Otherwise, generate and save to the path.
    """
    print(f"\n[generate_datasets] Processing {opt.path}")
    options = to_string_opt(opt)
    path_dataset = os.path.join(opt.path, f"dataset_{options}.p")
    path_dataset_edit = os.path.join(opt.path, f"dataset_edit_{options}.p")

    if os.path.exists(path_dataset) and os.path.exists(path_dataset_edit) and not opt.overwrite:
        print("[generate_datasets] Read from pickled files")
        dataset = read_from_pickle(path_dataset)
        dataset_edit = read_from_pickle(path_dataset_edit)
    else:
        sources, num_file = read_data(opt.path)

        dataset = construct_dataset(sources)  # D_proj = {(x, y)}
        if len(dataset) > 10000:
            print(f"[generate_datasets] Skipping too large project (|dataset| = {len(dataset)})")
            dataset_edit = []
        else:
            dataset_edit = construct_dataset_edit(opt, dataset)  # D_edit = {(x, y, x', y')}

            # Save datasets for future use
            if opt.save:
                save_as_pickle(path_dataset, dataset)
                save_as_pickle(path_dataset_edit, dataset_edit)

    print(f"\n[generate_datasets] Number of examples: {len(dataset)} ({len(set(dataset))} unique examples)")
    print(f"[generate_datasets] Number of examples for edit: {len(dataset_edit)} ({len(set(dataset_edit))} unique examples)\n")

    return dataset, dataset_edit


###############################################################################
# Render html
###############################################################################

ignorable = '(\s*""".*?"""\s*|\s*\'\'\'.*?\'\'\'\s*|\s)*'


def generate_html(sources):
    html = ""
    for source in sources:
        if len(source.strip()) == 0:
            continue
        html += f"#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n\n{source}\n\n"
    return html


def collect_code_with_edits(dataset_edit, html):
    """
    Collect all exact code (~ x + y) that have example edits in dataset_edit.
    """
    # TODO Ignore one liner comments starting with #
    code_with_edits = set()
    cnt_not_found, cnt_found = 0, 0
    for x, y, x_prime, y_prime in dataset_edit:  # NOTE y is not abstracted
        x, y = reverse_process_example(x, y)  # Cut and paste the suffix of x to the beginning of y
        code_to_find_ptr = re.escape(x.strip()) + ignorable + re.escape(y.strip())  # Add ignorable
        code_to_find_ptr = re.compile(code_to_find_ptr, re.MULTILINE|re.DOTALL)

        # Find actual code from html
        found = re.search(code_to_find_ptr, html)
        if not found:
            print("[!] code block not found in source code")
            print(x)
            print(y)
            cnt_not_found += 1
        else:
            code = found.group(0).strip()
            code_with_edits.add(code)
            cnt_found += 1
    print("\nMark code with edits in html")
    print("Found:", cnt_found)
    print("Not found:", cnt_not_found)
    return sorted(code_with_edits, key=len, reverse=True)


###############################################################################
# Main
###############################################################################

def main(opt):
    print("[main] Executing main function with options")
    sources, num_files = read_data(opt.path)
    dataset, dataset_edit = generate_datasets(opt)

    html = generate_html(sources)  # For rendering
    code_with_edits = collect_code_with_edits(dataset_edit, html)  # For rendering
    examples_with_suffix = construct_examples_with_suffix(dataset)  # For metadata

    print("\nTotal number of files processed:", num_files)

    print("\nDistance metric (x):", opt.distance_metric_x)
    print("Distance threshold (x):", opt.distance_threshold_x)
    print("Distance metric (y):", opt.distance_metric_y)
    print("Distance threshold (y):", opt.distance_threshold_y)

    # print("\nNumber of leftmost tokens for keywords (n):", opt.n_leftmost_tokens)
    # print("Maximum number of candidates to generate example edits (k):", opt.max_num_candidates)
    # print("Enforce exact match of the suffix of x and x':", opt.check_exact_match_suffix)

    print(f"\nTotal number of examples: {len(dataset)}")
    print(f"Total number of example edits: {len(dataset_edit)} ({len(dataset_edit) / len(dataset) * 100:.2f}%)")
    print(f"Total number of code with edits: {len(code_with_edits)}", "\n")

    return {
        'sources': sources,
        'num_files': num_files,
        'dataset': dataset,
        'dataset_edit': dataset_edit,
        'html': html,
        'code_with_edits': code_with_edits,
        'examples_with_suffix': examples_with_suffix
    }




###############################################################################
# Flask
###############################################################################

@app.route("/apply_options", methods=["POST"])
def apply_options():
    options = request.get_json(force=True)
    print("\n[apply_options] Applying requested options:", options)

    # Update opt values
    global opt
    opt.distance_metric_x = options['distance_metric_x']
    opt.distance_threshold_x = float(options['distance_threshold_x'])
    opt.distance_metric_y = options['distance_metric_y']
    opt.distance_threshold_y = float(options['distance_threshold_y'])

    # Update processed data for new options
    global results
    results = main(opt)

    return render_template("index.html",
                           html=results['html'],
                           dataset_size=len(results['dataset']),
                           dataset_edit_size=len(results['dataset_edit']),
                           coverage=f"{len(results['dataset_edit']) / len(results['dataset']) * 100:.2f}",
                           distance_metric_x=opt.distance_metric_x,
                           distance_threshold_x=opt.distance_threshold_x,
                           distance_metric_y=opt.distance_metric_y,
                           distance_threshold_y=opt.distance_threshold_y)



@app.route("/get_code_with_edits", methods=["POST"])
def get_code_with_edits():
    return jsonify(results['code_with_edits'])


@app.route("/get_metadata", methods=["POST"])
def get_metadata():
    example = request.get_json(force=True)

    # Process example
    x = example['x'].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    y = example['y'].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    x, y = process_example(x, y)

    # Get metadata
    x_suffix = get_suffix(x)
    example_edits = rank_based_on_distance(opt,
                                           (x, y),
                                           results['examples_with_suffix'][x_suffix],
                                           check_exact_match_suffix=False,  # No need to check if passing examples_with_suffix
                                           include_unsatisfying_examples=False,
                                           exclude_same_context=False,
                                           exclude_same_example=True)

    example_edits = [{
                        'edit_distance_x': f'{edit_distance_x:.2f}',
                        'edit_distance_y': f'{edit_distance_y:.2f}',
                        'x': x,
                        'y': y,
                    }
                    for (edit_distance_x, edit_distance_y, x, y) in example_edits]
    data = {
        'x': x,
        'y': y,
        'example_edits': example_edits,
    }
    return jsonify(data)


@app.route("/")
def intro():
    return render_template("index.html",
                           html=results['html'],
                           dataset_size=len(results['dataset']),
                           dataset_edit_size=len(results['dataset_edit']),
                           coverage=f"{len(results['dataset_edit']) / len(results['dataset']) * 100:.2f}",
                           distance_metric_x=opt.distance_metric_x,
                           distance_threshold_x=opt.distance_threshold_x,
                           distance_metric_y=opt.distance_metric_y,
                           distance_threshold_y=opt.distance_threshold_y,
                           min_len=opt.min_len)


if __name__ == "__main__":
    # # Read default options
    # parser = configargparse.ArgumentParser(description="index.py")
    # opts.system_opts(parser)
    # opt = parser.parse_args()

    results = None
    # Execute main function with default options
    results = main(opt)
    app.run(host="0.0.0.0",  # Public
            debug=opt.debug)

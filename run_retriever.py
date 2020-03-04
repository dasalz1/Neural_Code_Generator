import ray
ray.init()
import pickle, os, math
import pandas as pd
from itertools import filterfalse
from tqdm import tqdm
import pathlib, logging
from rule_based_retriever.index import *
from DataClass.data_utils import set_logger
from collections import namedtuple, Counter
import gc
from DataClass.data_utils import preprocess_tokens
import numpy as np
logger = logging.getLogger(__name__)


def generate_datasets(opt, max_lines = 10000):
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
        if len(dataset) > max_lines:
            print(f"[generate_datasets] Skipping too large project (|dataset| = {len(dataset)})")
            dataset_edit = []
        else:
            dataset_edit = [ray.remote(opt, example) for example in dataset]
            res = ray.get(dataset_edit)
            dataset_edit = list(filterfalse(lambda x: len(x) == 0, res)) #D_edit = {(x, y, x', y')}

            print(f"\n[generate_datasets] Number of examples: {len(dataset)} ({len(set(dataset))} unique examples)")
            print(f"[generate_datasets] Number of examples for edit: {len(dataset_edit)} ({len(set(dataset_edit))} unique examples)\n")

            # dataset_edit = construct_dataset_edit(opt, dataset)  # D_edit = {(x, y, x', y')}

            # Save datasets for future use
            if opt.save:
                save_as_pickle(path_dataset, dataset)
                save_as_pickle(path_dataset_edit, dataset_edit)

    print(f"\n[generate_datasets] Number of examples: {len(dataset)} ({len(set(dataset))} unique examples)")
    print(f"[generate_datasets] Number of examples for edit: {len(dataset_edit)} ({len(set(dataset_edit))} unique examples)\n")

    return dataset, dataset_edit

def get_params(p, debug = False, save = True, overwrite = False, verbose = False,
               unit = 'line', min_len = 10, distance_metric_x = 'nc',
               distance_threshold_x = 0.3, distance_metric_y = 'c',
               distance_threshold_y = 2, n_leftmost_tokens = 1, max_num_candidates = 1,
               check_exact_match_suffix = False):
    opt = namedtuple('opt', ['path', 'debug', 'save', 'overwrite', 'verbose', 'unit', 'min_len',
                             'distance_metric_x', 'distance_threshold_x', 'distance_metric_y',
                             'distance_threshold_y', 'n_leftmost_tokens', 'max_num_candidates',
                             'check_exact_match_suffix'])
    param = opt(p, debug, save, overwrite, verbose, unit, min_len, distance_metric_x,
               distance_threshold_x, distance_metric_y,
               distance_threshold_y, n_leftmost_tokens, max_num_candidates, check_exact_match_suffix)
    return param

def load_saved_datasets(dataset_path, dataset_edit_path = None):
    if dataset_edit_path is not None:
        with open(dataset_edit_path, 'rb') as f:
            dataset_edit = pickle.load(f)
        f.close()

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    f.close()
    return dataset, dataset_edit

def load_pickle(filename, obj):
    with open(filename, 'wb') as file:
        x = pickle.load(file)
    file.close()
    return x

# def save_pickle(filename, obj):
#     with open(filename, 'wb') as file:
#         pickle.dump(obj, file , protocol=pickle.HIGHEST_PROTOCOL)
#     file.close()

def run_example(example, examples_with_suffix, opt):

    # Process example
    x = example[0].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    y = example[1].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    # x, y = process_example(x, y)


    # Get metadata
    x_suffix = get_suffix(x)
    example_edits = rank_based_on_distance(opt,
                                           (x, y),
                                           examples_with_suffix[x_suffix],
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
    return data

# @ray.remote
def main(line, opt, examples_with_suffix, i, top_k = 10):
    all_data = {}
    data = run_example(line, examples_with_suffix, opt)
    all_data[(i, line[0], line[1])] = data['example_edits'][:top_k]
    return all_data

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

def readcsv(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=0)
    t1 = df.iloc[:, 0].tolist()
    t2 = df.iloc[:, 1].tolist()
    src = ' \n '.join(t1)
    return [src], len(t1)

@ray.remote
def parallel_retriever(subdirectory, pickle_dir, debug = True, repo = False):
    opt = get_params(str(subdirectory))
    if repo:
        sources, num_files = read_data(opt.path)
        dataset = construct_dataset(sources)  # D_proj = {(x, y)}
    else:
        sources, num_lines = readcsv(opt.path)

    if num_lines < 1000:
        print(opt)
        logging.info(f'Params is : {opt}')
        print("[main] Executing main function with options")
        dataset = construct_dataset(sources)
        examples_with_suffix = construct_examples_with_suffix(dataset)  # For metadata
        dir_name = opt.path.split('/')[-1].replace('.csv', '')

        # if opt.save:
        #     fname_dataraw = '_'.join([str(dir_name), 'dataset.pkl'])
        #     with open(pickle_dir / fname_dataraw, 'wb') as f:
        #         pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        #     f.close()
        del sources
        gc.collect()

        print(f"Processing Directory {dir_name} ... ")
        if debug:
            result = []
            i = 0
            while dataset:
                line = dataset.pop()
            # for i, line in enumerate(dataset):
                result.append(main(line, opt, examples_with_suffix, i, top_k=3))
        else:
            output = ray.get([main.remote(line, opt, examples_with_suffix, i) for i, line in enumerate(dataset)])
            result = list(filterfalse(lambda x: len(x) == 0, output))

        if opt.save:
            fname_edit_output = ''.join([str(dir_name), 'dataset_edit.pkl'])
            with open(pickle_dir / fname_edit_output, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        del result
        gc.collect()
    return subdirectory

if __name__ == '__main__':

    # log_dir = pathlib.Path.cwd() / 'logs'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    output_dir = pathlib.Path.cwd() / 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle_dir = pathlib.Path.cwd() / 'github_data' / 'pickle_files'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    # set_logger(log_dir / 'rule_based.log')
    p = pathlib.Path.cwd() / 'github_data' / 'repo_files'
    # subdirs = [x for x in p.iterdir() if 'tensorflow-vgg' in str(x)]
    existing_files = [str(x).split('/')[-1] for x in pickle_dir.iterdir()]
    finished_files = Counter(existing_files)
    subdirs = [x for x in p.iterdir() if ('.csv' in str(x))
               and (str(x).split('/')[-1].replace('.csv', 'dataset_edit.pkl') not in finished_files)]
    # subdirs = [p / 'yt-dl_line_pairs.csv']
    print(f'length of subdirs is {len(subdirs)}')
    retrieved_examples = {}
    # for i, subdir in enumerate(subdirs):
    #     name = str(subdir).split('/')[-1]
    #     if i % 100 == 0:
    #         print(f'Completed {i // len(subdirs)}')
    #     retrieved_examples[subdir] = parallel_retriever(subdir, pickle_dir, debug = True)
    # for subdirect in np.array_split(subdirs, 500):
    retrieved_repos = ray.get([parallel_retriever.remote(subdir, pickle_dir, debug = True) for subdir in subdirs])

    # for repo_name, result in retrieved_repos:
        # retrieved_examples[repo_name] = result

    x_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 0]), self.max_dim)
    y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 1]), self.max_dim)

    x_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in x_tokens]
    y_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in y_tokens]

    output_name = '_'.join([to_string_opt(opt), '.pkl'])
    with open(output_dir / output_name, 'wb') as f:
        pickle.dump(retrieved_examples, f)
    f.close()
    print(f'Length of final output is {len(retrieved_examples)}')
    logging.info(f'pickle file saved to {output_name}')
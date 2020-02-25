import ray
ray.init()
import pickle, os, math
import pathlib, configargparse, logging
from rule_based_retriever.index import construct_examples_with_suffix, get_suffix, rank_based_on_distance, \
    generate_datasets, generate_html, collect_code_with_edits, read_data
from DataClass.data_utils import set_logger
from collections import namedtuple

def get_params(p, debug = False, save = True, overwrite = False, verbose = False,
               unit = 'line', min_len = 10, distance_metric_x = 'nc',
               distance_threshold_x = math.inf, distance_metric_y = 'c',
               distance_threshold_y = math.inf, n_leftmost_tokens = 1, max_num_candidates = 1,
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
#
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

def main(ds, opt, top_k = 10):
    examples_with_suffix = construct_examples_with_suffix(ds)
    all_data = {}
    for i, line in enumerate(ds):
        data = run_example(line, examples_with_suffix, opt)
        all_data[(i, line[0], line[1])] = data['example_edits'][:top_k]
    return all_data

@ray.remote
def parallel_retriever(subdirectory):
    # parser = configargparse.ArgumentParser(description="run_retriever.py")
    # system_opts(parser, subdirectory)
    # opt = parser.parse_args()
    opt = get_params(str(subdirectory))
    print(opt)
    logging.info(f'Params is : {opt}')
    print("[main] Executing main function with options")
    sources, num_files = read_data(opt.path)
    dataset, dataset_edit = generate_datasets(opt, max_lines = 100000)

    html = generate_html(sources)  # For rendering
    code_with_edits = collect_code_with_edits(dataset_edit, html)  # For rendering
    examples_with_suffix = construct_examples_with_suffix(dataset)  # For metadata

    logging.info(f"Processing Directory {opt.path.split('/')[-1]} ... ")
    logging.info(f'Total number of files processed: {num_files}')
    logging.info(f'Total number of examples: {len(dataset)}')
    try:
        logging.info(f'Total number of example edits: {len(dataset_edit)} ({len(dataset_edit) / len(dataset) * 100:.2f}%)')
    except:
        print(f"Error For {opt.path.split('/')[-1]}")
    logging.info(f'Total number of code with edits: {len(code_with_edits)} \n')

    result = {
        'sources': sources,
        'num_files': num_files,
        'dataset': dataset,
        'dataset_edit': dataset_edit,
        'html': html,
        'code_with_edits': code_with_edits,
        'examples_with_suffix': examples_with_suffix
    }
    output = main(dataset, opt)
    return output

if __name__ == '__main__':
    # edit_path = pathlib.Path.cwd() /'github_data' / 'tensorflow-vgg' / 'dataset_edit_u_line__d_metric_x_nc__d_thre_x_inf__d_metric_y_c__d_thre_y_inf__n_1__max_can_1.p'
    # ds_path = pathlib.Path.cwd() / 'github_data' / 'tensorflow-vgg' / 'dataset_u_line__d_metric_x_nc__d_thre_x_inf__d_metric_y_c__d_thre_y_inf__n_1__max_can_1.p'
    #
    # ds, ds_edit = load_saved_datasets(ds_path, edit_path)
    #
    # all_data = main(ds)
    log_dir = pathlib.Path.cwd() / 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    set_logger(log_dir / 'rule_based.log')
    p = pathlib.Path.cwd() / 'github_data'
    subdirs = [x for x in p.iterdir() if x.is_dir()]
    result = [parallel_retriever.remote(str(d)) for d in subdirs]
    retrieved_ex = ray.get(result)
    # result = parallel_retriever(str(subdirs[0]))


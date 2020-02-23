import pickle
import pathlib

from rule_based_retriever.index import construct_examples_with_suffix, get_suffix, rank_based_on_distance, opt


def load_saved_datasets(dataset_edit_path, dataset_path):
    with open(dataset_edit_path, 'rb') as f:
        dataset_edit = pickle.load(f)
    f.close()

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    f.close()
    return dataset, dataset_edit

def run_example(example, examples_with_suffix):

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

def main(ds, top_k = 10):
    examples_with_suffix = construct_examples_with_suffix(ds)
    all_data = {}
    for i, line in enumerate(ds[:100]):
        data = run_example(line, examples_with_suffix)
        all_data[(i, line[0], line[1])] = data['example_edits'][:top_k]
    return all_data

if __name__ == '__main__':
    edit_path = pathlib.Path.cwd() /'github_data' / 'tensorflow-vgg' / 'dataset_edit_u_line__d_metric_x_nc__d_thre_x_inf__d_metric_y_c__d_thre_y_inf__n_1__max_can_1.p'
    ds_path = pathlib.Path.cwd() / 'github_data' / 'tensorflow-vgg' / 'dataset_u_line__d_metric_x_nc__d_thre_x_inf__d_metric_y_c__d_thre_y_inf__n_1__max_can_1.p'

    ds, ds_edit = load_saved_datasets(edit_path, ds_path)

    all_data = main(ds)




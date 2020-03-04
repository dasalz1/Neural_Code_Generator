import ray
ray.init()
import pickle, pathlib, os, argparse, itertools
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from DataClass import data_utils, torchData
from DataClass.torchData import *
import argparse, random
import numpy as np

def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    file.close()
    return x

def join_tuple_xy(k, xkey = None, ykey = None):
    if xkey is None:
        s = k[1].split('\n')
    else:
        s = k[xkey].split('\n')
    if len(s) > 2:
        y_pre = s[-1]
        x = '\n'.join(s[:-1])
        y = ' '.join([y_pre, k[2]]) if ykey is None else ' '.join([y_pre, k[ykey]])
    else:
        x = s[0]
        y_pre = s[-1]
        y = ''.join([y_pre, k[2]]) if ykey is None else ' '.join([y_pre, k[ykey]])
    return x, y

@ray.remote
def process_to_df(subdirs, write_dir):
    file = load_pickle(subdirs)

    df = {'linenum' : [],
          'x' : [], 'y' : [],
          'x1' : [],  'y1' : [], 'edit_distance_x1' : [], 'edit_distance_y1' : [],
          'x2' : [], 'y2' : [], 'edit_distance_x2' : [], 'edit_distance_y2' : []}

    for i, l in enumerate(file):
        for keys, values in l.items():
            if len(values) != 0:
                cnt = 0
                df['linenum'].append(i)
                x, y = join_tuple_xy(keys)
                df['x'].append(x)
                df['y'].append(y)
                while values and (cnt < 2):
                    line = values.pop(0)
                    xp, yp = join_tuple_xy(line, xkey = 'x', ykey = 'y')
                    df['x' + str(cnt + 1)].append(xp)
                    df['y' + str(cnt + 1)].append(yp)
                    df['edit_distance_x' + str(cnt + 1)].append(float(line['edit_distance_x']))
                    df['edit_distance_y' + str(cnt + 1)].append(float(line['edit_distance_y']))

                    cnt += 1
                if cnt == 1:
                    df['x2'].append('')
                    df['y2'].append('')
                    df['edit_distance_x2'].append(np.NaN)
                    df['edit_distance_y2'].append(np.NaN)
    dfs = pd.DataFrame.from_dict(df, orient = 'columns')
    fname = str(subdirs).split('/')[-1].replace('.pkl', '.csv')
    if dfs.shape[0] == 0:
        return False
    dfs.to_csv(write_dir / fname, index = False)
    return True

def read_files(directory):
    list_of_files = [x for x in directory.iterdir() if '.csv' in str(x)]
    dfs = []
    for file in list_of_files:
        fname = str(file).split('/')[-1]
        df = pd.read_csv(file, header = 0, index_col = 0)
        df = df.assign(filename = fname)
        dfs.append(df)
    dfs = pd.concat(dfs, ignore_index = True, axis = 0)
    return dfs




if __name__ == '__main__':
    parser = argparse.ArgumentParser('rulebased_examples.py')
    parser.add_argument('--process_raw', default = 0, help = 'process from scratch')
    parser.add_argument('--evaluate', default = 1, help = 'evaluate BLEU score')
    parser.add_argument('--filepath', default = './github_data/repo_files', help = 'directory to repo_files')
    parser.add_argument('--batch_size', default = 128, help = 'batch_size')
    args = parser.parse_args()

    path = pathlib.Path.cwd() / 'github_data' / 'pickle_files'
    subdirs = [x for x in path.iterdir() if 'dataset_edit.pkl' in str(x)]
    write_dir = pathlib.Path.cwd() / 'github_data' / 'rulebased_ret'

    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if args.process_raw:
        result = ray.get([process_to_df.remote(subdir, write_dir) for subdir in subdirs])
        print(f'Successfully Processed : {sum(result) / len(subdirs)}')

    directory = pathlib.Path.cwd() / 'github_data' / 'rulebased_ret'
    repo_files = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(args.filepath))[2]))
    num_validation_repos = 100

    dfs = read_files(directory)

    x_tokens = (dfs['x'].apply(lambda x: tokenize_fine_grained(x))
                        .apply(lambda x: preprocess_tokens(x, torchData.MAX_LINE_LENGTH))
                        .apply(lambda y: list(itertools.filterfalse(lambda x: x == '<PAD>', y))))
    y_tokens = (dfs['y'].apply(lambda x: tokenize_fine_grained(x))
                        .apply(lambda x: preprocess_tokens(x, torchData.MAX_LINE_LENGTH))
                        .apply(lambda y: list(itertools.filterfalse(lambda x: x == '<PAD>', y))))
    yp_tokens = (dfs['y1'].apply(lambda x: tokenize_fine_grained(x))
                          .apply(lambda x: preprocess_tokens(x, torchData.MAX_LINE_LENGTH))
                          .apply(lambda y: list(itertools.filterfalse(lambda x: x == '<PAD>', y))))


    # bleu = corpus_bleu(dfs['y'].tolist(), dfs['y1'].tolist(), smoothing_function = SmoothingFunction().method1)

    ray.shutdown()
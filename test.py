import pickle
import pathlib2

data_dir = pathlib2.Path.cwd() / 'github_data' / 'test' / '3drtx_line_pairs.pickle'

with open(str(data_dir), 'rb') as f:
    obj = pickle.load(f)
f.close()
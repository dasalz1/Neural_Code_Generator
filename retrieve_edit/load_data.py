import pickle, pathlib, ray
ray.init()
# import pathlib2
# import paths
import os
import pandas as pd
import itertools
from collections import namedtuple
from DataClass3.data_utils import tokenize_fine_grained
# os.environ['COPY_EDIT_DATA']='./data/'
# os.environ['CUDA_VISIBLE_DEVICES']='2'
# from gtd.utils import Config

# from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRuns
# print os.environ['COPY_EDIT_DATA']
# from editor_code.copy_editor import edit_training_run
# cd = pathlib2.Path.cwd() / 'github_data' / 'processed_repo_pkl' / 'train'
# all_data = cd / 'train'
# direct = [str(x) for x in all_data.iterdir() if '.pickle']
# with open(str(cd), 'rb') as f:
#     tmp = pickle.load(f)
# f.close()
# print(f)
# data_dir = pathlib.Path.cwd() / 'github_data' / 'pickle_files' / 'search-arxiv_line_pairsdataset_edit.pkl'
# with open(str(data_dir), 'rb') as f:
#     tmp = pickle.load(f)
# f.close()
# data_editor = edit_training_run.EditExample(data_dir, config)
import sys

class UnicodeMixin(object):

  """Mixin class to handle defining the proper __str__/__unicode__
  methods in Python 2 or 3."""

  if sys.version_info[0] >= 3: # Python 3
      def __str__(self):
          return self.__unicode__()
  else:  # Python 2
      def __str__(self):
          return self.__unicode__().encode('utf8')

from nltk import word_tokenize

class EditExample(namedtuple('EditExample', ['input_words', 'target_words']), UnicodeMixin):
    """
    Attributes:
        input_words (list[list[unicode]]): a list of sequences, one sequence per input channel
        target_words (list[unicode]): a single output sequence
    """
    @classmethod
    def from_prompt(cls):
        get_input = lambda prompt: word_tokenize(input(prompt).decode('utf-8'))
        input_words = []
        for i in itertools.count():
            input_i = get_input('Enter input #{} (leave blank to terminate):\n'.format(i))
            if len(input_i) == 0:
                break
            input_words.append(input_i)

        target_words = get_input('Enter target sentence (OK to leave empty):\n')
        return EditExample(input_words, target_words)

    def __new__(cls, input_words, target_words):
        # verify input_words is a list of sequences, not just a single sequence
        try:
            assert isinstance(input_words[0], list)
        except:
            input_words = [input_words]
        #lower = lambda seq: [w.lower() for w in seq]
        #input_words = [lower(seq) for seq in input_words]
        #target_words = lower(target_words)
        self = super(EditExample, cls).__new__(cls, input_words, target_words)
        return self

    def __unicode__(self):
        return u'INPUT:\n{}\nTARGET: {}'.format(u'\n'.join([u'\t' + u' '.join(words) for words in self.input_words]),
                                                u' '.join(self.target_words))

@ray.remote(num_cpus=0.1)
def process_data(data_path, seq_length_limit):
    examples = {}
    MAX_LINE_LENGTH = 128
    fname = str(data_path).split('/')[-1].split('.')
    name = '{}.pickle'.format(fname)
    file = pathlib.Path.cwd() / 'github_data' / 'processed_repo_pkl' / name
    max_seq_length = lambda ex: max(max(len(seq) for seq in ex.input_words), len(ex.target_words))
    # if os.path.exists(str(file)):
    #     with open(str(file), 'rb') as f:
    #         examples = pickle.load(f)
    #     f.close()
    #     return list(examples.values())
    # count total lines before loading
    # for line in verboserate(data_paths, desc='Reading data file.', total=num_direct):
    df = pd.read_csv(data_path, skiprows=2, header=None, names=[0, 1], dtype=str).fillna('OSOFo')
    df[0] = df[0].apply(lambda x: tokenize_fine_grained(x))
    # df[0] = df[0].apply(lambda x: preprocess_tokens(x, MAX_LINE_LENGTH))
    df[1] = df[1].apply(lambda x: tokenize_fine_grained(x))
    try:
        ex = []
        for i, row in df.iterrows():
            try:
                ex.append(EditExample(row[0], row[1]))
            except:
                # print('bad formatting in file ' + str(line).split('/')[-1])
                # print line
                count = 1
        # skip sequences that are too long, because they use up memory
        # if max_seq_length(ex) > seq_length_limit:
        #     continue
        ex = list(itertools.filterfalse(lambda x: max_seq_length(x) > seq_length_limit, ex))
        # examples[str(line).split('/')[-1]] = ex
        examples[(str(data_path).split('/')[-1], len(ex))] = ex
        print('done {}'.format((str(data_path).split('/')[-1])))
    except Exception as e:
        print(e, 'bad formatting')
        return None

    file = pathlib.Path.cwd() / 'github_data' / 'pickle_files_py2' / name

    # with open(file, 'wb') as f:
    #     pickle.dump(examples, f, protocol=2)
    # f.close()
    return examples


data_path = pathlib.Path.cwd() / 'github_data' / 'repo_files'
existing_files = [str(x) for x in data_path.iterdir() if '.csv' in str(x)]
result = ray.get([process_data.remote(p, 150) for p in existing_files])

flatten = lambda l: [item for sublist in l for item in sublist]

write_dir = pathlib.Path.cwd() / 'github_data' / 'pickle_files_py2'
from tqdm import tqdm
for examp in tqdm(result):
    if examp is not None and len(examp) != 0:
        for k, v in examp.items():
            file = write_dir / '{}.pickle'.format(k[0].split('.')[0])
            with open(file, 'w+') as f:
                pickle.dump(examp, f, protocol = 2)
            f.close()

file = write_dir / 'all.pickle'
with open(file, 'wb') as f:
    pickle.dump(result, f, protocol = 2)
f.close()




# from copy import deepcopy
# tmp = result[:5]
#
# test_dir = pathlib.Path.cwd() / 'github_data' / 'test'
# with open(test_dir / 'tmp.pkl', 'wb') as f:
#     pickle.dump(ee, f, protocol = 2)
# f.close()
#
# test_edit_ex = list(tmp[0].values())[0]
#
# ee = EditExample(test_edit_ex[0].input_words, test_edit_ex[0].target_words)

# with open(file, 'wb') as f:
#     pickle.dump(result, f, protocol=2)
# f.close()

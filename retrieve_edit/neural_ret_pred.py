
import paths
from editor_code.copy_editor.edit_training_run import EditDataSplits
import os

#os.environ['COPY_EDIT_DATA'] = paths.data_dir
os.environ['COPY_EDIT_DATA']='./data/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import sys
def set_output_encoding(encoding='utf-8'):
    import sys
    import codecs
    '''When piping to the terminal, python knows the encoding needed, and
       sets it automatically. But when piping to another program (for example,
       | less), python can not check the output encoding. In that case, it 
       is None. What I am doing here is to catch this situation for both 
       stdout and stderr and force the encoding'''
    current = sys.stdout.encoding
    if current is None :
        sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    current = sys.stderr.encoding
    if current is None :
        sys.stderr = codecs.getwriter(encoding)(sys.stderr)

#Note - we need this or else the program crashes due to a utf-8 error when trying to pipe the outputs to a text file. comment this line out when running interactively
set_output_encoding()

from gtd.utils import Config

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRun
from editor_code.copy_editor.editor import EditExample
from editor_code.copy_editor.vocab import HardCopyDynamicVocab

from gtd.utils import bleu

print os.environ['COPY_EDIT_DATA']

# no-profile
profile = False

# config = Config.from_file('editor_code/configs/editor/github.txt')
config = Config.from_file('editor_code/configs/editor/default.txt')
src_dir = os.environ['COPY_EDIT_DATA']+'edit_runs/11' # for loading checkpoint
load_expt = RetrieveEditTrainingRun(config, src_dir)

import numpy as np

vae_editor = load_expt.editor.vae_model
ret_model = load_expt.editor.ret_model
edit_model = load_expt.editor.edit_model
examples = load_expt._examples

from gtd.utils import chunks
from tqdm import tqdm

new_vecs = []
for batch in tqdm(chunks(examples.train,32), total=len(examples.train)/32):
    encin = ret_model.encode(batch, train_mode=False).data.cpu().numpy()
    for vec in encin:
        new_vecs.append(vec)
    del encin

new_lsh = ret_model.make_lsh(new_vecs)




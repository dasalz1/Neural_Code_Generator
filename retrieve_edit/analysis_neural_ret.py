import paths
import os
os.environ['COPY_EDIT_DATA']='./data/'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from gtd.utils import Config

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRuns, RetrieveEditTrainingRun
print os.environ['COPY_EDIT_DATA']
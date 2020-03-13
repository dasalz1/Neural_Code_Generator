import paths
import os
os.environ['COPY_EDIT_DATA']='./data/'
os.environ['CUDA_VISIBLE_DEVICES']='0'
from gtd.utils import Config

from editor_code.copy_editor.retrieve_edit_run import RetrieveEditTrainingRuns, RetrieveEditTrainingRun
print os.environ['COPY_EDIT_DATA']
import sys, pathlib2

#no-profile
profile=False

src_dir = os.environ['COPY_EDIT_DATA']+'/edit_runs/7' #for codalab
# load_expt = RetrieveEditTrainingRun(config, src_dir)
# runs = RetrieveEditTrainingRuns()

config_file = 'default.txt'
config = Config.from_file('editor_code/configs/editor/'+config_file)
run = RetrieveEditTrainingRun(config, src_dir)

# run = runs.new(config)

if profile:
    from gtd.chrono import Profiling, Profiler

    profiler = Profiler.default()

    import editor_code.copy_editor.retriever
    import editor_code.copy_editor.editor
    profiler.add_module(editor_code.copy_editor.editor)
    profiler.add_module(editor_code.copy_editor.retriever)
    Profiling.start()
    run.train()
    Profiler.report(profiler)  # prints out report

else:
    run.train()


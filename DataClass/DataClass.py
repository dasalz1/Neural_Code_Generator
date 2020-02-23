import numpy as np
import pandas as pd
import os
from regex_utils import remove_comments, split_newlines
from data_utils import read_data




class Dataset:

	def __init__(self):
		pass

	def _generate_pair_dataset_from_repo(self, path, output_dir = '.', filename_ending='_line_pairs'):
		sources, num_file = read_data(path)
		if len(sources) == 0: return
		all_lines = None
		for source in sources:
		    lines = self.extract_examples_line_by_line(source)
		    if lines == []: continue
		    y = pd.DataFrame(lines)
		    y.columns = ['line']
		    y = y[y['line'].apply(lambda x: len(str(x).strip()) > 0)].reset_index(drop=True)    
		    x = pd.concat([pd.DataFrame([""]), y['line'][:-1]]).reset_index(drop=True)
		    pair = pd.concat([x, y], axis=1)
		    all_lines = pd.concat([all_lines, pair], axis=0)

		all_lines = pd.concat([pd.DataFrame(np.array([all_lines.shape[0], None]).reshape(1, -1), columns=all_lines.columns), all_lines], axis=0)
		all_lines.to_csv(output_dir + '/' + path[len(path) - path[::-1].find('/'):] + filename_ending + '.csv', header=None, index=None)

	# assumes dir is a path to a directory whose subfolders are the repos
	def generate_pair_datasets(self, repo_directory, output_dir = './repo_files'):
		try:
			os.mkdir(output_dir)
		except:
			pass

		repos = next(os.walk(repo_directory))[1]
		for repo in repos:	
			if repo.startswith('.'): continue
			self._generate_pair_dataset_from_repo(repo_directory+'/' + repo, output_dir=output_dir)

	def extract_examples_line_by_line(self, source):  # If line, no need to filter based on length
	    source = remove_comments(source)
	    lines = split_newlines(source)
	    if not lines:
	        return []
	    
	    return lines
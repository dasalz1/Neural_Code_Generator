import os
import pandas as pd

class RuleRetrieverDataClass:

	# directory is a directory containing all the different csv files of repos
	def __init__(self, repo_directory):
		
		self.repo_directory = repo_directory
		self.all_folders = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(repo_directory))[2]))
		self.curr_dir_idx = 0
		self.n_repos = len(self.all_folders)

	def get_next_repo(self):

		repo_df = pd.read_csv(self.repo_directory + '/' + self.all_folders[self.curr_dir_idx], skiprows=1, header=None)

		self.curr_dir_idx += 1
		self.curr_dir_idx = 0 if self.curr_dir_idx == self.n_repos else self.curr_dir_idx

		return repo_df

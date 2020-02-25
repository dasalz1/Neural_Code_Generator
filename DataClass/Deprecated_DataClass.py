import numpy as np
import pandas as pd
import os, pickle
from DataClass.regex_utils import remove_comments, split_newlines
from DataClass.data_utils import read_data, tokenize_fine_grained
from threading import Lock, Thread

MAX_THREADS = 2#160



class Dataset:

	def __init__(self):
		pass

	def _generate_pair_dataset_from_repo(self, path, output_dir = '.', filename_ending='_line_pairs', tokenizing=False, tokens=None, tokenize_threads=None, tokenize_lock=None):
		sources = read_data(path)
		if len(sources) == 0: return
		all_lines = None
		for source in sources:
		    lines = self.extract_examples_line_by_line(source)
		    if lines == []: continue

		    if tokenizing:
		    	tokenize_threads.append(Thread(target=threaded_tokenizer, args=(lines, tokenize_lock, tokens,)))
		    	tokenize_threads[-1].start()

		    y = pd.DataFrame(lines)
		    y.columns = ['line']
		    y = y[y['line'].apply(lambda x: len(str(x).strip()) > 0)].reset_index(drop=True)    
		    x = pd.concat([pd.DataFrame([""]), y['line'][:-1]]).reset_index(drop=True)
		    pair = pd.concat([x, y], axis=1)
		    all_lines = pd.concat([all_lines, pair], axis=0)

		all_lines = pd.concat([pd.DataFrame(np.array([all_lines.shape[0], None]).reshape(1, -1), columns=all_lines.columns), all_lines], axis=0)
		all_lines.to_csv(output_dir + '/' + path[len(path) - path[::-1].find('/'):] + filename_ending + '.csv', header=None, index=None)

	# assumes dir is a path to a directory whose subfolders are the repos
	def generate_pair_datasets(self, repo_directory, output_dir = './repo_files', tokenizing=False, repo_threading=True):
		try:
			os.mkdir(output_dir)
		except:
			pass

		tokenize_threads = [] if tokenizing else None
		tokenize_lock = Lock() if tokenizing else None
		tokens = {} if tokenizing else None

		if repo_threading:
			repo_threads = []

		repos = next(os.walk(repo_directory))[1]

		for repo in repos:
			print("Starting repo %s" % repo)
			if repo.startswith('.'): continue

			while(len(repo_threads) == MAX_THREADS):
				check_threads(repo_threads)

			if repo_threading:
				repo_threads.append(Thread(target=self._generate_pair_dataset_from_repo, args=(repo_directory+'/'+repo, 
																								output_dir, 
																								'_line_pairs',
																								tokenizing, 
																								tokens, 
																								tokenize_threads, 
																								tokenize_lock)))
				repo_threads[-1].start()
			else:
				self._generate_pair_dataset_from_repo(repo_directory+'/' + repo, 
															output_dir=output_dir, 
															tokenizing=tokenizing, 
															tokens=tokens, 
															tokenize_threads=tokenize_threads, 
															tokenize_lock=tokenize_lock)

		if repo_threading:
			for r_t in repo_threads:
				r_t.join()

		if tokenizing:
			print("Waiting for tokens")
			for t in tokenize_threads:
				t.join()

			with open(output_dir + '/' + 'all_tokens.pickle', 'wb') as f:
				pickle.dump(tokens, f)
		

	def extract_examples_line_by_line(self, source):  # If line, no need to filter based on length
	    source = remove_comments(source)
	    lines = split_newlines(source)
	    if not lines:
	        return []
	    
	    return lines


def check_threads(repo_threads):
	spawn_new = False
	for idx in reversed(range(len(repo_threads))):
		if not repo_threads[idx].is_alive():
			del repo_threads[idx]
			spawn_new = True

	return spawn_new


def threaded_tokenizer(lines, lock, tokens):
	for line in lines:
		line_tokens = tokenize_fine_grained(line)
		for token in line_tokens:
			if token in tokens: continue
			lock.acquire()
			tokens[token] = tokens.get(token, len(tokens))
			lock.release()
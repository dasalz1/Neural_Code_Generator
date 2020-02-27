import numpy as np
import pandas as pd
import os, pickle
from DataClass.regex_utils import remove_comments, split_newlines
from DataClass.data_utils import read_data, tokenize_fine_grained, get_urls_from_csv
from threading import Lock, Thread
from random import shuffle
from DataClass.Constants import PAD_WORD, START_WORD, END_WORD, PAD_IDX, START_IDX, END_IDX, NO_CONTEXT_IDX, NO_CONTEXT_WORD

MAX_REPO_THREADS = 64#160
MAX_TOKENIZE_THREADS = 128
MAX_LINES = 100#10000
SPLITTER_RANGE = 10



class Crawler:

	def __init__(self):
		pass

	def _generate_pair_dataset_from_url(self, url, output_dir = '.', filename_ending='_line_pairs', tokenizing=False, tokens=None, tokenize_lock=None):
		try:
			current_token_threads = []
			name = url.split('/')[-1][:-4]
			repo_path = os.path.join(output_dir, name)
			if os.path.exists(repo_path):
				print(f'Skipping existing repository: {name}')
			else:
				print(f'Cloning: {name}')
				t = Thread(target=os.system, args=("git clone {}".format(url),))
				t.start()
				t.join(30)
				# probably has a password so just ignore this repo
				if t.is_alive(): 
					return
 
			print("Finished repo %s" % name)
			sources = read_data(name)
		except:
			return
		if (sources==None) or (len(sources) == 0):
			os.system("rm -rf %s " % name)
			return
		all_lines = None
		for source in sources:
		    lines = self.extract_examples_line_by_line(source)
		    if lines == []: continue

		    if tokenizing:
		    	max_line_sz = [0]
		    	current_token_threads.append(Thread(target=threaded_tokenizer, args=(lines, tokenize_lock, tokens, max_line_sz,)))
		    	current_token_threads[-1].start()

		    y = pd.DataFrame(lines)
		    y.columns = ['line']
		    y = y[y['line'].apply(lambda x: len(str(x).strip()) > 0)].reset_index(drop=True)    
		    x = pd.concat([pd.DataFrame([""]), y['line'][:-1]]).reset_index(drop=True)
		    pair = pd.concat([x, y], axis=1)
		    all_lines = pd.concat([all_lines, pair], axis=0)

		all_lines = pd.concat([pd.DataFrame(np.array([all_lines.shape[0], None]).reshape(1, -1), columns=all_lines.columns), all_lines], axis=0)
		all_lines.to_csv(name.replace('.', '_') + filename_ending + '.csv', header=None, index=None)
		os.system("rm -rf %s " % name)
		for tokenize_thread in current_token_threads:
			tokenize_thread.join()

	# assumes dir is a path to a directory whose subfolders are the repos
	def generate_pair_datasets(self, url_csv='repos.csv', output_dir = './repo_files', tokenizing=False):
		try:
			os.mkdir(output_dir)
		except:
			pass

		full_path = os.getcwd()
		tokenize_threads = []
		tokenize_lock = Lock()
		tokens = {}
		tokens[PAD_WORD] = PAD_IDX
		tokens[END_WORD] = END_IDX
		tokens[START_WORD] = START_IDX
		tokens[NO_CONTEXT_WORD] = NO_CONTEXT_IDX

		if repo_threading:
			repo_threads = []

		urls = get_urls_from_csv(url_csv)
		shuffle(urls)

		os.chdir(output_dir)

		for urls_processed, url in enumerate(urls[:250000]):
			while(len(repo_threads) == MAX_REPO_THREADS):
				check_threads(repo_threads)

			repo_threads.append(Thread(target=self._generate_pair_dataset_from_url, args=(url,#repo_directory+'/'+repo, 
																								output_dir,
																								'_line_pairs',
																								tokenizing, 
																								tokens, 
																								tokenize_lock)))
			repo_threads[-1].start()
			# save current tokens in case error down the line
			if (urls_processed+1) % 1000 == 0:
				pickle.dump(tokens, open('all_tokens.pickle', 'wb'))

		if repo_threading:
			for r_t in repo_threads:
				r_t.join()

		if tokenizing:
			print("Waiting for tokens")
			for t in tokenize_threads:
				t.join()

			with open('max_line_size.txt', 'a') as f:
				f.write(str(max_line_sz[0]))
			pickle.dump(tokens, open('all_tokens.pickle', 'wb'))
		

	def extract_examples_line_by_line(self, source):  # If line, no need to filter based on length
	    source = remove_comments(source)
	    lines = split_newlines(source)
	    if not lines:
	        return []
	    
	    return lines

	# filepath is directory containing line pairs of csvs for repos
	def tokenize_from_files(self, filepath='.', tokens_filename='all_tokens', output_dir=None):
		tokens = {}
		tokens[PAD_WORD] = PAD_IDX
		tokens[END_WORD] = END_IDX
		tokens[START_WORD] = START_IDX
		tokens[NO_CONTEXT_WORD] = NO_CONTEXT_IDX

		tokenize_threads = []
		tokenize_lock = Lock()

		repos = next(os.walk(filepath))[2]
		max_line_sz = [0]
		curr_dir = os.getcwd()
		num_repos = len(repos)
		count = 0
		for repo in repos:
			count += 1
			print("On repo [%d/%d]" % (count, num_repos))
			repo = str(repo)
			if not repo.endswith('.csv'): continue
			while(len(tokenize_threads) == MAX_TOKENIZE_THREADS):
				check_threads(tokenize_threads)


			repo_new = repo[:-4].replace('.', '_') + '.csv'
			if repo_new != repo:
				os.system("mv %s %s" % (filepath + '/' + repo, filepath + '/' + repo_new))

			lines = pd.read_csv(filepath + '/' + repo_new).fillna(NO_CONTEXT_WORD)

			num_lines = int(lines.columns[0])
			lines = lines.iloc[:, 1]

			if num_lines > MAX_LINES:
				intermediate_size = int(num_lines/SPLITTER_RANGE)
    			
				ranges = [(val, val+intermediate_size) for val in range(0, num_lines, intermediate_size)]
				ranges[-1] = (ranges[-1][0], num_lines)
				for start, end in ranges:
					tokenize_threads.append(Thread(target=threaded_tokenizer, args=(lines.iloc[start:end], tokenize_lock, tokens, max_line_sz,)))
					tokenize_threads[-1].start()
    			
			else:
				tokenize_threads.append(Thread(target=threaded_tokenizer, args=(lines, tokenize_lock, tokens, max_line_sz,)))
				tokenize_threads[-1].start()

		for tokenize_thread in tokenize_threads:
			tokenize_thread.join()


		with open(output_dir if output_dir else filepath + '/' + 'max_line_size.txt', 'a') as f:
			f.write(str(max_line_sz[0]))
		pickle.dump(tokens, open(output_dir if output_dir else filepath + '/' + tokens_filename + '.pickle', 'wb'))


def check_threads(threads):
	spawn_new = False
	for idx in reversed(range(len(threads))):
		if not threads[idx].is_alive():
			del threads[idx]
			spawn_new = True

	return spawn_new


def threaded_tokenizer(lines, lock, tokens, max_line_sz):
	for line in lines:
		line_tokens = tokenize_fine_grained(line)
		if len(line_tokens) > max_line_sz[0]:
			max_line_sz[0] = len(line_tokens)
		for token in line_tokens:
			if token in tokens: continue
			lock.acquire()
			tokens[token] = tokens.get(token, len(tokens))
			lock.release()
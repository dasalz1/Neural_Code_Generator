import pandas as pd
import torch, os, pickle, warnings, csv
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from DataClass.data_utils import tokenize_fine_grained, create_vocab_dictionary, preprocess_tokens, preprocess_context, fix_quote_strings, fix_quote_strings_context
import numpy as np
from DataClass.Constants import NO_CONTEXT_WORD, UNKNOWN_IDX
from threading import Thread
from random import shuffle

warnings.simplefilter('ignore', pd.errors.ParserWarning)

# tokens_file = '../all_tokens.pickle'
# tokens_dict = pickle.load(open(tokens_file, 'rb'))


word2idx, idx2word = create_vocab_dictionary(path='..', file='all_tokens.pickle', save=True)
VOCAB_SIZE = len(word2idx)
# word2idx, idx2word = create_vocab_dictionary(path='..', file='all_tokens.pickle', save=True)

# UNKNOWN_IDX = word2idx[UNKNOWN_WORD]
MAX_LINE_LENGTH = 128


# use retrieved as support

# use random full context as support

# use random pair as support

class MetaRetrieved(Dataset):

    def __init__(self, filename, n_retrieved=1):
        
        self.filename = filename
        self.n_retrieved = n_retrieved
        self.chunksize = 1 # more than this requires reshaping in this class or collate function
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.max_dim = MAX_LINE_LENGTH
        self.len = int(temp.values[0][0] / self.chunksize)
        self.num_cols = self.n_retrieved*2+2
        
    def __len__(self):
        return self.len

    def read_pandas_line_quote(self, idx):
        return next(pd.read_csv(self.filename, 
                skiprows=idx*self.chunksize+1,
                chunksize=self.chunksize, header=None,
                sep=',\s+', quoting=csv.QUOTE_ALL, dtype=str)).fillna(NO_CONTEXT_WORD).values

    def read_pandas_line(self, idx):
        return next(pd.read_csv(self.filename,
            skiprows=idx*self.chunksize+1,
            chunksize=self.chunksize,
            header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values

    def __getitem__(self, idx):
        try:
            x = self.read_pandas_line(idx)
            
            # something is broken here so just give filler
            if len(x[0]) != self.num_cols:
                return self.__getitem__(np.random.randint(0, self.len))
        except:
            x = self.read_pandas_line_quote(idx)

            x = np.array(fix_quote_strings_context(x[0, 0], self.n_retrieved))
        
        query_x = [word2idx.get(token, UNKNOWN_IDX) for token in preprocess_tokens(tokenize_fine_grained(x[0, 0]), self.max_dim)]

        support_list_x = []
        support_list_y = []
        for i in range(self.n_retrieved):
            support_list_x.append([word2idx.get(token, UNKNOWN_IDX) for token in preprocess_tokens(tokenize_fine_grained(x[0, i*2+2]), self.max_dim)])
            support_list_y.append([word2idx.get(token, UNKNOWN_IDX) for token in preprocess_tokens(tokenize_fine_grained(x[0, i*2+3]), self.max_dim)])

        query_y = [word2idx.get(token, UNKNOWN_IDX) for token in preprocess_tokens(tokenize_fine_grained(x[0, 1]), self.max_dim)]

        support_x = torch.LongTensor(pd.DataFrame(support_list_x).values.astype('int64'))
        support_y = torch.LongTensor(pd.DataFrame(support_list_y).values.astype('int64'))

        query_x = torch.LongTensor(pd.DataFrame(query_x).values.astype('int64')).contiguous().view(1, -1)
        query_y = torch.LongTensor(pd.DataFrame(query_y).values.astype('int64')).contiguous().view(1, -1)
        
        return support_x, support_y, query_x, query_y


class MetaRepo(Dataset):

    def __init__(self, filename, retrieve_context=True, n_retrieved=1, k_shot=5, query_batch_size=10):
        
        self.filename = filename
        self.n_retrieved = n_retrieved
        self.retrieve_context = retrieve_context
        self.chunksize = 1 # more than this requires reshaping in this class or collate function
        temp_lines = int(next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None)).values[0][0])

        self.split_idx = int(temp_lines - temp_lines/(k_shot-2))

        self.query_indices = list(range(self.split_idx, temp_lines))
        shuffle(self.query_indices)
        self.query_indices = self.query_indices + self.query_indices[:int(len(self.query_indices)/2)]
        shuffle(self.query_indices)
        # self.

        self.len = int(len(self.query_indices)/query_batch_size)
        self.max_dim = MAX_LINE_LENGTH

        self.num_cols = self.n_retrieved*2+2 if retrieve_context else 2
        self.k_shot = k_shot
        self.query_batch_size = query_batch_size

        # split up repo into support and query parts
        # change getting them to those things
        
    def __len__(self):
        return self.len

    def read_pandas_line_quote(self, idx):
        return next(pd.read_csv(self.filename, 
                skiprows=idx*self.chunksize+1,
                chunksize=self.chunksize, header=None,
                sep=',\s+', quoting=csv.QUOTE_ALL, dtype=str)).fillna(NO_CONTEXT_WORD).values

    def read_pandas_line(self, idx):
        return next(pd.read_csv(self.filename,
            skiprows=idx*self.chunksize+1,
            chunksize=self.chunksize,
            header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values

    def words2tokens(self, x):
        x_tokens = preprocess_context(x, self.n_retrieved, self.max_dim) if self.retrieve_context else preprocess_tokens(tokenize_fine_grained(x[0, 0]), self.max_dim)
        y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 1]), self.max_dim)

        x_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in x_tokens]
        y_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in y_tokens]

        return x_tokens, y_tokens

    # def get_line(self, idx, not_idx, data_list):
    def get_line(self, idx, data_list, query=False):
        # if idx == not_idx:
            # return self.get_line(np.random.randint(0, self.len), not_idx)
        try:
            x = self.read_pandas_line(idx)

            if len(x[0]) != self.num_cols:
                if query:
                    idx = self.query_indices.pop(0)
                else:
                    idx = np.random.randint(0, self.split_idx)

                self.get_line(idx, data_list)
                return
        except:
            x = self.read_pandas_line_quote(idx)

            x = np.array(fix_quote_strings(x[0, 0]) if self.retrieve_context else fix_quote_strings_context(x[0, 0], self.n_retrieved))

        x_tokens, y_tokens = self.words2tokens(x)

        data_list.append((x_tokens, y_tokens))

    def __getitem__(self, idx):
        support_list = []
        query_list = []
        data_threads = []
        idx += self.split_idx
        support_indices = np.random.randint(0, self.split_idx, self.k_shot)

        for support_idx in support_indices:
            data_threads.append(Thread(target=self.get_line, args=(support_idx, support_list, False)))
            data_threads[-1].start()

        for query_idx in range(self.query_batch_size):
            data_threads.append(Thread(target=self.get_line, args=(self.query_indices.pop(0), query_list, True)))
            data_threads[-1].start()

        # try:
        #     x = self.read_pandas_line(idx)
            
        #     # something is broken here so just give filler
        #     if len(x[0]) != self.num_cols:
        #         idx = max(0, idx-1)
        #         return self.__getitem__(self.len-1 if idx == 0 else idx)
        # except:
        #     x = self.read_pandas_line_quote(idx)

        #     x = np.array(fix_quote_strings(x[0, 0]) if self.retrieve_context else fix_quote_strings_context(x[0, 0]))

        # query_x, query_y = self.words2tokens(x)

        for dt in data_threads:
            dt.join()

        query_x, query_y = zip(*query_list)
        support_x, support_y = zip(*support_list)
        # support_x = torch.LongTensor(pd.DataFrame(support_x).values.astype('int64'))
        # support_y = torch.LongTensor(pd.DataFrame(support_x).values.astype('int64'))


        # query_x = torch.LongTensor(pd.DataFrame(query_x).values.astype('int64')).contiguous().view(1, -1)
        # query_y = torch.LongTensor(pd.DataFrame(query_y).values.astype('int64')).contiguous().view(1, -1)
        
        support_x = pd.DataFrame(support_x).values.astype('int64')
        support_y = pd.DataFrame(support_x).values.astype('int64')

        query_x = pd.DataFrame(query_x).values.astype('int64')#.reshape(1, -1)
        query_y = pd.DataFrame(query_y).values.astype('int64')#.reshape(1, -1)

        return support_x, support_y, query_x, query_y








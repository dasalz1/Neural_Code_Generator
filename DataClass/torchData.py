import pandas as pd
import torch, os, pickle
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from DataClass.data_utils import tokenize_fine_grained, create_vocab_dictionary
import numpy as np
from DataClass.Constants import NO_CONTEXT_WORD, UNKNOWN_IDX, PAD_WORD, START_WORD, END_WORD, SEP_CONTEXT_WORD, SEP_PAIR_WORD, SEP_RET_WORD

tokens_file = './repo_files/all_tokens.pickle'
tokens_dict = pickle.load(open(tokens_file, 'rb'))
word2idx, idx2word = create_vocab_dictionary(tokens_dict)
# UNKNOWN_IDX = word2idx[UNKNOWN_WORD]
MAX_LINE_LENGTH = 128


def preprocess_tokens(tokens, max_dim):
    tokens = [START_WORD] + tokens
    n = len(tokens) + 1
    # minus one since end word needs to go on too
    tokens = tokens[:min(n, max_dim-1)] + [END_WORD] + [PAD_WORD]*max(0, max_dim-n)
    return tokens

def preprocess_context(context, n, max_dim):
    context_tokens = preprocess_tokens(tokenize_fine_grained(context[0, 0]), max_dim)
    context_tokens += [SEP_CONTEXT_WORD]
    
    for idx in range(n-1):
        context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, idx*2-1]), max_dim) + [SEP_PAIR_WORD]
        context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, idx*2]), max_dim) + [SEP_RET_WORD]
    
    context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, -2]), max_dim) + [SEP_PAIR_WORD]
    context_tokens += preprocess_tokens(tokenize_fine_grained(context[0, -1]), max_dim)
    return context_tokens


class PairDataset(Dataset):

    def __init__(self, filename):
        
        self.filename = filename
        self.chunksize = 1 # more than this requires reshaping in this class or collate function
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.max_dim = MAX_LINE_LENGTH
        self.len = int(temp.values[0][0] / self.chunksize)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = next(pd.read_csv(self.filename,
                            skiprows=idx * self.chunksize+1,
                            chunksize=self.chunksize, header=None)).fillna(NO_CONTEXT_WORD).values

        x_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 0]), self.max_dim)
        y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 1]), self.max_dim)
        return x_tokens, y_tokens
        # return np.array(x_tokens).reshape(-1, 1), np.array(y_tokens).reshape(-1, 1)

class RetrieveDataset(Dataset):

    def __init__(self, filename, chunksize, n_retrieved):
        
        self.filename = filename
        self.chunksize = 1 # more than this and requires a lot more changes in collate fn
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.n_retrieved = n_retrieved
        self.max_dim = MAX_LINE_LENGTH
        self.len = int(temp.values[0][0] / self.chunksize)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = next(pd.read_csv(self.filename,
                            skiprows=idx * self.chunksize+1,
                            chunksize=self.chunksize, header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values

        context_tokens = preprocess_context(x[:, :-1], self.n_retrieved, self.max_dim)
        y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, -1]), self.max_dim)
        return context_tokens, y_tokens
        # return np.array(x_tokens).reshape(-1, 1), np.array(y_tokens).reshape(-1, 1)



# for meta-learning, turn RetrieveDataset into dataloader and get list of dataloaders of which batch size is k-shot
    
def batch_collate_fn(data):
        x, y = zip(*data)
        
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        
        x = np.where(x.isin(tokens_dict.keys()), x.replace(tokens_dict), UNKNOWN_IDX).astype('int64')
        y = np.where(y.isin(tokens_dict.keys()), y.replace(tokens_dict), UNKNOWN_IDX).astype('int64')

        batch_xs = torch.LongTensor(x)
        batch_ys = torch.LongTensor(y)
        return batch_xs, batch_ys
    
    

def createDataLoaderAllFiles(dataset_dir, dataset_class=PairDataset, shuffle=True, batch_size=128):
    datasets = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(dataset_dir))[2]))
    return DataLoader(ConcatDataset([dataset_class(dataset_dir + '/' + dataset, chunksize=batch_size) for dataset in datasets]), 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=batch_collate_fn)





import pandas as pd
import torch, os, pickle, warnings, csv
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from DataClass.data_utils import tokenize_fine_grained, create_vocab_dictionary, preprocess_tokens, preprocess_context, fix_quote_strings
import numpy as np
from DataClass.Constants import NO_CONTEXT_WORD, UNKNOWN_IDX

warnings.simplefilter('ignore', pd.errors.ParserWarning)

tokens_file = '../all_tokens.pickle'
# tokens_dict = pickle.load(open(tokens_file, 'rb'))
word2idx, idx2word = create_vocab_dictionary(path='.', file='all_tokens.pickle', save=True)

# UNKNOWN_IDX = word2idx[UNKNOWN_WORD]
MAX_LINE_LENGTH = 128

class PairDataset(Dataset):

    def __init__(self, filename):
        super(PairDataset).__init__()
        self.filename = filename
        self.chunksize = 1 # more than this requires reshaping in this class or collate function
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.max_dim = MAX_LINE_LENGTH
        self.len = int(temp.values[0][0] / self.chunksize)
        self.num_cols = 2
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            x = next(pd.read_csv(self.filename,
                                skiprows=idx * self.chunksize+1,
                                chunksize=self.chunksize, header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values
            
            
            # something is broken here so just give filler
            if len(x[0]) != self.num_cols:
                # idx = max(0, idx-1)
                return self.__getitem__(np.random.randint(0, self.len))
        except:
            x = next(pd.read_csv(self.filename,
                                skiprows=idx * self.chunksize+1,
                                chunksize=self.chunksize, header=None,
                                sep=',\s+', quoting=csv.QUOTE_ALL, dtype=str)).fillna(NO_CONTEXT_WORD).values

            x = np.array(fix_quote_strings(x[0, 0]))

        x_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 0]), self.max_dim)
        y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, 1]), self.max_dim)

        x_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in x_tokens]
        y_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in y_tokens]
        
        return x_tokens, y_tokens

class RetrieveDataset(Dataset):

    def __init__(self, filename, chunksize, n_retrieved):
        super(RetrieveDataset).__init__()

        self.filename = filename
        self.chunksize = 1 # more than this and requires a lot more changes in collate fn
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.n_retrieved = n_retrieved
        self.max_dim = MAX_LINE_LENGTH
        self.len = int(temp.values[0][0] / self.chunksize)
        self.num_cols = self.n_retrieved*2+2
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        try:
            x = next(pd.read_csv(self.filename,
                            skiprows=idx * self.chunksize+1,
                            chunksize=self.chunksize, header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values

                    # something is broken here so just give filler
            if len(x[0]) != self.num_cols:
                # idx = max(0, idx-1)
                return self.__getitem__(np.random.randint(0, self.len))
        except:
            x = next(pd.read_csv(self.filename,
                                skiprows=idx * self.chunksize+1,
                                chunksize=self.chunksize, header=None,
                                sep=',\s+', quoting=csv.QUOTE_ALL, dtype=str)).fillna(NO_CONTEXT_WORD).values

            x = np.array(fix_quote_strings_context(x[0, 0], self.n_retrieved))

        context_tokens = preprocess_context(x, self.n_retrieved, self.max_dim)

        y_tokens = preprocess_tokens(tokenize_fine_grained(x[0, -1]), self.max_dim)

        context_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in context_tokens]
        y_tokens = [word2idx.get(token, UNKNOWN_IDX) for token in y_tokens]

        return context_tokens, y_tokens



# for meta-learning, turn RetrieveDataset into dataloader and get list of dataloaders of which batch size is k-shot
    
def batch_collate_fn(data):
        # data = list(filter(lambda z : z is not None, data))
        x, y = zip(*data)
        
        x = pd.DataFrame(x).values.astype('int64')
        y = pd.DataFrame(y).values.astype('int64')
        
        # x = np.where(x.isin(word2idx.keys()), x.replace(word2idx), UNKNOWN_IDX).astype('int64')
        # y = np.where(y.isin(word2idx.keys()), y.replace(word2idx), UNKNOWN_IDX).astype('int64')

        batch_xs = torch.LongTensor(x)
        batch_ys = torch.LongTensor(y)
        return batch_xs, batch_ys
    
    

def createDataLoaderAllFiles(dataset_dir, dataset_class=PairDataset, shuffle=True, batch_size=128):
    datasets = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(dataset_dir))[2]))
    return DataLoader(ConcatDataset([dataset_class(dataset_dir + '/' + dataset, chunksize=batch_size) for dataset in datasets]), 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=batch_collate_fn)





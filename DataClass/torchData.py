import pandas as pd
import torch, os, pickle
from torch.utils.data import Dataset, Dataset, ConcatDataset, DataLoader
from DataClass.data_utils import tokenize_fine_grained



NO_CONTEXT_WORD = 'OSOFo'
PAD_WORD = '<PAD>'
START_WORD = '<BOS>'
END_WORD = '<EOS>'

tokens_file = './DataClass/token_dict.pickle'
tokens_dict = pickle.load(open(tokens_file, 'rb'))
tokens_dict[NO_CONTEXT_WORD] = len(tokens_dict)
tokens_dict[PAD_WORD] = len(tokens_dict)
tokens_dict[START_WORD] = len(tokens_dict)
tokens_dict[END_WORD] = len(tokens_dict)
PAD_IDX = tokens_dict[PAD_WORD]
BOS_IDX = tokens_dict[START_WORD]
EOS_IDX = tokens_dict[END_WORD]



class PairDatasetLazy(Dataset):

    def __init__(self, filename, chunksize):
        
        self.filename = filename
        self.chunksize = 1 # more than this and requires a lot more changes in collate fn
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.len = int(temp.values[0][0] / self.chunksize)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = next(pd.read_csv(self.filename,
                            skiprows=idx * self.chunksize+1,
                            chunksize=self.chunksize, header=None, dtype=str)).fillna(NO_CONTEXT_WORD).values
        
        x_tokens = [START_WORD] + tokenize_fine_grained(x[0, 0]) + [END_WORD]
        y_tokens = [START_WORD] + tokenize_fine_grained(x[0, 1]) + [END_WORD]

        return x_tokens, y_tokens
    
    
def batch_collate_fn(data):
        x, y = zip(*data)
        
        x = pd.DataFrame(x).replace(tokens_dict).fillna(PAD_IDX)
        y = pd.DataFrame(y).replace(tokens_dict).fillna(PAD_IDX)
        batch_xs = torch.LongTensor(x.values)
        batch_ys = torch.LongTensor(y.values)
        return batch_xs, batch_ys
    
    

def createDataLoaderAllFiles(dataset_dir, dataset_class=PairDatasetLazy, shuffle=True, batch_size=128):
    datasets = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(dataset_dir))[2]))
    return DataLoader(ConcatDataset([dataset_class(dataset_dir + '/' + dataset, chunksize=batch_size) for dataset in datasets]), 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        collate_fn=batch_collate_fn)





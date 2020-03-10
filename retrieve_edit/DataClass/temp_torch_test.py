from torch.utils.data import Dataset, Dataset, ConcatDataset, DataLoader
import pandas as pd
import torch, os


class PairDatasetLazy(Dataset):

    def __init__(self, filename, chunksize):
        
        self.filename = filename
        self.chunksize = chunksize
        temp = next(pd.read_csv(self.filename, skiprows = 0, chunksize=1, header=None))
        self.len = int(temp.values[0][0] / self.chunksize)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = next(pd.read_csv(self.filename,
                            skiprows=idx * self.chunksize+1,
                            chunksize=self.chunksize, header=None, dtype=str)).fillna('OSOFo').values
        
        x_tokens = tokenize_fine_grained(x[0, 0])# for idx in range(self.chunksize)]
        y_tokens = tokenize_fine_grained(x[0, 1])# for idx in range(self.chunksize)]

        return x_tokens, y_tokens

# dataset dir is directory assuming new subdirectories
def createDataLoaderAllFiles(dataset_dir, dataset_class=PairDatasetLazy, shuffle=True, batch_size=128):
    datasets = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(dataset_dir))[2]))
    return DataLoader(ConcatDataset([dataset_class(dataset_dir + '/' + dataset, chunksize=batch_size) for dataset in datasets]), batch_size=batch_size, shuffle=shuffle)
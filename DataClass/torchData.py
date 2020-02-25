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
                            chunksize=self.chunksize, header=None, dtype=str)).fillna('OSOFo').values
        
        x_tokens = tokenize_fine_grained(x[0, 0])
        y_tokens = tokenize_fine_grained(x[0, 1])

        return x_tokens, y_tokens
    
    
def batch_collate_fn(data):
        x, y = zip(*data)
        # off by one in tokens dictionary
        x = pd.DataFrame(x).replace(token_dict).fillna(-1)+1
        y = pd.DataFrame(y).replace(token_dict).fillna(-1)+1
        batch_xs = torch.LongTensor(x.values)
        batch_ys = torch.LongTensor(y.values)
        return batch_xs, batch_ys
    
    

def createDataLoaderAllFiles(dataset_dir, dataset_class=PairDatasetLazy, shuffle=True, batch_size=128):
    datasets = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(dataset_dir))[2]))
    return DataLoader(ConcatDataset([dataset_class(dataset_dir + '/' + dataset, chunksize=batch_size) for dataset in datasets]), batch_size=batch_size, shuffle=shuffle, collate_fn=batch_collate_fn)
import pickle, pathlib2
from DataClass import data_utils

word2idx, idx2word = data_utils.create_vocab_dictionary(path = '.', file = 'all_tokens.pickle', save = True)
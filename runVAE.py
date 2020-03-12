from tensorboard_utils import Tensorboard
from DataClass.torchData import *
from DataClass.Constants import PAD_IDX, END_IDX, END_IDX, UNKNOWN_IDX, START_IDX
from DataClass.torchData import MAX_LINE_LENGTH
from VAE.model import *
from VAETrainer import VAETrainer
from torch.utils.data import ConcatDataset, DataLoader
import torch
from random import shuffle
from datetime import date
import argparse, random
import numpy as np
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument("--filepath", default='../repo_files', type=str)
parser.add_argument("--exp_name", default='EditorPairTrain', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--key_dimension", default=64, type=int)
parser.add_argument("--value_dimension", default=64, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--d_word_vec", default=512, type=int)
parser.add_argument("--inner_dimension", default=2048, type=int)
parser.add_argument("--batch_size", default=24, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--use_retriever", default=False, action='store_true')
parser.add_argument("--n_retrieved", default=1, type=int)
parser.add_argument("--n_src_length", default=MAX_LINE_LENGTH, type=int)
parser.add_argument("--n_trg_length", default=MAX_LINE_LENGTH, type=int)
args = parser.parse_args()

def main(args):
	random.seed(68492)
	np.random.seed(68492)
	torch.manual_seed(68492)

	VOCAB_SIZE = len(word2idx)
	num_validation_repos = 50

	if args.use_retriever:
		args.n_src_length = (args.n_src_length+1) + args.n_retrieved*2*args.n_src_length + (2*args.n_retrieved-1)
		print(args.n_src_length)

	if args.use_retriever: args.exp_name += ("retrieved_%d" % args.n_retrieved)

	tb = Tensorboard(args.exp_name, unique_name=args.unique_id)

	repo_files = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(args.filepath))[2]))
	# shuffle(repo_files)

	if args.use_retriever:
		datasets = ConcatDataset([RetrieveDataset(args.filepath +'/'+dataset, args.n_retrieved) for dataset in repo_files[num_validation_repos:]])
		validsets = ConcatDataset([RetrieveDataset(args.filepath +'/'+dataset, args.n_retrieved) for dataset in repo_files[:num_validation_repos]])
	else:
		datasets = ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[num_validation_repos:60]])
		validsets = ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[:num_validation_repos]])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data_loader = DataLoader(datasets,
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn)#,
							# num_workers=92)

	print("Finished creating data loader")

	validation_loader = DataLoader(validsets,
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn)#,
							# num_workers=92)

	print("Finished creating validation data loader")
	num_iterations = len(data_loader)

	model = SentenceVAE(vocab_size=VOCAB_SIZE,
					sos_idx = START_IDX, eos_idx=END_IDX, pad_idx = PAD_IDX, unk_idx = UNKNOWN_IDX,
					max_sequence_length=MAX_LINE_LENGTH,
					embedding_size=args.d_word_vec, rnn_type='gru',
					hidden_size=args.inner_dimension, word_dropout=0.0,
					embedding_dropout=0.5, latent_size=int(args.d_word_vec/2),
					bidirectional=True)

	if torch.cuda.is_available:
		torch.backends.cudnn.deterministic=True
		torch.backends.cudnn.benchmark = False
		
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs...")
		model = torch.nn.DataParallel(model)
	
	model.to(device)

	trainer = VAETrainer(device)

	trainer.train(model, data_loader, validation_loader, tb=tb, epochs=args.epochs)

if __name__=='__main__':
	main(args)